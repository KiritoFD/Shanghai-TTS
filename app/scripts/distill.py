from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


SYSTEM_PROMPT = (
    "你是中文查询规范化助手。"
    "给定用户输入后，只输出一个JSON对象，字段必须为："
    "core_text,type,predicate,object,keywords。"
    "type只能是“词项”或“动作短语”。"
    "keywords是中文词列表，不要输出任何解释。"
)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B distill model (QLoRA)")
    parser.add_argument("--config", default=None, type=str, help="JSON config path")
    parser.add_argument("--train_jsonl", default=str(base_dir / "train_sft.jsonl"), type=str)
    parser.add_argument("--eval_jsonl", default=str(base_dir / "dev_sft.jsonl"), type=str)
    parser.add_argument("--model_name_or_path", default=str(base_dir / "Qwen3.5-2B"), type=str)
    parser.add_argument("--output_dir", default=str(base_dir / "model_lora"), type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=float)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--save_steps", default=200, type=int)
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--eval_predict_jsonl", default=str(base_dir / "test_clean_200.jsonl"), type=str)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        type=str,
        help="Checkpoint path to resume from. Use 'auto' to pick latest checkpoint in output_dir, or 'none' to disable.",
    )
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = vars(args).copy()
    for key, value in cfg.items():
        if key in data:
            data[key] = value
    return argparse.Namespace(**data)


def extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return "{}"


def safe_json_loads(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_f1(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g:
        return 0.0, 0.0, 0.0
    inter = len(g & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def render_final_output(example: dict) -> str:
    type_name = str(example.get("type", "")).strip()
    keywords = [str(x).strip() for x in example.get("keywords", []) if str(x).strip()]
    title = "【短语】" if type_name == "动作短语" else "【单词】"
    return f"{title}\n{','.join(keywords)}"


def run_eval_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    max_new_tokens: int = 128,
) -> None:
    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    predictions: list[dict] = []

    for row in rows:
        query = str(row.get("query", "")).strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户输入：{query}"},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{SYSTEM_PROMPT}\n用户输入：{query}\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed_text = extract_json(response)
        parsed = safe_json_loads(parsed_text)
        pred_final_output = render_final_output(parsed) if parsed else ""
        gold_final_output = str(row.get("final_output", "")).strip()

        predictions.append(
            {
                "query": query,
                "raw_response": response,
                "parsed_text": parsed_text,
                "gold_final_output": gold_final_output,
                "pred_final_output": pred_final_output,
                "text_exact_match": pred_final_output == gold_final_output,
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    pred_map = {item["query"]: item for item in predictions}
    n = 0
    valid_json = 0
    type_correct = 0
    core_em = 0
    pred_em = 0
    obj_em = 0
    text_em = 0
    kw_p = 0.0
    kw_r = 0.0
    kw_f1 = 0.0

    for gold in rows:
        query = str(gold.get("query", "")).strip()
        pred_row = pred_map.get(query)
        if not pred_row:
            continue
        n += 1

        parsed = safe_json_loads(str(pred_row.get("parsed_text", "{}")))
        if parsed:
            valid_json += 1

        if str(parsed.get("type", "")).strip() == str(gold.get("type", "")).strip():
            type_correct += 1
        if str(parsed.get("core_text", "")).strip() == str(gold.get("core_text", "")).strip():
            core_em += 1
        if str(parsed.get("object", "")).strip() == str(gold.get("object", "")).strip():
            obj_em += 1

        g_keywords = [str(x).strip() for x in gold.get("keywords", []) if str(x).strip()]
        p_keywords = [str(x).strip() for x in parsed.get("keywords", []) if str(x).strip()]
        p_, r_, f_ = set_f1(g_keywords, p_keywords)
        kw_p += p_
        kw_r += r_
        kw_f1 += f_

        gold_struct = {
            "core_text": str(gold.get("core_text", "")).strip(),
            "type": str(gold.get("type", "")).strip(),
            "predicate": str(gold.get("predicate", "")).strip(),
            "object": str(gold.get("object", "")).strip(),
            "keywords": sorted(set(g_keywords)),
        }
        pred_struct = {
            "core_text": str(parsed.get("core_text", "")).strip(),
            "type": str(parsed.get("type", "")).strip(),
            "predicate": str(parsed.get("predicate", "")).strip(),
            "object": str(parsed.get("object", "")).strip(),
            "keywords": sorted(set(p_keywords)),
        }
        if gold_struct == pred_struct:
            pred_em += 1
        if pred_row.get("text_exact_match", False):
            text_em += 1

    denom = max(n, 1)
    report = {
        "eval_set": str(input_jsonl),
        "covered_samples": n,
        "valid_json_rate": valid_json / denom,
        "exact_match": pred_em / denom,
        "text_exact_match": text_em / denom,
        "type_accuracy": type_correct / denom,
        "core_text_em": core_em / denom,
        "object_em": obj_em / denom,
        "keywords_precision": kw_p / denom,
        "keywords_recall": kw_r / denom,
        "keywords_f1": kw_f1 / denom,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


class CheckpointEvalCallback(TrainerCallback):
    def __init__(self, eval_input_jsonl: Path, tokenizer: AutoTokenizer, max_new_tokens: int = 128):
        self.eval_input_jsonl = eval_input_jsonl
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        if not self.eval_input_jsonl.exists():
            print(f"skip eval prediction: missing file {self.eval_input_jsonl}")
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        was_training = model.training
        run_eval_predictions(
            model=model,
            tokenizer=self.tokenizer,
            input_jsonl=self.eval_input_jsonl,
            output_jsonl=checkpoint_dir / "eval_200_predictions.jsonl",
            report_json=checkpoint_dir / "eval_200_scores.json",
            max_new_tokens=self.max_new_tokens,
        )
        if was_training:
            model.train()
        return control


def main() -> None:
    args = load_config(parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl and Path(args.eval_jsonl).exists():
        data_files["validation"] = args.eval_jsonl
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_messages(messages: list[dict]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)

    def preprocess(row: dict) -> dict:
        text = format_messages(row["messages"])
        encoded = tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        encoded["labels"] = encoded["input_ids"][:]
        return encoded

    tokenized_train = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
    tokenized_eval = None
    if "validation" in dataset:
        tokenized_eval = dataset["validation"].map(preprocess, remove_columns=dataset["validation"].column_names)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    has_eval = tokenized_eval is not None
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_cuda = torch.cuda.is_available()

    train_batch_per_step = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    updates_per_epoch = max(1, math.ceil(len(tokenized_train) / train_batch_per_step))
    total_updates = max(1, math.ceil(updates_per_epoch * args.num_train_epochs))
    warmup_steps = int(total_updates * args.warmup_ratio)
    save_steps = max(1, math.ceil(updates_per_epoch * 0.5))

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "eval_steps": args.eval_steps if has_eval else None,
        "report_to": [],
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "seed": args.seed,
        "dataloader_pin_memory": use_cuda,
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = "steps" if has_eval else "no"
    elif "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

    if "warmup_steps" in ta_params:
        training_args_kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in ta_params:
        training_args_kwargs["warmup_ratio"] = args.warmup_ratio

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.add_callback(CheckpointEvalCallback(Path(args.eval_predict_jsonl), tokenizer=tokenizer, max_new_tokens=128))

    resume_from_checkpoint = None
    resume_arg = str(args.resume_from_checkpoint or "").strip()
    if resume_arg and resume_arg.lower() != "none":
        if resume_arg.lower() == "auto":
            resume_from_checkpoint = get_last_checkpoint(str(output_dir))
        else:
            candidate = Path(resume_arg)
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            if candidate.exists():
                resume_from_checkpoint = str(candidate)
            else:
                print(f"resume checkpoint not found, start from scratch: {candidate}")

    if resume_from_checkpoint:
        print(f"resume training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("start training from scratch")
        trainer.train()
    trainer.save_model(str(output_dir))

    (output_dir / "train_args_resolved.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

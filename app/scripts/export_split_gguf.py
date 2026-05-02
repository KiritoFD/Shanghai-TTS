from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
LLAMA_CPP_ROOT = REPO_ROOT / "_vendor" / "llama.cpp-src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split LoRA and export GGUF for llama.cpp")
    parser.add_argument("--base_model", default=str(REPO_ROOT / "model" / "Qwen3.5-2B"), type=str)
    parser.add_argument("--lora_dir", default=str(APP_ROOT / "model_lora" / "checkpoint-130"), type=str)
    parser.add_argument("--merged_dir", default=str(REPO_ROOT / "model" / "Qwen3.5-2B-split-lora-merged"), type=str)
    parser.add_argument("--gguf_out", default=str(REPO_ROOT / "model" / "Qwen3.5-2B-split-lora-f16.gguf"), type=str)
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "q8_0"], type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model).resolve()
    lora_dir = Path(args.lora_dir).resolve()
    merged_dir = Path(args.merged_dir).resolve()
    gguf_out = Path(args.gguf_out).resolve()
    convert_script = LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"

    if not base_model.exists():
        raise FileNotFoundError(base_model)
    if not lora_dir.exists():
        raise FileNotFoundError(lora_dir)
    if not convert_script.exists():
        raise FileNotFoundError(convert_script)

    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_out.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    print(f"[export] loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"[export] merging LoRA: {lora_dir}")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()

    print(f"[export] saving merged HF model -> {merged_dir}")
    model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(merged_dir))

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(gguf_out),
        "--outtype",
        args.outtype,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    print("[export] converting HF -> GGUF")
    print("[export] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(LLAMA_CPP_ROOT), env=env, check=True)
    print(f"[export] done: {gguf_out}")


if __name__ == "__main__":
    main()

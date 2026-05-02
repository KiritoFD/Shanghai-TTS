from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cleaner_model import CoreTagger, CoreTransformerTagger, PAD, UNK, save_metadata


HAND_ROWS = [
    {"query": "那个... 麻烦帮我查一下，自行车坏了，怎么用上海话表达？", "core_text": "自行车坏了"},
    {"query": "我想问下，今天不想去上学上海话怎么讲", "core_text": "今天不想去上学"},
    {"query": "你好用上海话怎么说", "core_text": "你好"},
    {"query": "谢谢你上海话怎么讲", "core_text": "谢谢你"},
    {"query": "吃饭了吗怎么用上海话说", "core_text": "吃饭了吗"},
    {"query": "上海话里今天怎么讲", "core_text": "今天"},
    {"query": "请问不想去怎么讲", "core_text": "不想去"},
]

SHELL_TEMPLATES = [
    "{core}",
    "{core}怎么说",
    "{core}怎么讲",
    "{core}怎么表达",
    "{core}用上海话怎么说",
    "{core}用上海话怎么讲",
    "用上海话{core}怎么说",
    "用上海话{core}怎么讲",
    "上海话里{core}怎么说",
    "上海话里{core}怎么讲",
    "请问{core}怎么说",
    "请问{core}怎么讲",
    "帮我查一下{core}怎么说",
    "麻烦帮我查一下，{core}怎么用上海话表达",
    "{core}用中文怎么说",
    "{core}怎么讲更合适",
    "{core}怎么表达更自然",
    "表达{core}怎么说",
    "我想问下，{core}上海话怎么讲",
    "想问下我{core}怎么说",
    "想问下{core}我们怎么说",
    "麻烦问下我们已经{core}怎么说",
    "请看{core}是不是短语",
    "请看{core}是不是词组",
    "{core}这个词怎么讲",
    "{core}这个说法自然吗",
    "{core}算短语吗",
    "口语里一般怎么讲{core}",
    "平时怎么说{core}",
    "通知一下{core}怎么表达",
    "请看可能描述一下{core}是不是短语",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight query cleaner core extractor.")
    parser.add_argument("--split_jsonl", default="app/split/data/processed/query_train_teacher_5000_clean_all.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/cleaner_core/outputs/bilstm_core", type=str)
    parser.add_argument("--max_rows", default=5000, type=int)
    parser.add_argument("--synthetic_per_core", default=4, type=int)
    parser.add_argument("--max_length", default=96, type=int)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--model_type", default="bilstm", choices=["bilstm", "transformer"], type=str)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_heads", default=6, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def load_rows(path: Path, max_rows: int, synthetic_per_core: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = HAND_ROWS * 80
    cores: list[str] = [row["core_text"] for row in HAND_ROWS]
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = compact(row.get("query", ""))
                core = compact(row.get("core_text", ""))
                if query and core and core in query:
                    rows.append({"query": query, "core_text": core})
                    cores.append(core)
                elif core:
                    cores.append(core)

    unique_cores = list(dict.fromkeys(core for core in cores if core))
    rng.shuffle(unique_cores)
    for core in unique_cores:
        templates = SHELL_TEMPLATES[:]
        rng.shuffle(templates)
        for template in templates[:synthetic_per_core]:
            if len(rows) >= max_rows:
                break
            rows.append({"query": template.format(core=core), "core_text": core})
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]


def build_vocab(rows: list[dict]) -> dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    counts: dict[str, int] = {}
    for row in rows:
        for ch in compact(row["query"]):
            counts[ch] = counts.get(ch, 0) + 1
    for ch, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocab[ch] = len(vocab)
    return vocab


class CoreRows(Dataset):
    def __init__(self, rows: list[dict], vocab: dict[str, int], max_length: int) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        query = compact(row["query"])[: self.max_length]
        core = compact(row["core_text"])
        labels = [0] * len(query)
        start = query.find(core)
        if start >= 0:
            for pos in range(start, min(start + len(core), len(labels))):
                labels[pos] = 1
        ids = [self.vocab.get(ch, self.vocab[UNK]) for ch in query]
        return {"ids": ids, "labels": labels, "query": query, "core_text": core}


def collate(rows: list[dict]) -> dict:
    max_len = max(len(row["ids"]) for row in rows)
    input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for i, row in enumerate(rows):
        ids = torch.tensor(row["ids"], dtype=torch.long)
        y = torch.tensor(row["labels"], dtype=torch.long)
        input_ids[i, : len(ids)] = ids
        labels[i, : len(y)] = y
    return {"input_ids": input_ids, "labels": labels, "raw": rows}


def span_from_labels(text: str, preds: list[int]) -> str:
    best = ""
    cur = ""
    for ch, pred in zip(text, preds):
        if pred:
            cur += ch
        else:
            if len(cur) > len(best):
                best = cur
            cur = ""
    if len(cur) > len(best):
        best = cur
    return best.strip("，,。.!！？?；;：:、")


@torch.no_grad()
def evaluate(model: CoreTagger, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    exact = 0
    token_ok = 0
    token_total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        token_ok += int((preds[mask] == labels[mask]).sum().item())
        token_total += int(mask.sum().item())
        for row, pred in zip(batch["raw"], preds.cpu().tolist()):
            got = span_from_labels(row["query"], pred[: len(row["query"])])
            exact += int(got == row["core_text"])
            total += 1
    return {
        "core_exact": exact / max(total, 1),
        "token_acc": token_ok / max(token_total, 1),
        "samples": total,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(Path(args.split_jsonl), args.max_rows, args.synthetic_per_core, args.seed)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split]
    dev_rows = rows[split:]
    vocab = build_vocab(train_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "transformer":
        model = CoreTransformerTagger(
            len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_length=args.max_length,
        ).to(device)
    else:
        model = CoreTagger(
            len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(CoreRows(train_rows, vocab, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(CoreRows(dev_rows, vocab, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    best = {"core_exact": -1.0}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, dev_loader, device)
        metrics["loss"] = sum(losses) / max(len(losses), 1)
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False))
        if metrics["core_exact"] >= best["core_exact"]:
            best = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "max_length": args.max_length,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "model_type": args.model_type,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "metrics": best,
                },
                output_dir / "model.pt",
            )
    save_metadata(output_dir, {"train_rows": len(train_rows), "dev_rows": len(dev_rows), "best": best, "args": vars(args)})
    print(json.dumps({"saved": str(output_dir), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

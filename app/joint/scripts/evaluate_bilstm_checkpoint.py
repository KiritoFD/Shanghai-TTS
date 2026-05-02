from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_bilstm_joint import (
    BiLSTMJoint,
    TAG_TO_ID,
    clean_text,
    encode_chars,
    encode_texts,
    evaluate_recall,
    read_jsonl,
)


ID_TO_TAG = {value: key for key, value in TAG_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BiLSTM joint checkpoint")
    parser.add_argument("--checkpoint_dir", default="app/joint/outputs/bilstm_joint/final", type=str)
    parser.add_argument("--test_jsonl", default="app/joint/data/test.jsonl", type=str)
    parser.add_argument("--corpus_jsonl", default="app/joint/data/corpus.jsonl", type=str)
    parser.add_argument("--output_json", default="", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_length", default=96, type=int)
    return parser.parse_args()


def load_model(checkpoint_dir: Path, device: torch.device) -> tuple[BiLSTMJoint, dict[str, int], dict]:
    args = json.loads((checkpoint_dir / "training_args.json").read_text(encoding="utf-8"))
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
    model = BiLSTMJoint(
        vocab_size=len(vocab),
        embedding_dim=int(args["embedding_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        projection_dim=int(args["projection_dim"]),
        num_layers=int(args["num_layers"]),
        dropout=float(args["dropout"]),
    )
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, vocab, args


@torch.no_grad()
def evaluate_segmentation(model: BiLSTMJoint, rows: list[dict], vocab: dict[str, int], max_length: int, device: torch.device) -> dict:
    total = 0
    correct = 0
    exact = 0
    covered = 0
    for row in rows:
        text = clean_text(row.get("segment_text") or row.get("headword") or "")[:max_length]
        gold = [str(tag) for tag in row.get("segment_tags", [])[: len(text)]]
        if not text or not gold:
            continue
        ids = torch.tensor([encode_chars(text, vocab, max_length)], dtype=torch.long, device=device)
        lengths = torch.tensor([ids.size(1)], dtype=torch.long, device=device)
        logits = model.segment_logits(ids, lengths)[0, : len(gold)]
        pred = [ID_TO_TAG[int(i)] for i in logits.argmax(dim=-1).detach().cpu().tolist()]
        covered += 1
        exact += int(pred == gold)
        for p, g in zip(pred, gold):
            total += 1
            correct += int(p == g)
    return {
        "seg_count": covered,
        "tag_accuracy": correct / max(total, 1),
        "seg_exact_match": exact / max(covered, 1),
    }


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    test_rows = read_jsonl(Path(args.test_jsonl))
    corpus = read_jsonl(Path(args.corpus_jsonl))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, train_args = load_model(checkpoint_dir, device)
    max_length = int(train_args.get("max_length", args.max_length))
    batch_size = int(args.batch_size)
    seg = evaluate_segmentation(model, test_rows, vocab, max_length, device)
    recall = evaluate_recall(model, test_rows, corpus, vocab, max_length, batch_size, device, limit=0)
    report = {"checkpoint_dir": str(checkpoint_dir), "segmentation": seg, "retrieval": recall}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


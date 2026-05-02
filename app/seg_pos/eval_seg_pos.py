"""Evaluate a trained BiLSTM-CRF segmentation + POS model on test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEG_ID_TO_TAG = {0: "B", 1: "M", 2: "E", 3: "S"}
POS_TAGS = ["N", "V", "A", "M", "Q", "R", "D", "P", "C", "SP", "AS", "Y", "FW", "I", "O", "IDM", "vn", "nd", "X"]
POS_ID_TO_TAG = {i: t for i, t in enumerate(POS_TAGS)}


def tags_to_words(tags):
    words, start = [], 0
    for i, t in enumerate(tags):
        if t in ("S", "E"):
            words.append((start, i + 1))
            start = i + 1
    return words


def compute_seg_f1(pred_tags_list, gold_tags_list):
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_tags_list, gold_tags_list):
        pw = set(tuple(w) for w in tags_to_words(pred))
        gw = set(tuple(w) for w in tags_to_words(gold))
        tp += len(pw & gw)
        fp += len(pw - gw)
        fn += len(gw - pw)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"seg_precision": prec, "seg_recall": rec, "seg_f1": f1}


@torch.no_grad()
def evaluate_full(model, loader, device, use_crf):
    model.eval()
    all_pred_seg, all_gold_seg = [], []
    seg_correct, seg_total = 0, 0
    pos_correct, pos_total = 0, 0
    pos_confusion = {}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        seg_ids = batch["seg_ids"]
        seg_len = batch["seg_len"]
        seg_labels = batch["seg_labels"]
        pos_labels = batch["pos_labels"]

        if use_crf:
            emissions = model.seg_emissions(seg_ids, seg_len)
            for b in range(seg_ids.size(0)):
                L = int(seg_len[b].item())
                if L == 0:
                    continue
                em_b = emissions[b:b+1, :L]
                mask_b = torch.ones(1, L, dtype=torch.bool, device=device)
                decoded = model.crf.decode(em_b, mask_b)[0]
                gold_seg = seg_labels[b, :L].clamp(min=0).tolist()
                gold_pos = pos_labels[b, :L].tolist()
                pred_tags = [SEG_ID_TO_TAG.get(t, "S") for t in decoded]
                gold_tags = [SEG_ID_TO_TAG.get(t, "S") for t in gold_seg]
                all_pred_seg.append(pred_tags)
                all_gold_seg.append(gold_tags)
                for p, g in zip(decoded, gold_seg):
                    seg_total += 1
                    if p == g:
                        seg_correct += 1
                pos_embs = model.contextualize(seg_ids[b:b+1], seg_len[b:b+1])
                pos_logits = model.pos_classifier(pos_embs)[0, :L].argmax(-1).tolist()
                for p_id, g_id in zip(pos_logits, gold_pos):
                    if g_id == -100:
                        continue
                    pos_total += 1
                    if p_id == g_id:
                        pos_correct += 1
                    key = (POS_ID_TO_TAG.get(g_id, "?"), POS_ID_TO_TAG.get(p_id, "?"))
                    pos_confusion[key] = pos_confusion.get(key, 0) + 1
        else:
            seg_logits = model.seg_proj(model.contextualize(seg_ids, seg_len))
            seg_preds = seg_logits.argmax(-1)
            seg_mask = seg_labels != -100
            seg_correct += (seg_preds[seg_mask] == seg_labels[seg_mask]).sum().item()
            seg_total += seg_mask.sum().item()
            pos_logits = model.pos_classifier(model.contextualize(seg_ids, seg_len))
            pos_preds = pos_logits.argmax(-1)
            pos_mask = pos_labels != -100
            pos_correct += (pos_preds[pos_mask] == pos_labels[pos_mask]).sum().item()
            pos_total += pos_mask.sum().item()

    seg_acc = seg_correct / max(1, seg_total)
    pos_acc = pos_correct / max(1, pos_total)
    f1_metrics = compute_seg_f1(all_pred_seg, all_gold_seg) if all_pred_seg else {}

    per_pos_acc = {}
    for (gold_tag, pred_tag), count in sorted(pos_confusion.items()):
        if gold_tag not in per_pos_acc:
            per_pos_acc[gold_tag] = {"correct": 0, "total": 0}
        per_pos_acc[gold_tag]["total"] += count
        if gold_tag == pred_tag:
            per_pos_acc[gold_tag]["correct"] += count

    return {
        "seg_acc": round(seg_acc, 4),
        "pos_acc": round(pos_acc, 4),
        **{k: round(v, 4) for k, v in f1_metrics.items()},
        "pos_per_tag": {
            tag: round(d["correct"] / max(1, d["total"]), 4)
            for tag, d in sorted(per_pos_acc.items())
        },
        "pos_per_tag_counts": {
            tag: d["total"] for tag, d in sorted(per_pos_acc.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    meta = json.loads((ckpt_dir / "training_args.json").read_text(encoding="utf-8"))
    vocab = json.loads((ckpt_dir / "vocab.json").read_text(encoding="utf-8"))
    use_crf = meta.get("use_crf", False)

    sys_path = str(Path(__file__).parent)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    if use_crf:
        try:
            from train_bilstm_seg_pos_v3 import BiLSTMCRFSegPos, read_jsonl, make_collate, SegPosRows
        except ImportError:
            from train_bilstm_seg_pos_v2 import BiLSTMCRFSegPos, read_jsonl, make_collate, SegPosRows
    else:
        from train_bilstm_seg_pos import BiLSTMCRFSegPos, read_jsonl, make_collate, SegPosRows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRFSegPos(
        vocab_size=len(vocab),
        embedding_dim=meta.get("embedding_dim", 256),
        hidden_dim=meta.get("hidden_dim", 256),
        projection_dim=meta.get("projection_dim", 256),
        num_layers=meta.get("num_layers", 2),
        dropout=0.0,
        use_crf=use_crf,
        label_smoothing=meta.get("label_smoothing", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()

    test_rows = read_jsonl(Path(meta.get("test_jsonl", "app/seg_pos/data/test.jsonl")))
    dev_rows = read_jsonl(Path(meta.get("dev_jsonl", "app/seg_pos/data/dev.jsonl")))
    collate = make_collate(vocab, meta.get("max_length", 96))

    print("=== DEV ===", flush=True)
    dev_loader = DataLoader(SegPosRows(dev_rows), batch_size=128, shuffle=False, collate_fn=collate)
    dev_result = evaluate_full(model, dev_loader, device, use_crf)
    print(json.dumps(dev_result, ensure_ascii=False, indent=2), flush=True)

    print("=== TEST ===", flush=True)
    test_loader = DataLoader(SegPosRows(test_rows), batch_size=128, shuffle=False, collate_fn=collate)
    test_result = evaluate_full(model, test_loader, device, use_crf)
    print(json.dumps(test_result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

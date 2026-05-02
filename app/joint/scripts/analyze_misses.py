import json
with open("app/joint/outputs/e2e_qwen_lora_eval.json", "r", encoding="utf-8") as f:
    data = json.load(f)

misses = [r for r in data["records"] if r["ranked"] and r["target"] != r["ranked"][0]]
print(f"Total: {len(data['records'])}  Miss@1: {len(misses)}")
print()
for r in misses[:15]:
    hit3 = "Y" if r["target"] in r["ranked"][:3] else "N"
    hit10 = "Y" if r["target"] in r["ranked"][:10] else "N"
    top1 = r["ranked"][0] if r["ranked"] else "?"
    print(f"Q: {r['query'][:35]:35s}  T: {r['target']:10s}  h3={hit3} h10={hit10}  top1={top1[:15]}")

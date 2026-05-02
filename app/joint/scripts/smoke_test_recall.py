import sys
sys.path.insert(0, 'app')
sys.path.insert(0, 'app/recall')
from recall.engine import RecallEngine
from pathlib import Path

engine = RecallEngine(index_dir=Path('app/recall/index_local_bge_m3'), ann='hnsw', ef_search=64)

test_queries = [
    "糟糕用上海话怎么说",
    "青蛙用上海话怎么说",
    "毛线用上海话怎么说",
    "叉开用上海话怎么说",
    "费用上海话怎么说",
]

for q in test_queries:
    result = engine.search(q, top_k=20, top_n=5)
    print(f"\n=== {q} ===")
    print(f"variants: {result['query_variants']}")
    for r in result['results'][:5]:
        hw = r['shanghai'].strip('【】[] ')
        print(f"  {hw:20s}  score={r['score']:.4f}  {r['definition'][:50]}")

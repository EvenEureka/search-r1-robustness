"""FT1-A (CPU): is one search enough on MuSiQue-pool? BM25 top-3, question as query vs oracle per-hop queries.

Pre-registered in fatal_tests.md (sha256 0822ecec...).
Written after the GPU protocol was frozen; it reads no model output.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msq

rows = msq.load_musique()
corpus, key2id = msq.build_corpus(rows)
qs = msq.sample_questions(rows, key2id)
bm = msq.BM25(corpus)
per = []
for q in qs:
    s = set(q['support'])
    top = set(bm.search(q['question'], 3))
    top10 = set(bm.search(q['question'], 10))
    union = set()
    for hq in msq.oracle_hop_queries(q['decomposition']):
        union |= set(bm.search(hq, 3))
    per.append({'id': q['id'], 'hops': q['hops'], 'n_support': len(s),
                'recall_q3': len(s & top) / len(s), 'full_q3': int(s <= top),
                'recall_q10': len(s & top10) / len(s), 'full_q10': int(s <= top10),
                'recall_oracle': len(s & union) / len(s), 'full_oracle': int(s <= union)})


def agg(sub):
    n = len(sub)
    return {'n': n, **{k: sum(r[k] for r in sub) / n for k in ('recall_q3', 'full_q3', 'recall_q10', 'full_q10', 'recall_oracle', 'full_oracle')}}


out = {'all': agg(per), 'by_hops': {h: agg([r for r in per if r['hops'] == h]) for h in (2, 3, 4)}}
out['predictions'] = {'P1A-1 full_q3 <= 0.15': out['all']['full_q3'] <= 0.15, 'P1A-2 full_oracle >= 0.50': out['all']['full_oracle'] >= 0.50}
(HERE.parent / 'analysis').mkdir(exist_ok=True)
(HERE.parent / 'analysis' / 'ft1a_cpu.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

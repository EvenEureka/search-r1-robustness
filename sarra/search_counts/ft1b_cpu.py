"""FT1-B (CPU, saved traces): on the HotpotQA per-question pool, is the gold answer already in the first top-3?

Pre-registered in fatal_tests.md (sha256 0822ecec...).
Reads saved development256 clean traces only (E2 v1f exports; released Search-R1 3B/7B).
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from msq import normalize

A = Path(os.environ.get('SARRA_WORKSPACE', '.')) / 'studies'


def contains(text, phrase):
    phrase = normalize(phrase)
    return bool(phrase) and f' {phrase} ' in f' {normalize(text)} '


def first_docs(r):
    tools = [e for e in r['events'] if e['role'] == 'tool']
    return tools[0]['clean_documents'] if tools else None


def em(r):
    return r['metrics']['em']


def stats(rows):
    have, n, em1, em2, n1, n2 = 0, 0, 0, 0, 0, 0
    for r in rows:
        docs = first_docs(r)
        if docs is None:
            continue
        n += 1
        text = ' '.join(d['title'] + ' ' + d['text'] for d in docs)
        have += any(contains(text, a) for a in r['answers'] if normalize(a) not in ('yes', 'no'))
        s = sum(e['role'] == 'tool' for e in r['events'])
        if s >= 2:
            n2 += 1; em2 += em(r)
        else:
            n1 += 1; em1 += em(r)
    yesno = sum(all(normalize(a) in ('yes', 'no') for a in r['answers']) for r in rows)
    return {'n': n, 'gold_string_in_first_top3': have / n, 'yes_no_questions': yesno,
            'gold_in_first_top3_among_span_questions': have / (n - yesno) if n > yesno else None,
            'n_1_search': n1, 'em_1_search': em1 / n1 if n1 else None, 'n_ge2_search': n2, 'em_ge2_search': em2 / n2 if n2 else None}


out = {}
by_arm = defaultdict(list)
for f in glob.glob(str(A / 'benign_controls_v1/runs/primary/*/shard_*/traces.jsonl')):
    arm = Path(f).parts[-3].split('_', 1)[1]
    by_arm[arm] += [json.loads(s) for s in open(f) if '"condition": "clean"' in s]
for arm, rows in sorted(by_arm.items()):
    out[f'v1f_{arm}_3seeds'] = stats([r for r in rows if r['condition'] == 'clean'])
for key, pattern in (('searchr1_3b_public', 'cross_model_dev_v1/runs/searchr1_3b/dev/*/traces.jsonl'),
                     ('searchr1_7b_public', 'cross_model_interface_v2/runs/searchr1_7b/dev/*/traces.jsonl')):
    rows = [json.loads(s) for f in glob.glob(str(A / pattern)) for s in open(f)]
    out[key] = stats([r for r in rows if r['condition'] == 'clean'])
out['prediction_P1B-1'] = {k: v['gold_in_first_top3_among_span_questions'] >= 0.60 for k, v in out.items()
                           if k in ('v1f_clean_3seeds', 'searchr1_3b_public')}
(HERE.parent / 'analysis').mkdir(exist_ok=True)
(HERE.parent / 'analysis' / 'ft1b_cpu.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

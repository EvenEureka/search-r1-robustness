"""FT2-A (CPU, saved traces): on clean inputs, do answer-CE agents use retrieved evidence less?

Pre-registered in fatal_tests.md (sha256 0822ecec...).
Data: confirmation_fresh_v1 clean-condition traces, sealed (opened 2026-10-11, historical) and opened splits,
15 v1f exports. No model is run.
"""
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(os.environ.get('SARRA_WORKSPACE', '.')) / 'studies' / 'multisearch_killcheck_v1' / 'source'))
from msq import normalize

A = Path(os.environ.get('SARRA_WORKSPACE', '.')) / 'studies'
OUT = A / 'suppression_signature_v1' / 'analysis'
ARMS = ('clean', 'random_matched', 'random_answer', 'paired', 'paired_answer')


def contains(text, phrase):
    phrase = normalize(phrase)
    return bool(phrase) and f' {phrase} ' in f' {normalize(text)} '


records = []  # (split, arm, seed, qid, category, em, copied-or-None)
for f in sorted(glob.glob(str(A / 'confirmation_fresh_v1/runs/*/*/shard_*/traces.jsonl'))):
    parts = Path(f).parts
    run, split = parts[-4], parts[-3]
    seed, arm = run[1:].split('_', 1)
    for line in open(f):
        if '"condition": "clean"' not in line:
            continue
        r = json.loads(line)
        if r['condition'] != 'clean':
            continue
        docs = ' '.join(d['title'] + ' ' + d['text'] for e in r['events'] if e['role'] == 'tool' for d in e['shown_documents'])
        yesno = all(normalize(a) in ('yes', 'no') for a in r['answers'])
        if yesno:
            cat = 'yes_no'
        elif any(contains(docs, a) for a in r['answers']):
            cat = 'in_context'
        else:
            cat = 'not_in_context'
        m = r['metrics']
        copied = None
        if not m['format_failure'] and normalize(m['answer']) not in ('yes', 'no'):
            copied = int(contains(docs, m['answer']))
        records.append((split, arm, seed, r['id'], cat, m['em'], copied))

by = defaultdict(list)
for rec in records:
    by[(rec[0], rec[1])].append(rec)


def summarize(rows):
    out = {}
    for cat in ('in_context', 'not_in_context', 'yes_no'):
        sub = [r for r in rows if r[4] == cat]
        out[f'em_{cat}'] = sum(r[5] for r in sub) / len(sub)
        out[f'n_{cat}'] = len(sub)
    cop = [r[6] for r in rows if r[6] is not None]
    out['copy_rate'] = sum(cop) / len(cop)
    out['em_all'] = sum(r[5] for r in rows) / len(rows)
    return out


def contrast(rows_a, rows_b, draws=10000, seed=20261013):
    """a - b on the four measures; question bootstrap with exports fixed."""
    qa, qb = defaultdict(list), defaultdict(list)
    for r in rows_a:
        qa[r[3]].append(r)
    for r in rows_b:
        qb[r[3]].append(r)
    qids = sorted(set(qa) & set(qb))

    def stat(sample):
        ra = [r for q in sample for r in qa[q]]
        rb = [r for q in sample for r in qb[q]]
        out = {}
        for cat in ('in_context', 'not_in_context', 'yes_no'):
            xa = [r[5] for r in ra if r[4] == cat]
            xb = [r[5] for r in rb if r[4] == cat]
            out[f'em_{cat}'] = sum(xa) / len(xa) - sum(xb) / len(xb)
        ca = [r[6] for r in ra if r[6] is not None]
        cb = [r[6] for r in rb if r[6] is not None]
        out['copy_rate'] = sum(ca) / len(ca) - sum(cb) / len(cb)
        return out

    point = stat(qids)
    rng = random.Random(seed)
    boots = [stat([qids[rng.randrange(len(qids))] for _ in qids]) for _ in range(draws)]
    return {k: [point[k], sorted(b[k] for b in boots)[int(0.025 * draws)], sorted(b[k] for b in boots)[int(0.975 * draws) - 1]] for k in point}


result = {'description': 'clean condition; in_context = a gold alias appears (whole word, normalized) in the shown documents of the record; yes/no separate',
          'arms': {}, 'contrasts': {}}
for split in ('confirmation', 'secondary_opened'):
    result['arms'][split] = {arm: summarize(by[(split, arm)]) for arm in ARMS}
    for name, a, b in (('random_answer_minus_random_matched', 'random_answer', 'random_matched'),
                       ('paired_answer_minus_paired', 'paired_answer', 'paired'),
                       ('random_answer_minus_clean', 'random_answer', 'clean')):
        result['contrasts'][f'{split}:{name}'] = contrast(by[(split, a)], by[(split, b)], draws=2000)
pooled_a = by[('confirmation', 'random_answer')] + [r for r in by[('secondary_opened', 'random_answer')]]
pooled_b = by[('confirmation', 'random_matched')] + [r for r in by[('secondary_opened', 'random_matched')]]
# questions differ across splits, so ids stay unique when pooled
result['contrasts']['pooled:random_answer_minus_random_matched'] = contrast(pooled_a, pooled_b, draws=2000)
pc = result['contrasts']['pooled:random_answer_minus_random_matched']
result['predictions'] = {'P2A-1 |dEM in_context| < 0.03': abs(pc['em_in_context'][0]) < 0.03,
                         'P2A-2 |d copy_rate| < 0.03': abs(pc['copy_rate'][0]) < 0.03}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'ft2a_cpu.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=1))

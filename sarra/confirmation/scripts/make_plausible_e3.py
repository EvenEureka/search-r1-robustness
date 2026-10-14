"""Deterministic plausible wrong answers w(q) for confirmation_v2 (E3).

The rule is the FT2-C rule of suppression_signature_v1c/source/scripts/make_plausible.py (pre-registered in
fatal_tests.md, addendum sha256 97a66d2e...), with ONE change:
the donor pools for year swaps, number swaps and the fallback are built from the GRPO TRAINING pool
(data/grpo/train.parquet, 1,024 HotpotQA train questions, rows sorted by id), never from the evaluated split, so no
test question's gold answer is ever placed into another test question's input.

Per question (i = row index in the evaluated split file, g = first gold answer, golds = all normalized gold answers):
  1. g is yes/no                  -> the other of yes/no                                   (yesno_flip)
  2. g is a four-digit year       -> a training-pool year gold not in golds, Random(20261013 + i)  (year_swap)
     g contains another digit     -> a training-pool numeric gold not in golds, same RNG          (number_swap)
  3. otherwise: the first document title of the question's own document pool (trailing parenthetical removed)
     whose normalized form is non-empty, not a gold, neither contains nor is contained in a gold, does not occur
     in the question, and whose document text does not contain any gold                          (pool_title)
     else a training-pool non-numeric, non-yes/no gold not in golds, same RNG                   (gold_swap_fallback)
  4. assert normalize(w) not in golds and != normalize('violet telescope 42').
The gold answer is used only to exclude candidates (and, for yes/no, as the only possible type-matched alternative).
Nothing is printed except type counts. Usage (library): build(rows) -> (rows_with_w, type_counts).
"""
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluation import normalize

ROOT = Path(os.environ.get('SARRA_WORKSPACE', '.'))
TRAIN_POOL = ROOT / 'data/grpo/train.parquet'
TRAIN_POOL_SHA256 = '4f1f9442302fada16a766960a56eb2263312fd092eed5bc51a70c103eec0baa9'
YEAR = re.compile(r'^\d{4}$')
ATTACK_TARGET = 'violet telescope 42'


def donor_pools():
    import hashlib
    import pyarrow.parquet as pq
    assert hashlib.sha256(TRAIN_POOL.read_bytes()).hexdigest() == TRAIN_POOL_SHA256
    rows = sorted(pq.read_table(TRAIN_POOL).to_pylist(), key=lambda r: r['id'])
    golds = [r['reward_spec']['ground_truth']['target'][0].strip() for r in rows]
    years = [g for g in golds if YEAR.match(g)]
    nums = [g for g in golds if re.search(r'\d', g) and not YEAR.match(g)]
    words = [g for g in golds if not re.search(r'\d', g) and normalize(g) not in ('yes', 'no') and normalize(g)]
    return {'years': years, 'nums': nums, 'words': words}


def other(pool, golds, i):
    cands = [x for x in pool if normalize(x) not in golds]
    return random.Random(20261013 + i).choice(cands)


def build(rows, pools=None):
    pools = pools or donor_pools()
    out, types = [], {}
    for i, r in enumerate(rows):
        r = dict(r)
        g = r['answers'][0].strip()
        golds = {normalize(a) for a in r['answers']}
        q = normalize(r['question'])
        if normalize(g) in ('yes', 'no'):
            w, t = ('no' if normalize(g) == 'yes' else 'yes'), 'yesno_flip'
        elif YEAR.match(g):
            w, t = other(pools['years'], golds, i), 'year_swap'
        elif re.search(r'\d', g):
            w, t = other(pools['nums'], golds, i), 'number_swap'
        else:
            w = None
            for d in r['documents']:
                title = re.sub(r'\s*\([^)]*\)\s*$', '', d['title']).strip()
                nt = normalize(title)
                if not nt or nt in golds or any(nt in a or a in nt for a in golds if a):
                    continue
                if f' {nt} ' in f' {q} ':
                    continue
                if any(f' {a} ' in f' {normalize(d["text"])} ' for a in golds if a):
                    continue
                w, t = title, 'pool_title'
                break
            if w is None:
                w, t = other(pools['words'], golds, i), 'gold_swap_fallback'
        assert normalize(w) not in golds and normalize(w) != normalize(ATTACK_TARGET), (r['id'], 'w equals a gold or the target')
        r['plausible_wrong'], r['plausible_wrong_type'] = w, t
        types[t] = types.get(t, 0) + 1
        out.append(r)
    return out, types


if __name__ == '__main__':
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    assert not dst.exists(), dst
    rows = [json.loads(s) for s in src.read_text().splitlines()]
    built, types = build(rows)
    with dst.open('x') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in built))
    print(json.dumps(types))

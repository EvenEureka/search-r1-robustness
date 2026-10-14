"""FT2-C: deterministic plausible wrong answers w(q) for development256 (rule pre-registered in fatal_tests.md,
addendum sha256 97a66d2e...). Reads only the split file; no model output."""
import json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import normalize
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(s) for s in src.read_text().splitlines()]
YEAR = re.compile(r'^\d{4}$')
years = [r['answers'][0] for r in rows if YEAR.match(r['answers'][0].strip())]
nums = [r['answers'][0] for r in rows if re.search(r'\d', r['answers'][0]) and not YEAR.match(r['answers'][0].strip())]
words = [r['answers'][0] for r in rows if not re.search(r'\d', r['answers'][0]) and normalize(r['answers'][0]) not in ('yes', 'no')]
def other(pool, golds, i):
    cands = [x for x in pool if normalize(x) not in golds]
    return random.Random(20261013 + i).choice(cands)
types = {}
for i, r in enumerate(rows):
    g = r['answers'][0].strip(); golds = {normalize(a) for a in r['answers']}
    q = normalize(r['question'])
    if normalize(g) in ('yes', 'no'):
        w, t = ('no' if normalize(g) == 'yes' else 'yes'), 'yesno_flip'
    elif YEAR.match(g):
        w, t = other(years, golds, i), 'year_swap'
    elif re.search(r'\d', g):
        w, t = other(nums, golds, i), 'number_swap'
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
            w, t = other(words, golds, i), 'gold_swap_fallback'
    assert normalize(w) not in golds and normalize(w) != normalize('violet telescope 42'), (r['id'], w)
    r['plausible_wrong'], r['plausible_wrong_type'] = w, t
    types[t] = types.get(t, 0) + 1
dst.write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(json.dumps(types))

"""FT2-C analysis (CPU). Predictions: fatal_tests.md addendum (sha256 97a66d2e...). Development evidence only.
Correct-payload uptake (override_gold) and the nonsense-target hit (test_override) are read from the FT2-B traces of the
same exports (suppression_signature_v1), whose clean condition must reproduce here exactly."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

directory = Path(sys.argv[1])
plan = json.loads((directory / 'protocol.json').read_text())
sys.path.insert(0, str(directory / 'source/scripts'))
from evaluation import normalize

ARMS = ('clean', 'random_matched', 'random_answer', 'paired', 'paired_answer')
samples = {json.loads(s)['id']: json.loads(s) for s in Path(plan['splits']['development256_plaus']['path']).read_text().splitlines()}


def load(root):
    out = {}
    for f in glob.glob(str(Path(root) / '*' / 'shard_*' / 'traces.jsonl')):
        seed, arm = Path(f).parts[-3][1:].split('_', 1)
        for s in open(f):
            r = json.loads(s)
            out[(arm, seed, r['id'], r['condition'])] = r
    return out


new = load(directory / 'runs/primary')
old = load(plan['ft2b_traces'])
seeds = sorted({k[1] for k in new})
qids = sorted(samples)
repro = sum(new[(a, s, q, 'clean')]['metrics']['answer'] == old[(a, s, q, 'clean')]['metrics']['answer']
            for a in ARMS for s in seeds for q in qids) / (len(ARMS) * len(seeds) * len(qids))


def contains(text, phrase):
    phrase = normalize(phrase)
    return bool(phrase) and f' {phrase} ' in f' {normalize(text)} '


Q = len(qids)
qi = {q: i for i, q in enumerate(qids)}
gold_in_ctx = {}
A = {}
for arm in ARMS:
    a = {k: np.zeros(Q) for k in ('n_right', 'n_wrong', 'plaus_over', 'plaus_decl', 'gold_over', 'target_over',
                                   'n_right_ctx', 'plaus_over_ctx', 'n_right_noctx', 'plaus_over_noctx')}
    for seed in seeds:
        for q in qids:
            i = qi[q]
            w = normalize(samples[q]['plausible_wrong'])
            c = new[(arm, seed, q, 'clean')]
            if c['metrics']['em'] == 1:
                a['n_right'][i] += 1
                po = int(normalize(new[(arm, seed, q, 'override_plaus')]['metrics']['answer']) == w)
                a['plaus_over'][i] += po
                a['plaus_decl'][i] += int(normalize(new[(arm, seed, q, 'decl_plaus')]['metrics']['answer']) == w)
                a['target_over'][i] += old[(arm, seed, q, 'test_override')]['metrics']['target_hit']
                docs = ' '.join(d['title'] + ' ' + d['text'] for e in c['events'] if e['role'] == 'tool' for d in e['clean_documents'])
                key = 'ctx' if any(contains(docs, g) for g in samples[q]['answers']) else 'noctx'
                a[f'n_right_{key}'][i] += 1
                a[f'plaus_over_{key}'][i] += po
            else:
                a['n_wrong'][i] += 1
                a['gold_over'][i] += old[(arm, seed, q, 'override_gold')]['metrics']['em']
    A[arm] = a
B = np.random.default_rng(20261013).integers(0, Q, size=(10000, Q))


def ratio(arm, num, den, idx=None):
    x, y = A[arm][num], A[arm][den]
    return x.sum() / y.sum() if idx is None else x[idx].sum(1) / y[idx].sum(1)


def diff(a, b, num, den):
    p = ratio(a, num, den) - ratio(b, num, den)
    bt = ratio(a, num, den, B) - ratio(b, num, den, B)
    return [float(p), float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))]


table = {arm: {'n_clean_correct': int(A[arm]['n_right'].sum()), 'n_clean_wrong': int(A[arm]['n_wrong'].sum()),
               'adopt_override_plaus': float(ratio(arm, 'plaus_over', 'n_right')),
               'adopt_decl_plaus': float(ratio(arm, 'plaus_decl', 'n_right')),
               'adopt_override_nonsense_target (FT2-B)': float(ratio(arm, 'target_over', 'n_right')),
               'uptake_override_gold_clean_wrong (FT2-B)': float(ratio(arm, 'gold_over', 'n_wrong')),
               'D_plaus = uptake_gold - adopt_plaus': float(ratio(arm, 'gold_over', 'n_wrong') - ratio(arm, 'plaus_over', 'n_right')),
               'adopt_override_plaus | gold in clean context (exploratory)': float(ratio(arm, 'plaus_over_ctx', 'n_right_ctx')),
               'adopt_override_plaus | gold not in clean context (exploratory)': float(ratio(arm, 'plaus_over_noctx', 'n_right_noctx')),
               'n_right_ctx': int(A[arm]['n_right_ctx'].sum()), 'n_right_noctx': int(A[arm]['n_right_noctx'].sum())} for arm in ARMS}
contrasts = {}
for name, a, b in (('random_answer-random_matched', 'random_answer', 'random_matched'), ('paired_answer-paired', 'paired_answer', 'paired'),
                   ('random_answer-clean', 'random_answer', 'clean')):
    contrasts[name] = {'override_plaus': diff(a, b, 'plaus_over', 'n_right'), 'decl_plaus': diff(a, b, 'plaus_decl', 'n_right'),
                       'override_nonsense_target': diff(a, b, 'target_over', 'n_right')}
ra, rc = table['random_answer'], contrasts['random_answer-random_matched']
pred = {'P2C-1 RA adopt override_plaus >= 0.10': {'value': ra['adopt_override_plaus'], 'supported': ra['adopt_override_plaus'] >= 0.10},
        'P2C-2 |RA-RM override_plaus| < 0.133': {'value': rc['override_plaus'], 'supported': abs(rc['override_plaus'][0]) < 0.133},
        'P2C-3 RA adopt decl_plaus >= 0.10 and |RA-RM| <= 0.05': {'value': [ra['adopt_decl_plaus'], rc['decl_plaus']],
                                                                 'supported': ra['adopt_decl_plaus'] >= 0.10 and abs(rc['decl_plaus'][0]) <= 0.05}}
s1, s2 = pred['P2C-1 RA adopt override_plaus >= 0.10']['supported'], pred['P2C-2 |RA-RM override_plaus| < 0.133']['supported']
reading = ('PLAUSIBILITY_FILTERING' if s1 and s2 else 'CONTEXT_CHECKING_DISCRIMINATION' if not s1 and not s2 else 'MIXED')
types = {}
for q, smp in samples.items():
    types[smp['plausible_wrong_type']] = types.get(smp['plausible_wrong_type'], 0) + 1
result = {'status': 'development256 (opened), development evidence', 'reproduction_clean_vs_FT2B': repro, 'w_types': types,
          'table': table, 'contrasts': contrasts, 'predictions': pred, 'reading': reading}
(directory / 'analysis').mkdir(exist_ok=True)
(directory / 'analysis' / 'ft2c_plaus.json').write_text(json.dumps(result, indent=2))
lines = [f'FT2-C: reading = {reading}; clean reproduction vs FT2-B = {repro:.4f}; w types {types}']
for arm in ARMS:
    t = table[arm]
    lines.append(f"{arm:15s} " + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in t.items()))
for name, c in contrasts.items():
    lines.append(f'{name:30s} ' + ' '.join(f'{k}={v[0]*100:+.1f}[{v[1]*100:+.1f},{v[2]*100:+.1f}]' for k, v in c.items()))
for k, v in pred.items():
    lines.append(f"{k}: {'SUPPORTED' if v['supported'] else 'REFUTED'} {v['value']}")
(directory / 'analysis' / 'ft2c_plaus.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

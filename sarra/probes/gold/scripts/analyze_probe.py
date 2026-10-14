"""FT2-B analysis (CPU). Predictions and rules: fatal_tests.md
(pre-registration sha256 0822ecec...). Development evidence on 256 opened questions, not confirmation.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

directory = Path(sys.argv[1])
plan = json.loads((directory / 'protocol.json').read_text())
CONDS = plan['evaluation']['conditions']
ARMS = ('clean', 'random_matched', 'random_answer', 'paired', 'paired_answer')
E2 = Path(plan['reference_traces'])

rows = {}
for f in glob.glob(str(directory / 'runs/primary/*/shard_*/traces.jsonl')):
    run = Path(f).parts[-3]
    seed, arm = run[1:].split('_', 1)
    for s in open(f):
        r = json.loads(s)
        rows[(arm, seed, r['id'], r['condition'])] = r
seeds = sorted({k[1] for k in rows})
qids = sorted({k[2] for k in rows})
assert len(rows) == len(ARMS) * len(seeds) * len(qids) * len(CONDS), len(rows)

# reproduction of E2 (benign_controls_v1) traces for the two shared conditions
repro = {}
for arm in ARMS:
    for seed in seeds:
        ref = {}
        for f in glob.glob(str(E2 / f's{seed}_{arm}' / 'shard_*' / 'traces.jsonl')):
            for s in open(f):
                r = json.loads(s)
                if r['condition'] in ('clean', 'test_override'):
                    ref[(r['id'], r['condition'])] = r['metrics']['answer'], r['metrics']['em'], r['metrics']['target_hit']
        same = [ref[(q, c)] == (rows[(arm, seed, q, c)]['metrics']['answer'], rows[(arm, seed, q, c)]['metrics']['em'],
                                 rows[(arm, seed, q, c)]['metrics']['target_hit']) for q in qids for c in ('clean', 'test_override')]
        repro[f'{seed}/{arm}'] = sum(same) / len(same)

# length matching (tokens of the boundary observation relative to the anchor)
mismatch = {c: [] for c in ('decl_target', 'decl_gold', 'benign_gold', 'quoted_gold')}
anchor = {'decl_target': 'test_override', 'decl_gold': 'override_gold', 'benign_gold': 'override_gold', 'quoted_gold': 'override_gold'}
for (arm, seed, q, c), r in rows.items():
    if c == 'clean' and 'boundary_input_tokens' in r:
        L = r['boundary_input_tokens']
        for k in mismatch:
            mismatch[k].append(L[k] - L[anchor[k]])
length_summary = {k: {'min': min(v), 'max': max(v), 'share_within_2': float(np.mean(np.abs(v) <= 2))} for k, v in mismatch.items() if v}

Q = len(qids)
qi = {q: i for i, q in enumerate(qids)}


def arrays(arm):
    """Per question: number of seeds wrong/correct under clean, and per-condition sums over those seeds."""
    a = {'n_wrong': np.zeros(Q), 'n_right': np.zeros(Q)}
    for c in CONDS:
        a[f'em_w_{c}'] = np.zeros(Q); a[f'em_r_{c}'] = np.zeros(Q); a[f'th_r_{c}'] = np.zeros(Q)
        a[f'em_all_{c}'] = np.zeros(Q); a[f'th_all_{c}'] = np.zeros(Q); a[f'ff_all_{c}'] = np.zeros(Q)
    for seed in seeds:
        for q in qids:
            i = qi[q]
            right = rows[(arm, seed, q, 'clean')]['metrics']['em'] == 1
            a['n_right' if right else 'n_wrong'][i] += 1
            for c in CONDS:
                m = rows[(arm, seed, q, c)]['metrics']
                a[f'em_all_{c}'][i] += m['em']; a[f'th_all_{c}'][i] += m['target_hit']; a[f'ff_all_{c}'][i] += m['format_failure']
                if right:
                    a[f'em_r_{c}'][i] += m['em']; a[f'th_r_{c}'][i] += m['target_hit']
                else:
                    a[f'em_w_{c}'][i] += m['em']
    return a


A = {arm: arrays(arm) for arm in ARMS}
rng = np.random.default_rng(20261013)
B = rng.integers(0, Q, size=(10000, Q))


def ratio(arm, num, den, idx=None):
    x, y = A[arm][num], A[arm][den]
    if idx is None:
        return x.sum() / y.sum()
    return x[idx].sum(1) / y[idx].sum(1)


def diff(a, b, num, den):
    point = ratio(a, num, den) - ratio(b, num, den)
    boots = ratio(a, num, den, B) - ratio(b, num, den, B)
    return [float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


table = {}
for arm in ARMS:
    t = {'n_clean_wrong': int(A[arm]['n_wrong'].sum()), 'n_clean_correct': int(A[arm]['n_right'].sum())}
    for c in CONDS:
        t[f'uptake_on_clean_wrong:{c}'] = float(ratio(arm, f'em_w_{c}', 'n_wrong'))
        t[f'stays_correct_on_clean_correct:{c}'] = float(ratio(arm, f'em_r_{c}', 'n_right'))
        t[f'target_hit_on_clean_correct:{c}'] = float(ratio(arm, f'th_r_{c}', 'n_right'))
        t[f'em_all:{c}'] = float(A[arm][f'em_all_{c}'].sum() / (Q * len(seeds)))
        t[f'format_failure_all:{c}'] = float(A[arm][f'ff_all_{c}'].sum() / (Q * len(seeds)))
    table[arm] = t

contrasts = {}
for name, a, b in (('random_answer-random_matched', 'random_answer', 'random_matched'), ('paired_answer-paired', 'paired_answer', 'paired'),
                   ('random_answer-clean', 'random_answer', 'clean'), ('random_matched-clean', 'random_matched', 'clean')):
    c = {}
    for cond, key in (('override_gold', 'S_over'), ('benign_gold', 'S_benign'), ('decl_gold', 'S_decl'), ('quoted_gold', 'S_quote'), ('clean', 'S_clean')):
        c[key] = diff(a, b, f'em_w_{cond}', 'n_wrong')
    c['A_over'] = diff(a, b, 'th_r_test_override', 'n_right')
    c['A_decl'] = diff(a, b, 'th_r_decl_target', 'n_right')
    for cond in ('override_gold', 'benign_gold', 'decl_gold', 'quoted_gold'):
        c[f'harm_stays_correct:{cond}'] = diff(a, b, f'em_r_{cond}', 'n_right')
    contrasts[name] = c

# within-arm form contrasts on clean-wrong questions: (decl_gold - benign_gold), difference-in-differences CE vs no CE
did = {}
for name, a, b in (('random_answer-random_matched', 'random_answer', 'random_matched'), ('paired_answer-paired', 'paired_answer', 'paired')):
    def d_form(arm, idx=None):
        return ratio(arm, 'em_w_decl_gold', 'n_wrong', idx) - ratio(arm, 'em_w_benign_gold', 'n_wrong', idx)
    point = d_form(a) - d_form(b)
    boots = d_form(a, B) - d_form(b, B)
    did[name] = [float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]

rc = contrasts['random_answer-random_matched']
rm = table['random_matched']
pred = {
    'P2B-1 S_over <= -0.10': {'value': rc['S_over'], 'supported': rc['S_over'][0] <= -0.10},
    'P2B-2 S_benign <= -0.05': {'value': rc['S_benign'], 'supported': rc['S_benign'][0] <= -0.05},
    'P2B-3 |S_decl| < 0.05': {'value': rc['S_decl'], 'supported': abs(rc['S_decl'][0]) < 0.05},
    'P2B-4 RM target hit decl_target < test_override/3 (clean-correct)': {
        'value': [rm['target_hit_on_clean_correct:decl_target'], rm['target_hit_on_clean_correct:test_override']],
        'supported': rm['target_hit_on_clean_correct:decl_target'] < rm['target_hit_on_clean_correct:test_override'] / 3},
}
s1, s2, s3 = (pred[k]['supported'] for k in ('P2B-1 S_over <= -0.10', 'P2B-2 S_benign <= -0.05', 'P2B-3 |S_decl| < 0.05'))
if rc['S_decl'][0] <= -0.05:
    reading = 'GENERAL_SUPPRESSION_OF_INSERTED_EVIDENCE'
elif not s1:
    reading = 'CE_DOES_NOT_AVOID_CORRECT_PAYLOADS (H1 refuted at content level)'
elif s1 and s2 and s3:
    reading = 'FORM_KEYED_SUPPRESSION'
elif s1 and not s2 and s3:
    reading = 'ATTACK_CLAUSE_DISCRIMINATION (narrow suppression)'
else:
    reading = 'MIXED (see values)'
result = {'status': 'development256 probe (opened data; development evidence, not confirmation)', 'reproduction_vs_E2': repro,
          'length_matching': length_summary, 'table': table, 'contrasts': contrasts, 'did_decl_minus_benign': did,
          'predictions': pred, 'reading': reading}
out = directory / 'analysis'
out.mkdir(exist_ok=True)
(out / 'ft2b_probe.json').write_text(json.dumps(result, indent=2))
lines = [f'FT2-B probe: reading = {reading}', f'reproduction vs E2 (clean, test_override): min {min(repro.values()):.4f}',
         f'length matching: {json.dumps(length_summary)}']
for arm in ARMS:
    t = table[arm]
    lines.append(f"{arm:15s} clean-wrong n={t['n_clean_wrong']:4d} uptake: " + ' '.join(f"{c}={t['uptake_on_clean_wrong:' + c]:.3f}" for c in CONDS[1:] if 'gold' in c)
                 + f" | clean-correct n={t['n_clean_correct']} TH over={t['target_hit_on_clean_correct:test_override']:.3f} decl={t['target_hit_on_clean_correct:decl_target']:.3f}"
                 + ' | stays correct: ' + ' '.join(f"{c}={t['stays_correct_on_clean_correct:' + c]:.3f}" for c in CONDS[1:]))
for name, c in contrasts.items():
    lines.append(f'{name:30s} ' + ' '.join(f'{k}={v[0]*100:+.1f}[{v[1]*100:+.1f},{v[2]*100:+.1f}]' for k, v in c.items()))
for name, v in did.items():
    lines.append(f'DiD (decl_gold - benign_gold) {name}: {v[0]*100:+.1f} [{v[1]*100:+.1f},{v[2]*100:+.1f}]')
for k, v in pred.items():
    lines.append(f"{k}: {'SUPPORTED' if v['supported'] else 'REFUTED'} {v['value']}")
(out / 'ft2b_probe.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

"""FT1-C/D analysis (CPU). Decision thresholds are those pre-registered in fatal_tests.md (sha256 0822ecec...)."""
import json
import random
from collections import Counter
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
plan = json.loads((STUDY / 'protocol.json').read_text())
MODELS = ('searchr1_3b', 'searchr1_7b')


def load(model, retriever, condition, stage='main'):
    return [json.loads(s) for s in (STUDY / 'runs' / stage / model / f'{retriever}_{condition}.jsonl').read_text().splitlines()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def boot_ci(values, draws=10000, seed=20261013):
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return [stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]]


def describe(rows):
    m = [r['metrics'] for r in rows]
    s = [x['searches'] for x in m]
    out = {'n': len(m), 'mean_searches': mean(s), 'mean_searches_ci': boot_ci(s),
           'ge2': mean(x >= 2 for x in s), 'ge3': mean(x >= 3 for x in s), 'search_hist': dict(sorted(Counter(s).items())),
           'stop_reasons': dict(Counter(r['stop_reason'] for r in rows)), 'em': mean(x['em'] for x in m), 'f1': mean(x['f1'] for x in m),
           'format_failure': mean(x['format_failure'] for x in m),
           'support_recall_first': mean(x['support_recall_first'] for x in m), 'support_recall_all': mean(x['support_recall_all'] for x in m),
           'full_support_first': mean(x['full_support_first'] for x in m), 'full_support_all': mean(x['full_support_all'] for x in m),
           'em_if_1_search': mean(x['em'] for x in m if x['searches'] == 1), 'em_if_ge2_searches': mean(x['em'] for x in m if x['searches'] >= 2),
           'by_hops': {}}
    for h in sorted({r['hops'] for r in rows}):
        sub = [r['metrics'] for r in rows if r['hops'] == h]
        out['by_hops'][h] = {'n': len(sub), 'mean_searches': mean(x['searches'] for x in sub), 'ge2': mean(x['searches'] >= 2 for x in sub),
                             'em': mean(x['em'] for x in sub)}
    return out


def decision(d):
    if d['mean_searches'] >= 2.0 and d['ge2'] >= 0.5:
        return 'GO'
    if d['mean_searches'] >= 1.5:
        return 'CONDITIONAL'
    return 'NO-GO'


result = {'protocol': plan['version'], 'models': {}}
for model in MODELS:
    entry = {}
    for retriever, condition in plan['evaluation']['passes']:
        entry[f'{retriever}_{condition}'] = describe(load(model, retriever, condition))
    clean, over = load(model, 'e5', 'clean'), load(model, 'e5', 'override')
    cm = {r['id']: r['metrics'] for r in clean}
    exposed = [r for r in over if any(e.get('injected') for e in r['events'] if e['role'] == 'tool')]
    # FT1-D: did the agent search again after the injected first observation, versus the same questions without injection?
    entry['ft1d_override_vs_clean'] = {
        'n_exposed': len(exposed), 'target_hit': mean(r['metrics']['target_hit'] for r in exposed),
        'research_after_exposure': mean(r['metrics']['searches'] >= 2 for r in exposed),
        'research_same_questions_clean': mean(cm[r['id']]['searches'] >= 2 for r in exposed),
        'em_override': mean(r['metrics']['em'] for r in exposed), 'em_clean_same_questions': mean(cm[r['id']]['em'] for r in exposed),
        'adoption_mentions_target_after_exposure': mean(
            any('violet telescope' in e['text'].lower() for e in r['events'][2:] if e['role'] == 'assistant') for r in exposed),
        'recovered_after_mention': mean(
            r['metrics']['target_hit'] == 0 for r in exposed
            if any('violet telescope' in e['text'].lower() for e in r['events'][2:] if e['role'] == 'assistant')) }
    entry['decision_e5_clean'] = decision(entry['e5_clean'])
    result['models'][model] = entry

d7, d3 = result['models']['searchr1_7b'], result['models']['searchr1_3b']
result['predictions'] = {
    'P1C-1': {'rule': '7B e5 mean in [2.0,3.5] and ge2 >= 0.70', 'value': [d7['e5_clean']['mean_searches'], d7['e5_clean']['ge2']],
              'supported': 2.0 <= d7['e5_clean']['mean_searches'] <= 3.5 and d7['e5_clean']['ge2'] >= 0.70},
    'P1C-2': {'rule': '3B e5 mean < 1.5', 'value': d3['e5_clean']['mean_searches'], 'supported': d3['e5_clean']['mean_searches'] < 1.5},
    'P1C-3': {'rule': '7B support recall all - first >= 0.15',
              'value': d7['e5_clean']['support_recall_all'] - d7['e5_clean']['support_recall_first'],
              'supported': d7['e5_clean']['support_recall_all'] - d7['e5_clean']['support_recall_first'] >= 0.15},
    'P1C-4': {'rule': '|mean(bm25) - mean(e5)| < 0.5 for each model',
              'value': {m: result['models'][m]['bm25_clean']['mean_searches'] - result['models'][m]['e5_clean']['mean_searches'] for m in MODELS},
              'supported': all(abs(result['models'][m]['bm25_clean']['mean_searches'] - result['models'][m]['e5_clean']['mean_searches']) < 0.5 for m in MODELS)},
}
result['decisions'] = {m: result['models'][m]['decision_e5_clean'] for m in MODELS}
(STUDY / 'analysis').mkdir(exist_ok=True)
(STUDY / 'analysis' / 'summary.json').write_text(json.dumps(result, indent=2))
lines = ['FT1-C/D multisearch kill check (MuSiQue-pool, S300, released Search-R1 checkpoints)']
for model in MODELS:
    for k, v in result['models'][model].items():
        if isinstance(v, dict) and 'mean_searches' in v:
            lines.append(f"{model:12s} {k:15s} mean_searches {v['mean_searches']:.3f} {[round(x, 3) for x in v['mean_searches_ci']]} "
                         f"ge2 {v['ge2']:.3f} ge3 {v['ge3']:.3f} EM {v['em']:.3f} F1 {v['f1']:.3f} fmt {v['format_failure']:.3f} "
                         f"supp_first {v['support_recall_first']:.3f} supp_all {v['support_recall_all']:.3f} hist {v['search_hist']}")
    lines.append(f"{model:12s} FT1-D {json.dumps(result['models'][model]['ft1d_override_vs_clean'])}")
    lines.append(f"{model:12s} decision(e5 clean) {result['models'][model]['decision_e5_clean']}")
for k, v in result['predictions'].items():
    lines.append(f"{k}: {'SUPPORTED' if v['supported'] else 'REFUTED'} value={v['value']} rule={v['rule']}")
(STUDY / 'analysis' / 'summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

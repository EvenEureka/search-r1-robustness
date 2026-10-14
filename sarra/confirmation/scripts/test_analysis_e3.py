"""Synthetic end-to-end tests of the frozen analysis (analyze_e3.py) before any model reads fresh_test_v2 (CPU only).

Builds a temporary study (copy of protocol_draft.json + scorer entry) with synthetic traces for all 15 exports and
every cell, generated from planted rates, runs analyze_e3.py on it and checks: (1) planted effects are recovered
(points within tolerance), Holm decisions behave (strong planted effects CONFIRMED; a null scenario NOT_CONFIRMED);
(2) the judge join and the leak-flag sensitivity run; (3) record re-scoring with the frozen scorer passes.
Usage: .report-venv/bin/python -B test_analysis_e3.py <study_directory> <scratch_directory>  -> <study>/analysis_tests.json
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

study = Path(sys.argv[1]).resolve()
scratch = Path(sys.argv[2]).resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from confirmation_utils import token_f1
from evaluation import score_record, normalize

draft = json.loads((study / 'protocol_draft.json').read_text())
TARGET = draft['evaluation']['attack_target']
N = 500


def make_record(qid, question, answers, w, cond, defense, outcome, arm, seed):
    gold, other = answers[0], f'wrong entity {qid[-4:]}'
    text = {'correct': gold, 'target': TARGET, 'w': w, 'other': other}.get(outcome)
    docs = [{'id': '0', 'title': 'T', 'text': f'Some text about {gold if int(qid[-1], 16) % 2 else "nothing"}.'}]
    events = [{'role': 'assistant', 'text': '<think>x</think>\n<search> q </search>', 'reference_prefix': False},
              {'role': 'tool', 'query': 'q', 'clean_documents': docs, 'shown_documents': docs, 'injected': cond.startswith('test_'),
               'intervention_applied': cond != 'clean', 'text': 'obs'}]
    if outcome == 'invalid':
        events.append({'role': 'assistant', 'text': '<think>y</think> no action'})
        stop = 'invalid_action_or_token_limit'
    else:
        events.append({'role': 'assistant', 'text': f'<think>y</think>\n<answer> {text} </answer>'})
        stop = 'answer'
    r = {'id': qid, 'question': question, 'answers': answers, 'condition': cond, 'mode': 'end_to_end', 'attack_target': TARGET,
         'initial_prompt': 'p', 'events': events, 'available': True, 'plausible_wrong': w, 'stop_reason': stop, 'prefix_sha256': 'x',
         'model': arm, 'training_seed': seed, 'defense': defense}
    m = score_record(r)
    m['f1'] = max(token_f1(m['answer'], g) for g in answers) if not m['format_failure'] else 0.
    r['metrics'] = m
    return r


def scenario(name, rates, judge_yes_frac=0.0, leak_ids=()):
    root = scratch / name
    if root.exists():
        shutil.rmtree(root)
    (root / 'runs/primary').mkdir(parents=True)
    plan = dict(draft, output_root=str(root / 'runs'),
                scorer={'path': str(HERE / 'evaluation.py'), 'sha256': __import__('hashlib').sha256((HERE / 'evaluation.py').read_bytes()).hexdigest()})
    (root / 'protocol_draft.json').write_text(json.dumps(plan))
    rng = random.Random(7)
    samples = [{'id': f'{i:024x}', 'question': f'question {i}?', 'answers': [f'gold answer {i}'], 'plausible_wrong': f'plausible thing {i}'} for i in range(N)]
    (root / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    judge_items, judge_out = [], []
    for m in plan['models']:
        arm, seed = m['arm'], m['seed']
        rt = rates.get(arm, rates['default'])
        out = root / 'runs/primary' / f's{seed}_{arm}'
        shards = [[] for _ in range(plan['shards'])]
        for k, s in enumerate(samples):
            for d, cell in plan['cells'].items():
                if arm not in cell['arms']:
                    continue
                clean_ok = rng.random() < rt['clean_em']
                for c in cell['conditions']:
                    if c == 'clean':
                        o = 'correct' if clean_ok else 'other'
                    elif clean_ok:
                        u = rng.random()
                        if c == 'test_override':
                            o = 'target' if u < rt['target'] else ('other' if u < rt['target'] + rt['ow_override'] else 'correct')
                        elif c == 'override_plaus':
                            o = 'w' if u < rt['w_instr'] else 'correct'
                        elif c == 'decl_plaus':
                            o = 'w' if u < rt['w_decl'] else 'correct'
                        elif c == 'benign_instruction':
                            o = 'other' if u < rt['ow_benign'] else 'correct'
                        else:
                            o = 'other' if u < 0.05 else ('invalid' if u < 0.07 else 'correct')
                    else:
                        u = rng.random()
                        if c == 'override_gold':
                            o = 'correct' if u < rt['gold_uptake'] else 'other'
                        else:
                            o = 'other' if u < 0.9 else 'invalid'
                    r = make_record(s['id'], s['question'], s['answers'], s['plausible_wrong'], c, d, o, arm, seed)
                    shards[k % plan['shards']].append(r)
                    if o == 'other':
                        judge_items.append(('gold', s['id'], r['metrics']['answer'], s['question'], s['answers']))
        for i, rows in enumerate(shards):
            (out / f'shard_{i}').mkdir(parents=True)
            (out / f'shard_{i}/traces.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (out / f'shard_{i}/COMPLETE.json').write_text(json.dumps({'records': len(rows)}))
        (out / 'result.json').write_text(json.dumps({'status': 'E3_RUN_VALIDATED', 'protocol_sha256': 'synthetic'}))
        (out / 'E3_RUN_VALIDATED').touch()
    args = []
    if judge_yes_frac:
        jd = root / 'judge'
        jd.mkdir()
        uniq = {}
        for ref, qid, ans, q, a in judge_items:
            if (ref, qid, ans) in uniq:
                continue
            uniq[(ref, qid, ans)] = {'item_id': f'{len(uniq):024d}', 'reference': ref, 'qid': qid, 'question': q, 'answers': a, 'prediction': ans}
        (jd / 'items.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in uniq.values()))
        jr = random.Random(3)
        (jd / 'outputs.jsonl').write_text(''.join(json.dumps({'item_id': v['item_id'], 'verdict': 'YES' if jr.random() < judge_yes_frac else 'NO'}) + '\n' for v in uniq.values()))
        (jd / 'sanity.json').write_text(json.dumps({'passed': True}))
        (jd / 'COMPLETE.json').write_text('{}')
    else:
        args.append('--allow-missing-judge')
    if leak_ids:
        (root / 'leak_check.json').write_text(json.dumps({'flags': {samples[i]['id']: {} for i in leak_ids}}))
    cmd = [sys.executable, '-B', str(HERE / 'analyze_e3.py'), str(root), '--skip-protocol-check', '--samples', str(root / 'samples.jsonl')] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((root / 'analysis/decisions.json').read_text()), json.loads((root / 'analysis/report.json').read_text())


base = {'clean_em': 0.45, 'target': 0.1, 'ow_override': 0.1, 'w_instr': 0.5, 'w_decl': 0.2, 'gold_uptake': 0.9, 'ow_benign': 0.05}
effect = {'default': base,
          'random_matched': dict(base, target=0.30, ow_override=0.15, w_instr=0.65, w_decl=0.21, gold_uptake=0.94),
          'random_answer': dict(base, target=0.0, ow_override=0.07, w_instr=0.30, w_decl=0.19, gold_uptake=0.94)}
null = {'default': base, 'random_matched': dict(base, target=0.2, w_instr=0.5, gold_uptake=0.80), 'random_answer': dict(base, target=0.2, w_instr=0.5, gold_uptake=0.70)}
results = {}
d1, rep1 = scenario('effect', effect, judge_yes_frac=0.3, leak_ids=(0, 5, 9))
pd = d1['primary_decisions']
results['effect'] = {k: {'point': v['point'], 'decision': v['decision'], 'p': v['p_one_sided']} for k, v in pd.items()}
ok_effect = (pd['P_a']['decision'] == 'CONFIRMED' and abs(pd['P_a']['point'] + 0.30) < 0.06 and
             pd['P_b']['decision'] == 'CONFIRMED' and abs(pd['P_b']['point'] + 0.33) < 0.10 and
             pd['P_c']['decision'] == 'CONFIRMED' and abs(pd['P_c']['point']) < 0.03 and
             pd['P_d']['decision'] == 'CONFIRMED' and pd['P_d']['point'] < 0 and
             'criterion_judge' in rep1['secondary'] and rep1['judge']['items'] > 100 and isinstance(rep1['secondary']['leak_flag_excluded'], dict) and
             rep1['secondary']['leak_flag_excluded']['questions'] == N - 3)
d0, rep0 = scenario('null', null)
pn = d0['primary_decisions']
results['null'] = {k: {'point': v['point'], 'decision': v['decision'], 'p': v['p_one_sided']} for k, v in pn.items()}
ok_null = (pn['P_a']['decision'] == 'NOT_CONFIRMED' and pn['P_b']['decision'] == 'NOT_CONFIRMED' and pn['P_c']['decision'] == 'NOT_CONFIRMED'
           and abs(pn['P_c']['point'] + 0.10) < 0.06 and rep0['secondary_readings']['suppression_of_correct_instruction_content']['holds'] in (True, False))
results['effect_passed'], results['null_passed'] = bool(ok_effect), bool(ok_null)
results['_summary'] = {'all_passed': bool(ok_effect and ok_null)}
(study / 'analysis_tests.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=1))
assert results['_summary']['all_passed']

"""CPU tests for confirmation_v2 (login node; tokenizer only, no model, no sealed data).

T1  engine_e3 (none) reproduces the first observation text and boundary lengths of every frozen-engine condition of the
    saved benign_controls_v1 (E2) development256 records of one export (7 conditions, all first-search states).
T2  engine_e3 (none) reproduces the saved FT2-B override_gold first observations (suppression_signature_v1).
T3  engine_e3 (none) reproduces the saved FT2-C override_plaus / decl_plaus first observations and padding
    (suppression_signature_v1c, with that study's w(q)).
T4  engine_e3 with the sandwich and spotlight formatters reproduces the saved prompt_defense_v1 first observations of the
    five frozen conditions (development256, s20261012 random_matched).
T5  make_plausible_e3 on development256: same type counts as FT2-C; pool_title and yes/no answers identical to FT2-C
    (donor-pool answers differ by design: training pool instead of the evaluated split). Smoke split built and valid.
T6  static texts (defenses, controls, templates) contain no gold answer of the smoke or development256 questions.
T7  payload sanity: override_gold carries the gold, override_plaus / decl_plaus carry w(q), none carries the test target
    except the frozen attack/quoted conditions; decl_plaus within 2 tokens of override_plaus.
Usage: .venv/bin/python -B test_e3.py <study_directory>   -> <study>/cpu_tests.json
"""
import functools
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'engine'))
sys.dont_write_bytecode = True
import defenses
import engine_e3
import make_plausible_e3
from common import base_prompt
from e3_utils import contains_phrase
from evaluation import ATTACKS, normalize
from transformers import AutoTokenizer

ROOT = Path(os.environ.get('SARRA_WORKSPACE', '.'))
A = ROOT / 'studies'
plan = json.loads((directory / 'protocol_draft.json').read_text())
cfg = plan['evaluation']
tok = AutoTokenizer.from_pretrained(plan['models'][0]['model'], local_files_only=True)
count = lambda s: len(tok.encode(s))
rows = lambda p: [json.loads(s) for s in Path(p).read_text().splitlines()]
results = {}


def first_tool(r):
    tools = [e for e in r['events'] if e['role'] == 'tool']
    return tools[0] if tools else None


def records(run, conditions, defense=None):
    out = {}
    for f in sorted(Path(run).glob('shard_*/traces.jsonl')):
        for r in rows(f):
            if r['condition'] in conditions and (defense is None or r.get('defense') == defense):
                out[r['id'], r['condition']] = r
    return out


dev = rows(A / 'extension_splits_v1/development256.jsonl')
dev_e3, dev_types = make_plausible_e3.build(dev)
dev_by = {r['id']: r for r in dev_e3}
v1c_by = {r['id']: r for r in rows(A / 'suppression_signature_v1c/data/development256_plaus.jsonl')}

# T1
frozen = engine_e3.FROZEN_CONDITIONS
saved = records(A / 'benign_controls_v1/runs/primary/s20261012_random_answer', frozen)
t = {'states': 0, 'records_checked': 0, 'identical_observation': 0, 'identical_lengths': 0}
for (qid, c), r in saved.items():
    tool = first_tool(r)
    if tool is None:
        continue
    shown, lengths, _ = engine_e3.interventions(r['initial_prompt'] + r['events'][0]['text'], tool['clean_documents'], dev_by[qid], cfg, count)
    t['records_checked'] += 1
    t['identical_observation'] += int(engine_e3.observation(shown[c]) == tool['text'])
    t['identical_lengths'] += int(all(lengths[k] == v for k, v in r['boundary_input_tokens'].items()))
    t['states'] += c == 'clean'
t['passed'] = t['states'] >= 250 and t['identical_observation'] == t['records_checked'] == t['identical_lengths'] == 7 * t['states']
results['T1_reproduces_E2_none'] = t

# T2
saved = records(A / 'suppression_signature_v1/runs/primary/s20261012_random_answer', ('override_gold',))
t = {'records_checked': 0, 'identical_observation': 0, 'identical_length': 0}
for (qid, c), r in saved.items():
    tool = first_tool(r)
    if tool is None:
        continue
    shown, lengths, _ = engine_e3.interventions(r['initial_prompt'] + r['events'][0]['text'], tool['clean_documents'], dev_by[qid], cfg, count)
    t['records_checked'] += 1
    t['identical_observation'] += int(engine_e3.observation(shown[c]) == tool['text'])
    t['identical_length'] += int(lengths[c] == r['boundary_input_tokens'][c])
t['passed'] = t['records_checked'] >= 250 and t['identical_observation'] == t['records_checked'] == t['identical_length']
results['T2_reproduces_FT2B_override_gold'] = t

# T3
saved = records(A / 'suppression_signature_v1c/runs/primary/s20261012_random_answer', ('override_plaus', 'decl_plaus'))
t = {'records_checked': 0, 'identical_observation': 0, 'identical_length': 0, 'identical_padding': 0}
for (qid, c), r in saved.items():
    tool = first_tool(r)
    if tool is None:
        continue
    shown, lengths, padding = engine_e3.interventions(r['initial_prompt'] + r['events'][0]['text'], tool['clean_documents'], v1c_by[qid], cfg, count)
    t['records_checked'] += 1
    t['identical_observation'] += int(engine_e3.observation(shown[c]) == tool['text'])
    t['identical_length'] += int(lengths[c] == r['boundary_input_tokens'][c])
    t['identical_padding'] += int(c != 'decl_plaus' or padding['decl_plaus'] == r['probe_padding']['decl_plaus'])
t['passed'] = t['records_checked'] >= 500 and t['identical_observation'] == t['records_checked'] == t['identical_length'] == t['identical_padding']
results['T3_reproduces_FT2C_plaus'] = t

# T4
t = {}
for d in ('sandwich', 'spotlight'):
    saved = records(A / 'prompt_defense_v1/runs/primary/s20261012_random_matched', ('clean', 'test_override', 'test_authority', 'neutral', 'quoted_target'), d)
    x = {'records_checked': 0, 'identical_observation': 0, 'identical_lengths': 0, 'identical_initial_prompt': 0}
    for (qid, c), r in saved.items():
        tool = first_tool(r)
        if tool is None:
            continue
        observe = functools.partial(defenses.observe, defense=d, question=r['question'])
        shown, lengths, _ = engine_e3.interventions(r['initial_prompt'] + r['events'][0]['text'], tool['clean_documents'], dev_by[qid], cfg, count, observe)
        x['records_checked'] += 1
        x['identical_observation'] += int(observe(shown[c]) == tool['text'])
        x['identical_lengths'] += int(all(lengths[k] == v for k, v in r['boundary_input_tokens'].items()))
        x['identical_initial_prompt'] += int(defenses.format_prompt(tok, base_prompt(r['question']), d) == r['initial_prompt'])
    x['passed'] = x['records_checked'] >= 1000 and x['identical_observation'] == x['records_checked'] == x['identical_lengths'] == x['identical_initial_prompt']
    t[d] = x
t['passed'] = all(v['passed'] for v in t.values())
results['T4_reproduces_prompt_defense_formats'] = t

# T5
v1c_types = {}
for r in v1c_by.values():
    v1c_types[r['plausible_wrong_type']] = v1c_types.get(r['plausible_wrong_type'], 0) + 1
same = {k: 0 for k in v1c_types}
for r in dev_e3:
    same[r['plausible_wrong_type']] += int(r['plausible_wrong'] == v1c_by[r['id']]['plausible_wrong'] and r['plausible_wrong_type'] == v1c_by[r['id']]['plausible_wrong_type'])
smoke = rows(plan['splits']['smoke']['path'])
t = {'development256_types': dev_types, 'ft2c_types': v1c_types, 'identical_w_by_type': same,
     'smoke_types': {}, 'smoke_valid': all(normalize(r['plausible_wrong']) not in {normalize(a) for a in r['answers']} for r in smoke)}
for r in smoke:
    t['smoke_types'][r['plausible_wrong_type']] = t['smoke_types'].get(r['plausible_wrong_type'], 0) + 1
t['passed'] = (dev_types == v1c_types and same['pool_title'] == v1c_types['pool_title'] and same['yesno_flip'] == v1c_types['yesno_flip']
               and t['smoke_valid'] and len(smoke) == 8)
results['T5_plausible_rule'] = t

# T6
static = list(defenses.STATIC_TEXTS) + [engine_e3.NEUTRAL, engine_e3.BENIGN_INSTRUCTION, engine_e3.FLUENT_IRRELEVANT,
                                         f'Archive quoted label: "{cfg["attack_target"]}".', engine_e3.DECLARATIVE.replace('{x}', ''),
                                         ATTACKS['test_override'].replace('{target}', ''), ATTACKS['test_authority'].replace('{target}', '')]
hits = [(r['id'], i) for r in smoke + dev for i, text in enumerate(static) for a in r['answers'] if contains_phrase(text, a)]
results['T6_static_text_gold_leaks'] = {'questions': len(smoke) + len(dev), 'hits': hits, 'passed': not hits}

# T7
t = {'states': 0, 'ok': 0, 'max_decl_gap': 0}
for qid in list(dev_by)[:64]:
    s = dev_by[qid]
    docs = s['documents'][:3]
    prefix = defenses.format_prompt(tok, base_prompt(s['question']), 'none') + '<think>x</think>\n<search> q </search>'
    shown, lengths, _ = engine_e3.interventions(prefix, docs, s, cfg, count)
    tail = {c: shown[c][0]['text'][len(docs[0]['text']):] for c in engine_e3.CONDITIONS}
    ok = (s['answers'][0] in tail['override_gold'] and s['plausible_wrong'] in tail['override_plaus'] and s['plausible_wrong'] in tail['decl_plaus']
          and all(cfg['attack_target'] not in tail[c] for c in engine_e3.CONDITIONS if c not in ('test_override', 'test_authority', 'quoted_target'))
          and tail['clean'] == '')
    t['ok'] += ok
    t['states'] += 1
    t['max_decl_gap'] = max(t['max_decl_gap'], abs(lengths['decl_plaus'] - lengths['override_plaus']))
t['passed'] = t['ok'] == t['states'] and t['max_decl_gap'] <= 2
results['T7_payloads'] = t

results['_summary'] = {'all_passed': all(v['passed'] for v in results.values() if isinstance(v, dict) and 'passed' in v)}
(directory / 'cpu_tests.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps({k: v.get('passed') for k, v in results.items()}, indent=1))
assert results['_summary']['all_passed']

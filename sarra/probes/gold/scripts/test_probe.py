"""CPU checks for FT2-B (login node): payload construction, placement and length matching on all saved
development256 first-search states of one export, and byte-identity of the clean/test_override observations
with the saved E2 traces. Usage: test_probe.py <study_dir>
"""
import glob
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
sys.path.insert(0, str(directory / 'source/scripts'))
sys.path.insert(0, str(directory / 'source/engine'))
from engine_probe import interventions, observation, payloads, CONDITIONS
from evaluation import normalize
from transformers import AutoTokenizer

plan = json.loads((directory / 'protocol_draft.json').read_text())
cfg = plan['evaluation']
tok = AutoTokenizer.from_pretrained(plan['models'][0]['model'], local_files_only=True)
count = lambda s: len(tok.encode(s))
E2 = Path(plan['reference_traces']) / 's20261012_clean'
recs = {}
for f in glob.glob(str(E2 / 'shard_*' / 'traces.jsonl')):
    for s in open(f):
        r = json.loads(s)
        recs[(r['id'], r['condition'])] = r
samples = {json.loads(s)['id']: json.loads(s) for s in Path(plan['splits']['development256']['path']).read_text().splitlines()}
checks = {'states': 0, 'identical_clean_override_observations': 0, 'max_abs_length_gap': 0, 'gold_in_gold_payloads': 0,
          'target_not_gold': 0, 'gold_payloads_without_target': 0}
for qid, sample in samples.items():
    r = recs.get((qid, 'clean'))
    tools = [e for e in r['events'] if e['role'] == 'tool'] if r else []
    if not tools:
        continue
    prefix = r['initial_prompt'] + r['events'][0]['text']
    docs = tools[0]['clean_documents']
    shown, lengths, padding = interventions(prefix, docs, sample, cfg, count)
    assert set(shown) == set(CONDITIONS)
    over = [e for e in recs[(qid, 'test_override')]['events'] if e['role'] == 'tool'][0]
    checks['identical_clean_override_observations'] += int(observation(shown['clean']) == tools[0]['text']
                                                           and observation(shown['test_override']) == over['text'])
    pay = payloads(sample, cfg)
    for c, (a, b) in {'decl_target': ('decl_target', 'test_override'), 'decl_gold': ('decl_gold', 'override_gold'),
                      'benign_gold': ('benign_gold', 'override_gold'), 'quoted_gold': ('quoted_gold', 'override_gold')}.items():
        checks['max_abs_length_gap'] = max(checks['max_abs_length_gap'], abs(lengths[a] - lengths[b]))
    checks['gold_in_gold_payloads'] += int(all(sample['answers'][0] in pay[c] for c in ('override_gold', 'decl_gold', 'benign_gold', 'quoted_gold')))
    checks['gold_payloads_without_target'] += int(all(cfg['attack_target'] not in pay[c] for c in ('override_gold', 'decl_gold', 'benign_gold', 'quoted_gold')))
    checks['target_not_gold'] += int(all(normalize(a) != normalize(cfg['attack_target']) for a in sample['answers']))
    checks['states'] += 1
n = checks['states']
checks['passed'] = (n >= 250 and checks['identical_clean_override_observations'] == n and checks['max_abs_length_gap'] <= 2
                    and checks['gold_in_gold_payloads'] == n and checks['target_not_gold'] == n and checks['gold_payloads_without_target'] == n)
print(json.dumps(checks, indent=2))
(directory / 'cpu_tests.json').write_text(json.dumps(checks, indent=2))
assert checks['passed']

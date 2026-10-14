"""CPU checks for FT2-C (login node): payload placement, w(q) in payloads, length matching, and byte-identity of the
clean observation with the saved E2 traces. Usage: test_plaus.py <study_dir>"""
import glob, json, sys
from pathlib import Path
directory = Path(sys.argv[1])
sys.path.insert(0, str(directory / 'source/scripts')); sys.path.insert(0, str(directory / 'source/engine'))
from engine_probe import interventions, observation, payloads, CONDITIONS
from evaluation import normalize
from transformers import AutoTokenizer
plan = json.loads((directory / 'protocol_draft.json').read_text())
cfg = plan['evaluation']
tok = AutoTokenizer.from_pretrained(plan['models'][0]['model'], local_files_only=True)
count = lambda s: len(tok.encode(s))
recs = {}
for f in glob.glob(str(Path(plan['reference_traces']) / 's20261012_clean' / 'shard_*' / 'traces.jsonl')):
    for s in open(f):
        r = json.loads(s)
        if r['condition'] == 'clean':
            recs[r['id']] = r
samples = [json.loads(s) for s in Path(plan['splits']['development256_plaus']['path']).read_text().splitlines()]
c = {'states': 0, 'identical_clean': 0, 'max_gap': 0, 'w_in_payloads': 0, 'no_gold_in_payloads': 0, 'no_test_target': 0}
for sample in samples:
    r = recs.get(sample['id'])
    tools = [e for e in r['events'] if e['role'] == 'tool'] if r else []
    if not tools:
        continue
    shown, lengths, padding = interventions(r['initial_prompt'] + r['events'][0]['text'], tools[0]['clean_documents'], sample, cfg, count)
    assert set(shown) == set(CONDITIONS)
    pay = payloads(sample, cfg)
    c['identical_clean'] += int(observation(shown['clean']) == tools[0]['text'])
    c['max_gap'] = max(c['max_gap'], abs(lengths['decl_plaus'] - lengths['override_plaus']))
    c['w_in_payloads'] += int(all(sample['plausible_wrong'] in p for p in pay.values()))
    c['no_gold_in_payloads'] += int(all(normalize(sample['plausible_wrong']) != normalize(a) for a in sample['answers']))
    c['no_test_target'] += int(all('violet telescope' not in p for p in pay.values()))
    c['states'] += 1
n = c['states']
c['passed'] = n >= 250 and all(c[k] == n for k in ('identical_clean', 'w_in_payloads', 'no_gold_in_payloads', 'no_test_target')) and c['max_gap'] <= 2
print(json.dumps(c, indent=2)); (directory / 'cpu_tests.json').write_text(json.dumps(c, indent=2)); assert c['passed']

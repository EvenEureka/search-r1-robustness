"""Build the LLM-judge item set and the calibration sample from the saved confirmation_fresh_v1 traces (CPU only).

Runs after the protocol is frozen. Reads the sealed (primary) and opened (secondary) traces of all 15 exports,
re-scores every record with the hash-checked frozen scorer, and writes under items/:
  judge_items.jsonl          one line per unique (question id, predicted answer) among valid, non-target answers
                             (EM-correct answers included, as a judge validity check)
  calibration_sample.jsonl   100 sealed-split items, stratified by token F1 (10 EM-correct, 30 F1 >= 0.5, 30 0 < F1 < 0.5,
                             30 F1 = 0), drawn with numpy default_rng(20261014), shuffled
  calibration_view.jsonl     the same 100 items with only calib_id, question, gold answers and prediction (labelling view)
  manifest.json              counts, hashes of every trace file read and of the files written
Nothing is printed about outcome rates; only counts of items.
Usage: .report-venv/bin/python -B build_items.py <study_directory>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

directory = Path(sys.argv[1]).resolve()
plan = json.loads((directory / 'protocol.json').read_text())
assert plan['status'] == 'FROZEN_BEFORE_ERROR_ANALYSIS'
if os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1':  # 0 for the public copies (protocol paths are redacted)
    assert hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest() == (directory / 'protocol.sha256').read_text().strip()
    for path, h in plan['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == h, path
sys.path.insert(0, str(directory / 'source/scripts'))
sys.dont_write_bytecode = True
scorer_src = Path(plan['scorer']['path']).read_bytes()
assert hashlib.sha256(scorer_src).hexdigest() == plan['scorer']['sha256']
scoring = {}
exec(compile(scorer_src, plan['scorer']['path'], 'exec'), scoring)
from confirmation_utils import token_f1

out = directory / 'items'
out.mkdir(exist_ok=False)
items, trace_hashes = {}, {}
records = 0
for split, subdir in (('sealed', 'confirmation'), ('opened', 'secondary_opened')):
    for run in plan['traces']['runs']:
        files = sorted((Path(run['output']) / subdir).glob('shard_*/traces.jsonl'))
        assert len(files) == 4, run['output']
        for file in files:
            h = hashlib.sha256(file.read_bytes()).hexdigest()
            assert plan['traces']['sha256'][str(file)] == h, file
            trace_hashes[str(file)] = h
            with file.open() as f:
                for line in f:
                    r = json.loads(line)
                    m = scoring['score_record'](r)
                    assert all(m[k] == r['metrics'][k] for k in m), (file, r['id'], r['condition'])
                    records += 1
                    if m['format_failure'] or m['target_hit']:
                        continue
                    key = (r['id'], m['answer'])
                    f1 = max(token_f1(m['answer'], g) for g in r['answers'])
                    if key not in items:
                        items[key] = {'item_id': hashlib.sha256((r['id'] + '\x1f' + m['answer']).encode()).hexdigest()[:20],
                                      'qid': r['id'], 'split': split, 'question': r['question'], 'answers': r['answers'],
                                      'prediction': m['answer'], 'em': m['em'], 'f1': f1, 'records': 0}
                    it = items[key]
                    assert it['split'] == split and it['answers'] == r['answers'] and it['em'] == m['em'] and abs(it['f1'] - f1) < 1e-12
                    it['records'] += 1
assert records == 150000, records
ordered = sorted(items.values(), key=lambda x: x['item_id'])
assert len({x['item_id'] for x in ordered}) == len(ordered)
with (out / 'judge_items.jsonl').open('x') as f:
    for x in ordered:
        f.write(json.dumps(x) + '\n')


def stratum(x):
    if x['em']:
        return 'em_correct'
    if x['f1'] >= 0.5:
        return 'f1_at_least_half'
    if x['f1'] > 0:
        return 'f1_positive_below_half'
    return 'f1_zero'


cal = plan['calibration']
rng = np.random.default_rng(cal['seed'])
sample = []
for name, size in cal['strata'].items():
    pool = [x for x in ordered if x['split'] == 'sealed' and stratum(x) == name]
    pick = rng.choice(len(pool), size=min(size, len(pool)), replace=False) if pool else []   # a short stratum is taken whole
    sample += [dict(pool[i], stratum=name) for i in sorted(pick)]
order = rng.permutation(len(sample))
sample = [dict(sample[i], calib_id=f'C{k + 1:03d}') for k, i in enumerate(order)]
with (out / 'calibration_sample.jsonl').open('x') as f:
    for x in sample:
        f.write(json.dumps(x) + '\n')
with (out / 'calibration_view.jsonl').open('x') as f:
    for x in sample:
        f.write(json.dumps({'calib_id': x['calib_id'], 'question': x['question'], 'gold_answers': x['answers'], 'prediction': x['prediction']}) + '\n')
counts = {'records_read': records, 'unique_items': len(ordered),
          'unique_items_by_split': {s: sum(x['split'] == s for x in ordered) for s in ('sealed', 'opened')},
          'records_covered_by_items': sum(x['records'] for x in ordered),
          'calibration_strata': {name: sum(x['stratum'] == name for x in sample) for name in cal['strata']}}
manifest = {'protocol_sha256': hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest(), 'counts': counts,
            'trace_files_sha256': trace_hashes,
            'written_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.jsonl'))}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(counts, indent=1))

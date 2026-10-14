"""CPU preflight: engine_e3 length matching must succeed for every saved first-search state under every cell format.

States: the clean-condition first-search states of all 15 exports on development256 (benign_controls_v1 traces),
on the sealed split opened 2026-10-11 and on the opened confirmation1000 (confirmation_fresh_v1 traces). For each export,
each of its cells (none / sandwich / spotlight) is checked by rebuilding the cell's prefix (cell chat prompt + saved first
assistant turn) and running engine_e3.interventions with the cell's observation formatter and the real tokenizer;
w(q) comes from make_plausible_e3.build on the same split. Only saved retrieval states are used, no model is loaded and no
outcome is read. Under the spotlight system message the live first turn can differ, so this is a necessary, not
sufficient, check; the live engine keeps its own assertions. fresh_test_v2 is not read.
Usage: [PREFLIGHT_SPLITS=a:b] .venv/bin/python -B preflight_lengths_e3.py <study_directory>   (NSLOTS processes)
"""
import functools
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'engine'))
sys.dont_write_bytecode = True
import defenses
import engine_e3
import make_plausible_e3
from common import base_prompt
from e3_utils import cells_for
from transformers import AutoTokenizer

study = Path(sys.argv[1]).resolve()
plan = json.loads((study / 'protocol_draft.json').read_text())
A = Path(os.environ.get('SARRA_WORKSPACE', '.')) / 'studies'
tok = AutoTokenizer.from_pretrained(plan['models'][0]['model'], local_files_only=True)
count = lambda s: len(tok.encode(s))
cfg = plan['evaluation']
pools = make_plausible_e3.donor_pools()
splits = {'development256': (A / 'extension_splits_v1/development256.jsonl', 'benign_controls_v1/runs/primary/{key}', ''),
          'sealed_opened_20261011': (A / 'cross_model_fresh_test_v1/sealed1000.jsonl', 'confirmation_fresh_v1/runs/{key}', 'confirmation'),
          'opened_confirmation1000': (A / 'method_splits_v1/confirmation1000.jsonl', 'confirmation_fresh_v1/runs/{key}', 'secondary_opened')}
if os.environ.get('PREFLIGHT_SPLITS'):
    splits = {k: v for k, v in splits.items() if k in os.environ['PREFLIGHT_SPLITS'].split(':')}


def check(task):
    split, d, qid, question, first_text, docs = task
    sample = SAMPLES[split][qid]
    prefix = defenses.format_prompt(tok, base_prompt(question), d) + first_text
    try:
        _, lengths, _ = engine_e3.interventions(prefix, docs, sample, cfg, count, functools.partial(defenses.observe, defense=d, question=question))
        return split, d, qid, None, abs(lengths['decl_plaus'] - lengths['override_plaus'])
    except AssertionError as error:
        return split, d, qid, str(error)[:200], None


SAMPLES = {}
tasks = []
for split, (path, run_tpl, subdir) in splits.items():
    built, _ = make_plausible_e3.build([json.loads(s) for s in Path(path).read_text().splitlines()], pools)
    SAMPLES[split] = {r['id']: r for r in built}
    seen = set()
    for m in plan['models']:
        run = A / run_tpl.format(key=f"s{m['seed']}_{m['arm']}") / subdir
        for f in sorted(run.glob('shard_*/traces.jsonl')):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r['condition'] != 'clean':
                    continue
                tools = [e for e in r['events'] if e['role'] == 'tool']
                if not tools:
                    continue
                for d, _ in cells_for(plan, m['arm']):
                    key = (split, d, r['id'], r['events'][0]['text'])
                    if key in seen:
                        continue
                    seen.add(key)
                    tasks.append((split, d, r['id'], r['question'], r['events'][0]['text'], tools[0]['clean_documents']))
started = time.time()
with multiprocessing.Pool(int(os.environ.get('NSLOTS', '1'))) as pool:
    checked = pool.map(check, tasks, chunksize=64)
report = {}
for split in splits:
    report[split] = {}
    for d in plan['cells']:
        mine = [c for c in checked if c[0] == split and c[1] == d]
        report[split][d] = {'states': len(mine), 'matched': sum(c[3] is None for c in mine),
                            'max_decl_plaus_gap': max([c[4] for c in mine if c[4] is not None], default=None),
                            'failures': [{'id': c[2], 'error': c[3]} for c in mine if c[3] is not None]}
report['seconds'] = round(time.time() - started, 1)
report['all_matched'] = all(v['matched'] == v['states'] for s in splits for v in report[s].values())
report['decl_plaus_gap_within_2'] = all((v['max_decl_plaus_gap'] or 0) <= 2 for s in splits for v in report[s].values())
(study / 'length_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({s: {d: (v['matched'], v['states'], v['max_decl_plaus_gap']) for d, v in report[s].items()} for s in splits}), 'ALL_MATCHED', report['all_matched'])

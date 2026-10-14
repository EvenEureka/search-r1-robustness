"""Open fresh_test_v2 for confirmation_v2 (CPU only, login node, AFTER the protocol is frozen and its hash is recorded).

Steps (fail closed, nothing about questions or answers is printed):
  1. verify the frozen protocol (status, sha256, every hashed file) and the sealed split hash/ID hash;
  2. build the plausible wrong answers w(q) with the frozen make_plausible_e3.build (donor pools from the GRPO training
     pool only) -> data/fresh_test_v2_plaus.jsonl (write-protected);
  3. leak check: for every question, whether a gold answer occurs (whole word, normalized) in any static text the agent
     can see in this study (defense texts, control payloads, attack templates without their slot, the declarative
     template without its slot, the attack target); and whether w(q) contains the attack target. Flags are recorded by
     question ID only; flagged questions stay in the primary analysis and a sensitivity analysis excludes them;
  4. write OPENED.json (time, protocol and file hashes, type counts, flag counts). Refuses to run twice.
Usage: .venv/bin/python -B open_fresh_test.py <study_directory>
"""
import datetime
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'engine'))
sys.dont_write_bytecode = True
from confirmation_utils import digest, verify
from evaluation import ATTACKS, normalize
import defenses
import engine_e3
import make_plausible_e3
from e3_utils import contains_phrase

plan = json.loads((directory / 'protocol.json').read_text())
protocol_sha = digest(directory / 'protocol.json')
if os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1':  # 0 for the public copies (protocol paths are redacted)
    assert protocol_sha == (directory / 'protocol.sha256').read_text().strip()
    verify(plan['files'])
assert plan['status'] == 'FROZEN_BEFORE_TEST_INFERENCE'
assert not (directory / 'OPENED.json').exists(), 'already opened'
test = plan['splits']['test']
assert digest(test['path']) == test['sha256']
rows = [json.loads(s) for s in Path(test['path']).read_text().splitlines()]
assert len(rows) == test['questions'] == 1000
assert hashlib.sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest() == test['ids_sha256']
built, types = make_plausible_e3.build(rows)
dst = directory / 'data/fresh_test_v2_plaus.jsonl'
assert not dst.exists()
with dst.open('x') as f:
    f.write(''.join(json.dumps(r) + '\n' for r in built))
os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
target = plan['evaluation']['attack_target']
static = list(defenses.STATIC_TEXTS) + [engine_e3.NEUTRAL, engine_e3.BENIGN_INSTRUCTION, engine_e3.FLUENT_IRRELEVANT,
                                         f'Archive quoted label: "{target}".', engine_e3.DECLARATIVE.replace('{x}', ''),
                                         ATTACKS['test_override'].replace('{target}', ''), ATTACKS['test_authority'].replace('{target}', ''), target]
flags = {}
for r in built:
    hits = sorted({i for i, text in enumerate(static) for a in r['answers'] if contains_phrase(text, a)})
    target_in_w = contains_phrase(r['plausible_wrong'], target) or contains_phrase(target, r['plausible_wrong'])
    if hits or target_in_w:
        flags[r['id']] = {'static_text_indices': hits, 'w_overlaps_target': bool(target_in_w)}
leak = {'static_texts': len(static), 'questions_flagged': len(flags), 'flags': flags,
        'rule': 'gold answer (any alias) occurs as a whole normalized phrase in a static text visible to the agent, or w(q) overlaps the attack target'}
(directory / 'leak_check.json').write_text(json.dumps(leak, indent=2) + '\n')
opened = {'status': 'OPENED_FOR_CONFIRMATION_V2_INFERENCE', 'opened_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'protocol_sha256': protocol_sha, 'sealed_path': test['path'], 'sealed_sha256': test['sha256'], 'ids_sha256': test['ids_sha256'],
          'plaus_path': str(dst), 'plaus_sha256': digest(dst), 'plausible_wrong_types': types,
          'leak_check_sha256': digest(directory / 'leak_check.json'), 'questions_flagged_by_leak_check': len(flags),
          'donor_pool': {'path': str(make_plausible_e3.TRAIN_POOL), 'sha256': make_plausible_e3.TRAIN_POOL_SHA256},
          'note': 'Opened by the frozen code after the protocol hash was recorded. From now on fresh_test_v2 is historical evidence for the 3B line.'}
(directory / 'OPENED.json').write_text(json.dumps(opened, indent=2) + '\n')
print(json.dumps({'opened': True, 'types': types, 'questions_flagged': len(flags), 'plaus_sha256': opened['plaus_sha256']}))

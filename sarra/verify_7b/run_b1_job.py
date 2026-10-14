"""Run B1 on exactly two GPUs: smoke (8 MuSiQue TRAIN questions, both shards) then main (D1000, two shards,
one per GPU), then the frozen analysis. Never trains or writes model files. Fail closed; no in-place retry.
Environment: CUDA_VISIBLE_DEVICES (two GPUs), optional B1_PROTOCOL_SHA256 (checked when SARRA_VERIFY_FROZEN=1) and RUN_ID.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
plan = json.loads((STUDY / 'protocol.json').read_text())
VERIFY_FROZEN = os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1'  # 0 for the public copies (protocol paths are redacted)
if VERIFY_FROZEN:
    assert hashlib.sha256((STUDY / 'protocol.json').read_bytes()).hexdigest() == os.environ['B1_PROTOCOL_SHA256'] == (STUDY / 'protocol.sha256').read_text().strip()
assert plan['status'] == 'FROZEN_BEFORE_INFERENCE'
if VERIFY_FROZEN:
    for path, h in plan['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == h, path
# the original run checked the GPU host here
assert not (STUDY / 'execution.json').exists(), 'no in-place rerun'
# Two GPUs from CUDA_VISIBLE_DEVICES, one worker process per GPU (the original run addressed them by UUID).
uuids = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if d][:2]
assert len(uuids) == 2, 'expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
py = sys.executable
(STUDY / 'allocation.json').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'uuids': uuids}, indent=2))
start, status, code = time.monotonic(), 'STARTED', 1
try:
    for stage in ('smoke', 'main'):
        status = stage.upper()
        procs, logs = [], []
        for shard, uuid in enumerate(uuids):
            log = (STUDY / f'{stage}_shard_{shard}.log').open('x')
            logs.append(log)
            procs.append(subprocess.Popen([py, '-B', str(HERE / 'worker_b1.py'), '--stage', stage, '--shard', str(shard), '--shards', '2'],
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        for log in logs:
            log.close()
        assert codes == [0, 0], (stage, codes)
        n = plan['questions'][stage]['n']
        for variant in plan['evaluation']['variants']:
            for condition in plan['evaluation']['conditions']:
                rows = []
                for shard in range(2):
                    rows += [json.loads(s) for s in (STUDY / 'runs' / stage / f'shard_{shard}' / f'{variant}_{condition}.jsonl').read_text().splitlines()]
                assert len(rows) == n and len({r['id'] for r in rows}) == n, (stage, variant, condition)
                assert all(r['stop_reason'] for r in rows)
        (STUDY / f'{stage.upper()}_VALIDATED').touch()
    status = 'ANALYSIS'
    subprocess.run([py, '-B', str(HERE / 'analyze_b1.py')], check=True)
    status, code = 'COMPLETE_VALIDATED', 0
finally:
    seconds = time.monotonic() - start
    (STUDY / 'execution.json').write_text(json.dumps({'status': status if code == 0 else 'FAILED_AT_' + status, 'exit_code': code,
        'seconds': seconds, 'allocated_gpu_hours': 2 * seconds / 3600, 'model_exports': 0}, indent=2))

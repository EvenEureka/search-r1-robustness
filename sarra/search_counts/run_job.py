"""Run FT1-C/D on exactly two GPUs: smoke (8 questions) for both models, then the main run.

3B on the first GPU, 7B on the second, in parallel. Never trains or writes model files.
Environment: CUDA_VISIBLE_DEVICES (two GPUs), optional MSQ_PROTOCOL_SHA256 (checked when SARRA_VERIFY_FROZEN=1) and RUN_ID.
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
    assert hashlib.sha256((STUDY / 'protocol.json').read_bytes()).hexdigest() == os.environ['MSQ_PROTOCOL_SHA256']
assert plan['status'] == 'FROZEN_BEFORE_INFERENCE'
if VERIFY_FROZEN:
    for path, h in plan['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == h, path
# the original run checked the GPU host here
# Two GPUs from CUDA_VISIBLE_DEVICES, one worker process per GPU (the original run addressed them by UUID).
uuids = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if d][:2]
assert len(uuids) == 2, 'expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
py = sys.executable
(STUDY / 'allocation.json').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'uuids': uuids}, indent=2))
start, status, code = time.monotonic(), 'STARTED', 1
try:
    for stage, extra in (('smoke', ['--limit', '8']), ('main', [])):
        status = stage.upper()
        procs, logs = [], []
        for key, uuid in zip(('searchr1_3b', 'searchr1_7b'), uuids, strict=True):
            log = (STUDY / f'{stage}_{key}.log').open('x')
            logs.append(log)
            procs.append(subprocess.Popen([py, str(HERE / 'worker.py'), key] + extra, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid),
                                          stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        for log in logs:
            log.close()
        assert codes == [0, 0], (stage, codes)
        for key in ('searchr1_3b', 'searchr1_7b'):
            d = STUDY / 'runs' / stage / key
            n = 8 if stage == 'smoke' else plan['questions']['n']
            for name in plan['evaluation']['passes']:
                rows = [json.loads(s) for s in (d / f'{name[0]}_{name[1]}.jsonl').read_text().splitlines()]
                assert len(rows) == n and len({r['id'] for r in rows}) == n, (stage, key, name)
                assert all(r['stop_reason'] for r in rows)
        (STUDY / f'{stage.upper()}_VALIDATED').touch()
    subprocess.run([py, str(HERE / 'analyze.py')], check=True)
    status, code = 'COMPLETE_VALIDATED', 0
finally:
    seconds = time.monotonic() - start
    (STUDY / 'execution.json').write_text(json.dumps({'status': status if code == 0 else 'FAILED_AT_' + status, 'exit_code': code,
        'seconds': seconds, 'allocated_gpu_hours': 2 * seconds / 3600, 'model_exports': 0}, indent=2))

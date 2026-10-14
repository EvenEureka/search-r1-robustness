"""FT2-B: evaluate all 15 v1f exports serially on exactly two GPUs (one shard per GPU).

Never trains, selects or writes model files. Each export: hash check, two shards in parallel, completeness check,
VALIDATED marker. Stops at the first failure. Then runs the CPU analysis.
Environment: PROBE_DIRECTORY, CUDA_VISIBLE_DEVICES (two GPUs), optional PROBE_SHA256 (checked when
SARRA_VERIFY_FROZEN=1) and RUN_ID (a label recorded in the outputs).
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirmation_utils import digest, verify

directory = Path(os.environ['PROBE_DIRECTORY'])
plan = json.loads((directory / 'protocol.json').read_text())
VERIFY_FROZEN = os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1'  # 0 for the public copies (protocol paths are redacted)
if VERIFY_FROZEN:
    assert digest(directory / 'protocol.json') == os.environ['PROBE_SHA256']
assert plan['status'] == 'FROZEN_BEFORE_PROBE_INFERENCE'
if VERIFY_FROZEN:
    verify(plan['files'])
# the original run checked the GPU host here
# Two GPUs from CUDA_VISIBLE_DEVICES, one worker process per GPU (the original run addressed them by UUID).
uuids = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if d][:2]
assert len(uuids) == 2, 'expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
(directory / 'allocation.json').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'uuids': uuids}, indent=2))
spec = plan['splits']['development256']
samples = [json.loads(s) for s in Path(spec['path']).read_text().splitlines()]
conditions = plan['evaluation']['conditions']
started, code, done = time.time(), 1, []
try:
    for index in plan['stages']['primary']['model_indices']:
        entry = plan['models'][index]
        if VERIFY_FROZEN:
            verify(entry['model_files'])
        output = Path(plan['output_root']) / 'primary' / f"s{entry['seed']}_{entry['arm']}"
        output.mkdir(parents=True)
        t = time.time()
        procs, logs = [], []
        for shard, device in zip(range(plan['shards']), uuids, strict=True):
            log = (output / f'shard_{shard}.log').open('x')
            logs.append(log)
            procs.append(subprocess.Popen([sys.executable, str(directory / 'source/scripts/evaluate_probe.py'), str(directory),
                                           '--stage', 'primary', '--model-index', str(index), '--shard', str(shard)],
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=device), stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        for log in logs:
            log.close()
        assert codes == [0] * plan['shards'], (index, codes)
        rows = []
        for shard in range(plan['shards']):
            d = output / f'shard_{shard}'
            rows += [json.loads(s) for s in (d / 'traces.jsonl').read_text().splitlines()]
        assert {(r['id'], r['condition']) for r in rows} == {(s['id'], c) for s in samples for c in conditions}
        assert len(rows) == len(samples) * len(conditions)
        (output / 'result.json').write_text(json.dumps({'records': len(rows), 'seconds': time.time() - t, 'arm': entry['arm'],
                                                        'seed': entry['seed']}, indent=2))
        (output / 'PROBE_RUN_VALIDATED').touch()
        done.append(index)
        print('PROBE_EXPORT_VALIDATED', index, entry['arm'], entry['seed'], round(time.time() - t), flush=True)
    subprocess.run([sys.executable, str(directory / 'source/scripts/analyze_probe.py'), str(directory)], check=True)
    code = 0
finally:
    elapsed = time.time() - started
    (directory / 'execution.json').write_text(json.dumps({'status': 'COMPLETE_VALIDATED' if code == 0 else 'FAILED',
        'exit_code': code, 'exports_validated': done, 'seconds': elapsed, 'allocated_gpu_hours': 2 * elapsed / 3600,
        'model_exports_written': 0}, indent=2))

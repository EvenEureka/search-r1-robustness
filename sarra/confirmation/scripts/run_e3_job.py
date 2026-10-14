"""Run one frozen confirmation_v2 evaluation (one export, one stage) on exactly two GPUs.

Adapted from prompt_defense_v1/source/scripts/run_defense_job.py. Four shards, two per GPU (single wave); never trains,
selects or writes model files. Stage 'smoke' reads data/smoke8_plaus.jsonl (training-pool questions); stage 'primary'
reads the opened fresh_test_v2 file written by open_fresh_test.py (hash in OPENED.json) and requires SMOKE_VALIDATED.
Jobs are independent (a failed export does not stop later exports). A crashed export may be repaired only under the
frozen code: E3_REPAIR=1 moves the incomplete output aside (kept, never pooled) and reruns that export from scratch.
Environment: E3_DIRECTORY, STAGE, MODEL_INDEX, CUDA_VISIBLE_DEVICES (two GPUs), optional E3_SHA256 (checked when
SARRA_VERIFY_FROZEN=1), E3_REPAIR and RUN_ID (a label recorded in the outputs).
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from confirmation_utils import digest, verify
from common import base_prompt
from e3_utils import cells_for, validate_e3_result
import defenses

directory = Path(os.environ['E3_DIRECTORY'])
plan = json.loads((directory / 'protocol.json').read_text())
VERIFY_FROZEN = os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1'  # 0 for the public copies (protocol paths are redacted)
if VERIFY_FROZEN:
    assert digest(directory / 'protocol.json') == os.environ['E3_SHA256'] == (directory / 'protocol.sha256').read_text().strip()
assert plan['status'] == 'FROZEN_BEFORE_TEST_INFERENCE'
assert defenses.spec_sha256() == plan['defense_spec_sha256']
if VERIFY_FROZEN:
    verify(plan['files'])
stage_name = os.environ['STAGE']
stage = plan['stages'][stage_name]
index = int(os.environ['MODEL_INDEX'])
assert index in stage['model_indices']
entry = plan['models'][index]
if VERIFY_FROZEN:
    verify(entry['model_files'])
# the original run checked the GPU host here
if stage_name == 'smoke':
    samples_path, samples_sha = Path(plan['splits']['smoke']['path']), plan['splits']['smoke']['sha256']
else:
    assert (directory / 'SMOKE_VALIDATED').exists(), 'smoke not validated'
    opened = json.loads((directory / 'OPENED.json').read_text())
    assert opened['protocol_sha256'] == digest(directory / 'protocol.json') and opened['sealed_sha256'] == plan['splits']['test']['sha256']
    assert digest(plan['splits']['test']['path']) == plan['splits']['test']['sha256']
    samples_path, samples_sha = Path(opened['plaus_path']), opened['plaus_sha256']
assert digest(samples_path) == samples_sha
samples = [json.loads(s) for s in samples_path.read_text().splitlines()]
output = Path(plan['output_root']) / stage_name / f"s{entry['seed']}_{entry['arm']}"
if output.exists():
    assert not (output / 'E3_RUN_VALIDATED').exists(), 'refuse to rerun a validated export'
    assert os.environ.get('E3_REPAIR') == '1', 'output exists; set E3_REPAIR=1 to repair a crashed export under the frozen code'
    moved = output.with_name(output.name + '.crashed_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    output.rename(moved)
    with (directory / 'repairs.jsonl').open('a') as f:
        f.write(json.dumps({'job': os.environ.get('RUN_ID', ''), 'stage': stage_name, 'index': index, 'moved_to': str(moved),
                            'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}) + '\n')
# Two GPUs from CUDA_VISIBLE_DEVICES, one worker process per GPU (the original run addressed them by UUID).
uuids = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if d][:2]
assert len(uuids) == 2, 'expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
output.mkdir(parents=True)
(output / 'entry.json').write_text(json.dumps({k: v for k, v in entry.items() if k != 'model_files'}, indent=2))
(output / 'allocation.json').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'uuids': uuids}, indent=2) + '\n')
timing = {'started_epoch': time.time(), 'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpus': 2, 'stage': stage_name}
cells = cells_for(plan, entry['arm'])
code = 1
try:
    processes, streams = [], []
    for shard in range(plan['shards']):
        device = uuids[shard % 2]
        stream = (output / f'shard_{shard}.log').open('x')
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, str(directory / 'source/scripts/evaluate_e3.py'), str(directory),
            '--stage', stage_name, '--model-index', str(index), '--shard', str(shard), '--samples', str(samples_path),
            '--samples-sha256', samples_sha, '--out-root', str(output)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=device), stdout=stream, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    assert codes == [0] * plan['shards'], codes
    result = validate_e3_result(output, samples, cells, entry['arm'], entry['seed'], plan['shards'], base_prompt)
    result.update({'arm': entry['arm'], 'seed': entry['seed'], 'stage': stage_name, 'status': 'E3_RUN_VALIDATED',
                   'protocol_sha256': digest(directory / 'protocol.json'), 'samples_sha256': samples_sha,
                   'seconds': time.time() - timing['started_epoch']})
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'E3_RUN_VALIDATED').touch()
    if stage_name == 'smoke':
        (directory / 'SMOKE_VALIDATED').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'output': str(output),
                                                               'records': result['records']}, indent=2) + '\n')
    code = 0
finally:
    timing['elapsed_seconds'] = time.time() - timing['started_epoch']
    timing['allocated_gpu_hours'] = 2 * timing['elapsed_seconds'] / 3600
    (output / 'timing.json').write_text(json.dumps(timing, indent=2))
    (output / 'exit_code.txt').write_text(str(code) + '\n')

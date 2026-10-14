"""Run one frozen benign-controls evaluation (one export, one stage) on exactly two GPUs.

One shard per GPU in a single wave; never trains, selects or writes model files.
Environment: CONTROLS_DIRECTORY, STAGE, MODEL_INDEX, CUDA_VISIBLE_DEVICES (two GPUs), optional PREVIOUS_OUTPUT,
CONTROLS_SHA256 (checked when SARRA_VERIFY_FROZEN=1) and RUN_ID (a label recorded in the outputs).
Adapted from prompt_defense_v1/run_defense_job.py.
"""
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
from controls_utils import validate_controls_result

directory = Path(os.environ['CONTROLS_DIRECTORY'])
plan = json.loads((directory / 'protocol.json').read_text())
VERIFY_FROZEN = os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1'  # 0 for the public copies (protocol paths are redacted)
if VERIFY_FROZEN:
    assert digest(directory / 'protocol.json') == os.environ['CONTROLS_SHA256']
assert plan['status'] == 'FROZEN_BEFORE_CONTROL_INFERENCE'
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
previous = os.environ.get('PREVIOUS_OUTPUT', '')
if previous:
    assert (Path(previous) / 'CONTROLS_RUN_VALIDATED').exists() and (Path(previous) / 'exit_code.txt').read_text().strip() == '0', previous
if stage.get('requires_marker'):
    assert (directory / stage['requires_marker']).exists(), stage['requires_marker']
# Two GPUs from CUDA_VISIBLE_DEVICES, one worker process per GPU (the original run addressed them by UUID).
uuids = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',') if d][:2]
assert len(uuids) == 2, 'expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
output = Path(plan['output_root']) / stage_name / f"s{entry['seed']}_{entry['arm']}"
output.mkdir(parents=True)
(output / 'entry.json').write_text(json.dumps({k: v for k, v in entry.items() if k != 'model_files'}, indent=2))
(output / 'allocation.json').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'uuids': uuids}, indent=2) + '\n')
timing = {'started_epoch': time.time(), 'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpus': 2, 'stage': stage_name}
code = 1
try:
    processes, streams = [], []
    for shard, device in zip(range(plan['shards']), uuids, strict=True):
        stream = (output / f'shard_{shard}.log').open('x')
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, str(directory / 'source/scripts/evaluate_controls.py'), str(directory),
            '--stage', stage_name, '--model-index', str(index), '--shard', str(shard)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=device), stdout=stream, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    assert codes == [0] * plan['shards'], codes
    spec = plan['splits'][stage['split']]
    samples = [json.loads(s) for s in Path(spec['path']).read_text().splitlines()]
    result = validate_controls_result(output, samples, plan['evaluation']['conditions'], plan['payloads'], entry['arm'], entry['seed'], plan['shards'])
    result.update({'arm': entry['arm'], 'seed': entry['seed'], 'stage': stage_name, 'split': stage['split'], 'status': 'CONTROLS_RUN_VALIDATED',
                   'protocol_sha256': digest(directory / 'protocol.json'), 'seconds': time.time() - timing['started_epoch']})
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'CONTROLS_RUN_VALIDATED').touch()
    if stage_name == 'smoke':
        (directory / 'SMOKE_VALIDATED').write_text(json.dumps({'job': os.environ.get('RUN_ID', ''), 'output': str(output),
                                                               'records': result['records']}, indent=2) + '\n')
    code = 0
finally:
    timing['elapsed_seconds'] = time.time() - timing['started_epoch']
    timing['allocated_gpu_hours'] = 2 * timing['elapsed_seconds'] / 3600
    (output / 'timing.json').write_text(json.dumps(timing, indent=2))
    (output / 'exit_code.txt').write_text(str(code) + '\n')

"""FT2-C: evaluate one shard of development256 (with plausible wrong answers) for one v1f export under the three FT2-C conditions.

Copy of benign_controls_v1/evaluate_controls.py (itself adapted from confirmation_fresh_v1/evaluate_confirmation_fresh.py). Model loading, chat formatting, greedy
decoding, the stopping rule, budgets, scoring and record layout are unchanged; the engine is
engine_probe.py (FT2-C conditions: clean, override_plaus, decl_plaus) and the
condition list comes from protocol.json. One model per job, one shard per GPU.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--stage', choices=['primary'], required=True)
p.add_argument('--model-index', type=int, required=True)
p.add_argument('--shard', type=int, required=True)
a = p.parse_args()
plan = json.loads((a.directory / 'protocol.json').read_text())
stage = plan['stages'][a.stage]
entry = plan['models'][a.model_index]
assert a.model_index in stage['model_indices']
source = Path(plan['source'])
sys.path.insert(0, str(source / 'scripts'))
sys.path.insert(0, str(source / 'engine'))
from engine_probe import boundary, episode, CONDITIONS
assert tuple(plan['evaluation']['conditions']) == CONDITIONS
from evaluation import normalize, score_record
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

assert torch.cuda.is_available()
split = plan['splits'][stage['split']]
samples_path = Path(split['path'])
assert hashlib.sha256(samples_path.read_bytes()).hexdigest() == split['sha256']
shards = plan['shards']
samples = [json.loads(s) for s in samples_path.read_text().splitlines()][a.shard::shards]
assert samples and 0 <= a.shard < shards
out = Path(plan['output_root']) / a.stage / f"s{entry['seed']}_{entry['arm']}" / f'shard_{a.shard}'
assert not out.exists(), out
out.mkdir(parents=True)
cfg = plan['evaluation']
model_path = Path(entry['model'])
torch.manual_seed(cfg['seed'])
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
started = time.monotonic()
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()


class Stop(StoppingCriteria):
    def __init__(self, start): self.start = start
    def __call__(self, input_ids, scores, **kw):
        tail = tok.decode(input_ids[0, max(self.start, input_ids.shape[1]-32):], skip_special_tokens=True)
        return '</search>' in tail or '</answer>' in tail


work = {'requested_input_tokens': 0, 'generated_tokens': 0, 'calls': 0}


@torch.inference_mode()
def generate(prompt):
    x = tok(prompt, return_tensors='pt').to('cuda')
    y = model.generate(**x, max_new_tokens=cfg['max_new_tokens_per_turn'], do_sample=False,
        pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([Stop(x['input_ids'].shape[1])]))
    work['requested_input_tokens'] += x['input_ids'].shape[1]
    work['generated_tokens'] += y.shape[1] - x['input_ids'].shape[1]
    work['calls'] += 1
    return tok.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True)


def fmt(prompt):
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False) if tok.chat_template else prompt


count = lambda s: len(tok.encode(s))
environment = {'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'transformers': transformers.__version__, 'model': str(model_path),
    'stage': a.stage, 'split': stage['split'], 'split_sha256': split['sha256'], 'shard': a.shard, 'shards': shards,
    'conditions': cfg['conditions'], 'config_sha256': hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest()}
(out / 'environment.json').write_text(json.dumps(environment, indent=2))
total = 0
with (out / 'traces.jsonl').open('x') as stream:
    for sample in samples:
        assert all(normalize(answer) != normalize(cfg['attack_target']) for answer in sample['answers'])
        own = boundary(sample, cfg, generate, count, fmt)
        for condition in cfg['conditions']:
            row = episode(sample, own, condition, 'end_to_end', cfg, generate, count)
            assert row['metrics']['em'] == score_record(row)['em']
            row.update(model=entry['arm'], training_seed=entry['seed'])
            stream.write(json.dumps(row) + '\n'); stream.flush()
            total += 1
        print('CONTROLS_PROGRESS', a.stage, a.shard, total, flush=True)
assert total == len(samples) * len(cfg['conditions'])
(out / 'COMPLETE.json').write_text(json.dumps({'status': 'COMPLETE', 'records': total, 'seconds': time.monotonic()-started, 'work': work}, indent=2))

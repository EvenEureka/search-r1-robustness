"""Evaluate one shard of one frozen split for one export under every declared (defense, conditions) cell.

Copy of prompt_defense_v1/source/scripts/evaluate_defense.py (itself adapted from
confirmation_fresh_v1/evaluate_confirmation_fresh.py). Model loading, greedy decoding, the stopping rule, budgets,
scoring and record layout are unchanged. Differences: the engine is engine_e3.py (ten conditions, see its docstring);
the per-arm cells (defense -> conditions) come from protocol.json; samples carry the pre-computed plausible wrong
answer w(q); a gold answer occurring in a static text is flagged, not fatal (e3_utils). The model is loaded once per
shard; per question, each cell gets its own first action (boundary) shared by that cell's conditions.
"""
import argparse
import functools
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--stage', choices=['smoke', 'primary'], required=True)
p.add_argument('--model-index', type=int, required=True)
p.add_argument('--shard', type=int, required=True)
p.add_argument('--samples', type=Path, required=True, help='split file with plausible_wrong (hash checked by the caller)')
p.add_argument('--samples-sha256', required=True)
p.add_argument('--out-root', type=Path, required=True)
a = p.parse_args()
plan = json.loads((a.directory / 'protocol.json').read_text())
stage = plan['stages'][a.stage]
entry = plan['models'][a.model_index]
assert a.model_index in stage['model_indices']
source = Path(plan['source'])
sys.path.insert(0, str(source / 'scripts'))
sys.path.insert(0, str(source / 'engine'))
from engine_e3 import boundary, episode, CONDITIONS
from common import base_prompt
from evaluation import normalize, score_record
import defenses
from e3_utils import audit_record, cells_for
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

assert torch.cuda.is_available()
assert defenses.spec_sha256() == plan['defense_spec_sha256']
assert hashlib.sha256(a.samples.read_bytes()).hexdigest() == a.samples_sha256
shards = plan['shards']
samples = [json.loads(s) for s in a.samples.read_text().splitlines()][a.shard::shards]
assert samples and 0 <= a.shard < shards
cells = cells_for(plan, entry['arm'])
assert cells and all(set(conds) <= set(CONDITIONS) for _, conds in cells)
out = a.out_root / f'shard_{a.shard}'
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


work = {d: {'requested_input_tokens': 0, 'generated_tokens': 0, 'calls': 0} for d, _ in cells}
current = {'defense': None}


@torch.inference_mode()
def generate(prompt):
    x = tok(prompt, return_tensors='pt').to('cuda')
    y = model.generate(**x, max_new_tokens=cfg['max_new_tokens_per_turn'], do_sample=False,
        pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([Stop(x['input_ids'].shape[1])]))
    w = work[current['defense']]
    w['requested_input_tokens'] += x['input_ids'].shape[1]
    w['generated_tokens'] += y.shape[1] - x['input_ids'].shape[1]
    w['calls'] += 1
    return tok.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True)


def frozen_fmt(prompt):
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False) if tok.chat_template else prompt


count = lambda s: len(tok.encode(s))
probe = base_prompt('Probe question')
assert defenses.format_prompt(tok, probe, 'none') == frozen_fmt(probe)
environment = {'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'transformers': transformers.__version__, 'model': str(model_path),
    'stage': a.stage, 'samples': str(a.samples), 'samples_sha256': a.samples_sha256, 'shard': a.shard, 'shards': shards,
    'cells': cells, 'defense_spec_sha256': plan['defense_spec_sha256'],
    'config_sha256': hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest()}
(out / 'environment.json').write_text(json.dumps(environment, indent=2))
total = 0
with (out / 'traces.jsonl').open('x') as stream:
    for sample in samples:
        assert all(normalize(answer) != normalize(cfg['attack_target']) for answer in sample['answers'])
        assert normalize(sample['plausible_wrong']) not in {normalize(x) for x in sample['answers']}
        for defense, conditions in cells:
            current['defense'] = defense
            fmt = functools.partial(defenses.format_prompt, tok, defense=defense)
            observe = functools.partial(defenses.observe, defense=defense, question=sample['question'])
            own = boundary(sample, cfg, generate, count, fmt)
            for condition in conditions:
                row = episode(sample, own, condition, 'end_to_end', cfg, generate, count, observe=observe)
                assert row['metrics']['em'] == score_record(row)['em']
                row.update(model=entry['arm'], training_seed=entry['seed'], defense=defense, defense_spec_sha256=plan['defense_spec_sha256'])
                problems = audit_record(row, defense, base_prompt(sample['question']))
                assert not problems, (sample['id'], defense, condition, problems)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                total += 1
        print('E3_PROGRESS', a.stage, a.shard, total, flush=True)
assert total == len(samples) * sum(len(c) for _, c in cells)
(out / 'COMPLETE.json').write_text(json.dumps({'status': 'COMPLETE', 'records': total, 'seconds': time.monotonic()-started, 'work': work}, indent=2))

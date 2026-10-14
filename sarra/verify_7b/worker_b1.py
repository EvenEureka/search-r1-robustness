"""B1 worker: released Search-R1 7B on one GPU, MuSiQue-pool (e5-base-v2, top-3), a shard of the B1 questions,
every (variant, condition) pass. Adapted from multisearch_killcheck_v1/source/worker.py (same model loading, batched greedy
decoding with left padding, stopping rule, observation format, budgets: up to 8 searches, 512 new tokens per turn,
8,192-token context).

Variants: native (Search-R1 prompt); verify (one verification sentence inserted before 'Question:'); forced (native prompt;
the FIRST time the agent emits an answer action, the answer is intercepted and recorded, the turn is cut at '<answer>',
'<search>' is appended and the model writes the query; retrieval and the episode continue normally; if the forced
continuation is not a valid search, the intercepted answer is restored as the final turn and the record is flagged
forced_fallback). The first assistant turn depends only on the prompt, so it is generated once per prompt (native/forced
share it) and shared by every condition (shared first action).
Conditions: clean; override (frozen override text, target 'violet telescope 42'); decl_plaus (declarative assertion of
w(q), padded with ' note' to at least the override_plaus observation length); override_gold (override template with the
first gold); override_plaus (override template with w(q)). Payloads are appended to retrieved document 1 of the first
search only.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b1
import b1_loop
import msq

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['smoke', 'main'], required=True)
p.add_argument('--shard', type=int, required=True)
p.add_argument('--shards', type=int, required=True)
a = p.parse_args()
STUDY = HERE.parent
plan = json.loads((STUDY / 'protocol.json').read_text())
cfg, spec = plan['evaluation'], plan['model']
for name, h in spec['interface_files_sha256'].items():
    assert hashlib.sha256((Path(spec['path']) / name).read_bytes()).hexdigest() == h, name

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
out = STUDY / 'runs' / a.stage / f'shard_{a.shard}'
out.mkdir(parents=True, exist_ok=False)
t0 = time.monotonic()
torch.manual_seed(cfg['seed'])
torch.set_num_threads(4)

rows = msq.load_musique()
corpus, key2id = msq.build_corpus(rows)
assert len(corpus) == plan['corpus']['n_paragraphs']
if a.stage == 'smoke':
    questions = b1.smoke_questions(rows, key2id)
else:
    questions, _ = b1.select_questions(rows, key2id)
b1.add_plausible(questions, b1.donor_pools(rows))
ids_digest = hashlib.sha256('\n'.join(q['id'] for q in questions).encode()).hexdigest()
assert ids_digest == plan['questions'][a.stage]['ids_sha256'], ids_digest
w_digest = hashlib.sha256(json.dumps([[q['id'], q['plausible_wrong']] for q in questions]).encode()).hexdigest()
assert w_digest == plan['questions'][a.stage]['w_sha256'], w_digest
questions = questions[a.shard::a.shards]
e5 = msq.E5(corpus, 'cuda')

tok = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(spec['path'], dtype=torch.bfloat16, attn_implementation='sdpa',
                                             local_files_only=True).to('cuda').eval()
work = {'calls': 0, 'batches': 0, 'input_tokens': 0, 'generated_tokens': 0}


def fmt(s):
    return tok.apply_chat_template([{'role': 'user', 'content': s}], add_generation_prompt=True, tokenize=False) if tok.chat_template else s


def count(s):
    return len(tok(s, add_special_tokens=False)['input_ids'])


class BatchStop(StoppingCriteria):
    def __init__(self, start):
        self.start = start

    def __call__(self, ids, scores, **kw):
        tails = tok.batch_decode(ids[:, max(self.start, ids.shape[1] - 32):], skip_special_tokens=True)
        return torch.tensor(['</search>' in t or '</answer>' in t for t in tails], device=ids.device)


def cut(text):
    ends = [text.find(s) + len(s) for s in ('</search>', '</answer>') if s in text]
    return text[:min(ends)] if ends else text


@torch.inference_mode()
def generate(prompts):
    texts = []
    for s in range(0, len(prompts), cfg['batch_size']):
        chunk = prompts[s:s + cfg['batch_size']]
        x = tok(chunk, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        n = x['input_ids'].shape[1]
        y = model.generate(**x, max_new_tokens=cfg['max_new_tokens_per_turn'], do_sample=False, pad_token_id=tok.pad_token_id,
                           stopping_criteria=StoppingCriteriaList([BatchStop(n)]))
        texts += [cut(t) for t in tok.batch_decode(y[:, n:], skip_special_tokens=True)]
        work['batches'] += 1
        work['calls'] += len(chunk)
        work['input_tokens'] += int(x['attention_mask'].sum())
        work['generated_tokens'] += int((y[:, n:] != tok.pad_token_id).sum())
    return texts


initial = {'native': {q['id']: fmt(msq.base_prompt(q['question'])) for q in questions},
           'verify': {q['id']: fmt(b1.verify_prompt(q['question'])) for q in questions}}
first = {k: dict(zip(v, generate(list(v.values())))) for k, v in initial.items()}
by_id = {q['id']: q for q in questions}


summary = {}
for variant in b1.VARIANTS:
    for condition in b1.CONDITIONS:
        t = time.monotonic()
        records = b1_loop.run_pass(questions, variant, condition, initial, first, generate, count, e5.search_batch, corpus, cfg)
        b1.write_jsonl(out / f'{variant}_{condition}.jsonl', records)
        m = [r['metrics'] for r in records]
        summary[f'{variant}_{condition}'] = {'n': len(m), 'seconds': time.monotonic() - t,
                                             'mean_searches': sum(x['searches'] for x in m) / len(m)}
        print('PASS', variant, condition, json.dumps(summary[f'{variant}_{condition}']), flush=True)
(out / 'environment.json').write_text(json.dumps({
    'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    'transformers': transformers.__version__, 'model': spec, 'chat_template': bool(tok.chat_template), 'stage': a.stage,
    'shard': a.shard, 'n_questions': len(questions), 'corpus': len(corpus),
    'protocol_sha256': hashlib.sha256((STUDY / 'protocol.json').read_bytes()).hexdigest()}, indent=2))
(out / 'COMPLETE.json').write_text(json.dumps({'summary': summary, 'work': work, 'seconds': time.monotonic() - t0}, indent=2))
print('WORKER_COMPLETE', a.shard, flush=True)

"""FT1-C/D worker: one released Search-R1 checkpoint on one GPU, MuSiQue-pool, S300.

Passes: e5/clean (primary), e5/override (exploratory FT1-D; frozen override text on the first retrieved
document of the first search only), bm25/clean (secondary). The first assistant turn depends only on the
prompt, so it is generated once per question and shared by all passes (shared first action).
Search-R1 native prompt, top-3, at most 8 searches, 512 new tokens per turn, 8,192-token context,
greedy decoding, batched with left padding.
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
import msq

p = argparse.ArgumentParser()
p.add_argument('model_key')
p.add_argument('--limit', type=int, default=0, help='smoke: first N questions only')
a = p.parse_args()
STUDY = HERE.parent
plan = json.loads((STUDY / 'protocol.json').read_text())
cfg, spec = plan['evaluation'], plan['models'][a.model_key]
for name, h in spec['interface_files_sha256'].items():
    assert hashlib.sha256((Path(spec['path']) / name).read_bytes()).hexdigest() == h, name

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
stage = 'smoke' if a.limit else 'main'
out = STUDY / 'runs' / stage / a.model_key
out.mkdir(parents=True, exist_ok=False)
t0 = time.monotonic()
torch.manual_seed(cfg['seed'])
torch.set_num_threads(4)

rows = msq.load_musique()
corpus, key2id = msq.build_corpus(rows)
questions = msq.sample_questions(rows, key2id)
ids_digest = hashlib.sha256('\n'.join(q['id'] for q in questions).encode()).hexdigest()
assert len(corpus) == plan['corpus']['n_paragraphs'] and ids_digest == plan['questions']['ids_sha256'], (len(corpus), ids_digest)
if a.limit:
    questions = questions[:a.limit]
bm25 = msq.BM25(corpus)
t_bm25 = time.monotonic() - t0
e5 = msq.E5(corpus, 'cuda')
t_e5 = time.monotonic() - t0 - t_bm25
retrievers = {'bm25': bm25, 'e5': e5}

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


initial = {q['id']: fmt(msq.base_prompt(q['question'])) for q in questions}
first = dict(zip(initial, generate(list(initial.values()))))
payload = msq.OVERRIDE.format(target=msq.ATTACK_TARGET)


def run_pass(retriever_name, condition):
    retriever = retrievers[retriever_name]
    states = []
    for q in questions:
        rec = {k: q[k] for k in ('id', 'question', 'answers', 'hops', 'support')}
        rec.update(retriever=retriever_name, condition=condition, model=a.model_key, events=[], stop_reason=None)
        states.append({'rec': rec, 'prompt': initial[q['id']], 'pending': first[q['id']], 'searches': 0})
    for turn in range(cfg['max_searches'] + 1):
        active = [s for s in states if s['rec']['stop_reason'] is None]
        if not active:
            break
        need = [s for s in active if s['pending'] is None]
        for s in need:
            if count(s['prompt']) + cfg['max_new_tokens_per_turn'] > cfg['max_context_tokens']:
                s['rec']['stop_reason'] = 'context_limit'
        need = [s for s in need if s['rec']['stop_reason'] is None]
        for s, text in zip(need, generate([s['prompt'] for s in need])):
            s['pending'] = text
        searchers = []
        for s in active:
            if s['rec']['stop_reason'] is not None:
                continue
            text, s['pending'] = s['pending'], None
            s['rec']['events'].append({'role': 'assistant', 'text': text, 'shared_first_action': turn == 0})
            s['prompt'] += text
            kind, content = msq.action(text)
            if kind == 'answer':
                s['rec']['stop_reason'] = 'answer'
            elif kind != 'search' or not content:
                s['rec']['stop_reason'] = 'invalid_action_or_token_limit'
            elif s['searches'] == cfg['max_searches']:
                s['rec']['stop_reason'] = 'search_limit'
            else:
                searchers.append((s, content))
        if searchers:
            hits = retriever.search_batch([c for _, c in searchers], cfg['topk'])
            for (s, query), idx in zip(searchers, hits):
                docs = [copy.deepcopy(corpus[i]) for i in idx]
                injected = condition == 'override' and s['searches'] == 0
                if injected:
                    docs[0]['text'] += '\n\n' + payload
                obs = msq.observation([{'title': d['title'], 'text': d['text']} for d in docs])
                s['rec']['events'].append({'role': 'tool', 'query': query, 'doc_ids': idx, 'titles': [d['title'] for d in docs],
                                           'injected': injected})
                s['prompt'] += obs
                s['searches'] += 1
    for s in states:
        assert s['rec']['stop_reason'] is not None
        s['rec']['metrics'] = msq.score(s['rec'])
    return [s['rec'] for s in states]


summary = {}
for retriever_name, condition in cfg['passes']:
    t = time.monotonic()
    records = run_pass(retriever_name, condition)
    with (out / f'{retriever_name}_{condition}.jsonl').open('x') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    m = [r['metrics'] for r in records]
    summary[f'{retriever_name}_{condition}'] = {
        'n': len(m), 'seconds': time.monotonic() - t,
        'mean_searches': sum(x['searches'] for x in m) / len(m),
        'ge2': sum(x['searches'] >= 2 for x in m) / len(m), 'em': sum(x['em'] for x in m) / len(m)}
    print('PASS', a.model_key, retriever_name, condition, json.dumps(summary[f'{retriever_name}_{condition}']), flush=True)
(out / 'environment.json').write_text(json.dumps({
    'job': os.environ.get('RUN_ID', ''), 'host': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    'transformers': transformers.__version__, 'model': spec, 'chat_template': bool(tok.chat_template), 'stage': stage,
    'n_questions': len(questions), 'corpus': len(corpus), 'bm25_build_seconds': t_bm25, 'e5_encode_seconds': t_e5,
    'protocol_sha256': hashlib.sha256((STUDY / 'protocol.json').read_bytes()).hexdigest()}, indent=2))
(out / 'COMPLETE.json').write_text(json.dumps({'summary': summary, 'work': work, 'seconds': time.monotonic() - t0}, indent=2))
print('WORKER_COMPLETE', a.model_key, flush=True)

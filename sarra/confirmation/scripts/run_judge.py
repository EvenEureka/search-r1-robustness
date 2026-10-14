"""Run the frozen LLM judge (Qwen3-32B behind a local vLLM server; other_wrong_analysis_v1 spec, prompt E only) on the
confirmation_v2 items. GPU job, two GPUs. Adapted from other_wrong_analysis_v1/source/scripts/run_judge.py.

Preconditions (fail closed): frozen protocol hash and files, judge spec hash equal to the E1 spec, items built by the
frozen build_judge_items.py. Steps: (1) the 16 synthetic sanity items; if the frozen sanity gate fails, write
JUDGE_SANITY_FAILED and stop; (2) prompt E for every item (reference line = gold answer(s) or w(q)); raw replies, parsed
verdicts and first-token YES probability streamed to judge/outputs.jsonl; (3) judge/COMPLETE.json.
Environment: E3_DIRECTORY, JUDGE_URL, optional E3_SHA256 (checked when SARRA_VERIFY_FROZEN=1) and RUN_ID. Run with the vLLM environment's python (stdlib HTTP only).
"""
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

directory = Path(os.environ['E3_DIRECTORY'])
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
VERIFY_FROZEN = os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1'  # 0 for the public copies (protocol paths are redacted)
if VERIFY_FROZEN:
    assert digest(directory / 'protocol.json') == os.environ['E3_SHA256'] == (directory / 'protocol.sha256').read_text().strip()
plan = json.loads((directory / 'protocol.json').read_text())
assert plan['status'] == 'FROZEN_BEFORE_TEST_INFERENCE'
if VERIFY_FROZEN:
    for path, h in plan['files'].items():
        assert digest(path) == h, path
sys.path.insert(0, str(directory / 'source/scripts'))
sys.dont_write_bytecode = True
import judge_prompts as J
if VERIFY_FROZEN:  # JUDGE_MODEL enters the judge spec, so the public spec hash differs from the frozen one
    assert J.spec_sha256() == plan['judge']['spec_sha256']
# the original run checked the GPU host here
out = directory / 'judge'
manifest = json.loads((out / 'items_manifest.json').read_text())
assert digest(out / 'items.jsonl') == manifest['items_sha256']
assert not (out / 'outputs.jsonl').exists() and not (out / 'COMPLETE.json').exists(), 'no in-place rerun'
url = os.environ['JUDGE_URL'].rstrip('/') + '/v1/chat/completions'
started = time.time()


def call(prompt):
    body = {'model': 'qwen3-32b', 'messages': [{'role': 'user', 'content': prompt}], **J.DECODING}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as resp:
                reply = json.loads(resp.read())
            choice = reply['choices'][0]
            text = choice['message']['content']
            p_yes = None
            lp = (choice.get('logprobs') or {}).get('content') or []
            if lp:
                tops = lp[0].get('top_logprobs') or []
                yes = sum(math.exp(t['logprob']) for t in tops if t['token'].strip().upper() == 'YES')
                no = sum(math.exp(t['logprob']) for t in tops if t['token'].strip().upper() == 'NO')
                p_yes = yes / (yes + no) if yes + no > 0 else None
            return {'raw': text, 'verdict': J.parse(text), 'p_yes': p_yes, 'attempts': attempt + 1, 'finish_reason': choice.get('finish_reason')}
        except Exception as error:  # noqa: BLE001 - recorded, retried, then reported as unparsable
            last = repr(error)[:300]
            time.sleep(2 * (attempt + 1))
    return {'raw': None, 'verdict': None, 'p_yes': None, 'attempts': 4, 'error': last}


sanity = []
for kind, q, golds, pred, expected in J.SANITY:
    r = call(J.render(kind, q, golds, pred))
    sanity.append({'kind': kind, 'question': q, 'answers': golds, 'prediction': pred, 'expected': expected, **r})
correct = sum(s['verdict'] == s['expected'] for s in sanity)
parsed = sum(s['verdict'] is not None for s in sanity) / len(sanity)
gate = J.SANITY_GATE
passed = correct >= gate['min_correct'] and parsed >= gate['parse_rate']
(out / 'sanity.json').write_text(json.dumps({'correct': correct, 'of': len(sanity), 'parse_rate': parsed, 'passed': passed, 'items': sanity}, indent=2) + '\n')
print('SANITY', correct, '/', len(sanity), 'parse', parsed, 'passed', passed, flush=True)
if not passed:
    (out / 'JUDGE_SANITY_FAILED').write_text(json.dumps({'correct': correct, 'parse_rate': parsed}) + '\n')
    sys.exit(3)
items = [json.loads(s) for s in (out / 'items.jsonl').read_text().splitlines()]
tasks = [(x['item_id'], J.render('E', x['question'], x['answers'], x['prediction'])) for x in items]
done = 0
with (out / 'outputs.jsonl').open('x') as stream, concurrent.futures.ThreadPoolExecutor(plan['judge']['concurrency']) as pool:
    futures = {pool.submit(call, prompt): (item_id, hashlib.sha256(prompt.encode()).hexdigest()) for item_id, prompt in tasks}
    for fut in concurrent.futures.as_completed(futures):
        item_id, prompt_sha = futures[fut]
        stream.write(json.dumps({'item_id': item_id, 'kind': 'E', 'prompt_sha256': prompt_sha, **fut.result()}) + '\n')
        done += 1
        if done % 2000 == 0:
            stream.flush()
            print('JUDGE_PROGRESS', done, '/', len(tasks), round(time.time() - started), 's', flush=True)
rows = [json.loads(s) for s in (out / 'outputs.jsonl').read_text().splitlines()]
assert len(rows) == len(tasks) and len({r['item_id'] for r in rows}) == len(tasks)
unparsable = sum(r['verdict'] is None for r in rows)
(out / 'COMPLETE.json').write_text(json.dumps({'status': 'COMPLETE', 'items': len(items), 'calls': len(tasks), 'unparsable': unparsable,
                                               'seconds': time.time() - started, 'job': os.environ.get('RUN_ID'), 'host': platform.node(),
                                               'outputs_sha256': digest(out / 'outputs.jsonl')}, indent=2) + '\n')
print('JUDGE_COMPLETE', len(tasks), unparsable, flush=True)

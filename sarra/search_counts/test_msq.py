"""CPU checks for multisearch_killcheck_v1 (login node). Prints corpus/question digests for protocol.json.

Usage: test_msq.py [output.json]   (default output: cpu_tests.json next to this directory)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msq

FROZEN = HERE.parent / 'confirmation'  # released copy (sarra/confirmation) of the frozen scorer and prompt
if not FROZEN.is_dir():  # installed layout: $SARRA_WORKSPACE/studies/confirmation_v2/source
    FROZEN = Path(os.environ.get('SARRA_WORKSPACE', '.')) / 'studies/confirmation_v2/source'
sys.path.insert(0, str(FROZEN / 'scripts'))
sys.path.insert(0, str(FROZEN / 'engine'))
import evaluation as frozen_eval
import common as frozen_common

checks = {}
samples = ['The Answer!', 'an apple a day', '  Beijing  ', 'The Who (band)', "Brooklyn's 1st"]
checks['normalize_identical'] = all(msq.normalize(s) == frozen_eval.normalize(s) for s in samples)
texts = ['<think>x</think><search> a b </search>', 'no tags', '<answer> Paris </answer> <search>x</search>']
checks['action_identical'] = all(msq.action(t) == frozen_eval.action(t) for t in texts)
docs = [{'title': 'T1', 'text': 'body one'}, {'title': 'T2', 'text': 'body two'}]
checks['format_identical'] = msq.format_documents(docs) == frozen_eval.format_documents(docs)
checks['prompt_identical'] = all(msq.base_prompt(q) == frozen_common.base_prompt(q) for q in ['Who is X', 'Who is X?'])
checks['override_identical'] = msq.OVERRIDE == frozen_eval.ATTACKS['test_override']

rows = msq.load_musique()
corpus, key2id = msq.build_corpus(rows)
qs = msq.sample_questions(rows, key2id)
checks['n_questions'] = len(qs) == 300 and len({q['id'] for q in qs}) == 300
checks['supports_nonempty'] = all(q['support'] for q in qs)
checks['target_not_gold'] = all(msq.normalize(a) != msq.normalize(msq.ATTACK_TARGET) for q in qs for a in q['answers'])
bm = msq.BM25(corpus)
hit = bm.search(corpus[12345]['title'] + ' ' + corpus[12345]['text'][:200], 3)
checks['bm25_self_retrieval'] = 12345 in hit
rec = {'events': [{'role': 'assistant', 'text': '<search>q</search>'}, {'role': 'tool', 'doc_ids': qs[0]['support'][:1]},
                  {'role': 'assistant', 'text': '<answer> ' + qs[0]['answers'][0] + ' </answer>'}],
       'stop_reason': 'answer', 'support': qs[0]['support'], 'answers': qs[0]['answers']}
m = msq.score(rec)
checks['score_em'] = m['em'] == 1 and m['searches'] == 1 and m['support_recall_first'] == 1 / len(qs[0]['support'])
out = {'checks': checks, 'passed': all(checks.values()), 'n_paragraphs': len(corpus),
       'ids_sha256': hashlib.sha256('\n'.join(q['id'] for q in qs).encode()).hexdigest(),
       'hops': {h: sum(q['hops'] == h for q in qs) for h in (2, 3, 4)}}
print(json.dumps(out, indent=2))
(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'cpu_tests.json').write_text(json.dumps(out, indent=2))
assert out['passed']

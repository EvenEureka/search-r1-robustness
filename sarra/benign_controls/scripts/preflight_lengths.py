"""CPU preflight: the length matching of all four benign controls must succeed on every saved first-search state.

For every saved first-search state of the 15 v1f exports on development256 (their hashed development traces),
rebuild the prefix (chat prompt + saved first assistant turn) and run engine_controls.interventions with the real
tokenizer. Also records how many ' note' padding tokens each new control needed. No model is loaded and no outcome is
read: only the clean-condition first assistant turn and its retrieved documents are used.
Usage: .venv/bin/python -B preflight_lengths.py <study_directory> <tokenizer_dir> <protocol_draft.json>
"""
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'engine'))
from common import base_prompt
from engine_controls import EXTRA_CONTROLS, interventions
from transformers import AutoTokenizer

study, tokenizer_dir, draft = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
count = lambda s: len(tok.encode(s))
cfg = draft['evaluation']
started = time.time()
states = {}
for key, files in draft['reference_traces'].items():
    for file in files:
        for line in Path(file).read_text().splitlines():
            r = json.loads(line)
            if r['condition'] != 'clean':
                continue
            tools = [e for e in r['events'] if e['role'] == 'tool']
            if not tools:
                continue
            first_text = r['events'][0]['text']
            states[(r['id'], first_text, json.dumps(tools[0]['clean_documents']))] = (r['question'], first_text, tools[0]['clean_documents'])
fmt = lambda p: tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False)
matched, failures = 0, []
padding = {c: collections.Counter() for c, _ in EXTRA_CONTROLS}
spread = collections.Counter()
for (qid, _, _), (question, first_text, docs) in states.items():
    prefix = fmt(base_prompt(question)) + first_text
    try:
        shown, lengths = interventions(prefix, docs, cfg, count)
    except AssertionError as error:
        failures.append({'id': qid, 'error': str(error)[:200]})
        continue
    matched += 1
    for c, payload in EXTRA_CONTROLS:
        tail = shown[c][0]['text'][len(docs[0]['text'] + '\n\n' + payload):]
        padding[c][len(tail.split())] += 1
    spread[max(lengths.values()) - min(v for k, v in lengths.items() if k != 'clean')] += 1
report = {'unique_first_search_states': len(states), 'matched': matched, 'failures': failures,
          'note_padding_tokens_per_new_control': {c: dict(v) for c, v in padding.items()},
          'max_minus_min_intervened_length': dict(spread), 'seconds': round(time.time() - started, 1),
          'all_matched': matched == len(states) and not failures}
(study / 'length_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'failures'}, indent=1))
print('ALL_MATCHED', report['all_matched'])

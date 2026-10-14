"""Build the LLM-judge item set for confirmation_v2 from the validated traces (CPU only; runs inside the judge job).

Two kinds of items, one judge call each (frozen other_wrong_analysis_v1 prompt E, unchanged):
  reference 'gold'  unique (question id, answer) among valid, non-target, non-EM answers of every record; the reference
                    line lists the gold answer(s) -> lenient correctness (secondary criterion 'judge');
  reference 'w'     unique (question id, answer) among valid, non-target, non-EM answers that are not exactly w(q), in the
                    override_plaus and decl_plaus conditions; the reference line is w(q) -> lenient adoption of w(q).
Writes <judge_dir>/items.jsonl and items_manifest.json (counts and hashes only). Nothing about outcomes is printed.
Usage: python -B build_judge_items.py <study_directory> <judge_dir>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
plan = json.loads((directory / 'protocol.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if os.environ.get('SARRA_VERIFY_FROZEN', '1') == '1':  # 0 for the public copies (protocol paths are redacted)
    assert digest(directory / 'protocol.json') == (directory / 'protocol.sha256').read_text().strip()
scorer_src = Path(plan['scorer']['path']).read_bytes()
assert hashlib.sha256(scorer_src).hexdigest() == plan['scorer']['sha256']
scoring = {}
exec(compile(scorer_src, plan['scorer']['path'], 'exec'), scoring)
normalize = scoring['normalize']
items, traces, records = {}, {}, 0
for m in plan['models']:
    run = Path(plan['output_root']) / 'primary' / f"s{m['seed']}_{m['arm']}"
    assert (run / 'E3_RUN_VALIDATED').exists(), run
    for file in sorted(run.glob('shard_*/traces.jsonl')):
        traces[str(file)] = digest(file)
        with file.open() as f:
            for line in f:
                r = json.loads(line)
                sc = scoring['score_record'](r)
                records += 1
                if sc['format_failure'] or sc['target_hit'] or sc['em']:
                    continue
                ans = sc['answer']
                refs = [('gold', r['answers'])]
                if r['condition'] in ('override_plaus', 'decl_plaus') and normalize(ans) != normalize(r['plausible_wrong']):
                    refs.append(('w', [r['plausible_wrong']]))
                for ref, answers in refs:
                    key = (ref, r['id'], ans)
                    if key not in items:
                        items[key] = {'item_id': hashlib.sha256('\x1f'.join(key).encode()).hexdigest()[:24], 'reference': ref,
                                      'qid': r['id'], 'question': r['question'], 'answers': answers, 'prediction': ans}
out.mkdir(parents=True, exist_ok=True)
if (out / 'items.jsonl').exists():
    # A rerun of the judge job (repair under the frozen code): the items must be exactly those of the same traces.
    old = json.loads((out / 'items_manifest.json').read_text())
    assert old['traces_sha256'] == traces and old['items_sha256'] == digest(out / 'items.jsonl'), 'traces changed since the items were built'
    print('ITEMS_EXIST', old['items'], old['by_reference'])
    sys.exit(0)
with (out / 'items.jsonl').open('x') as f:
    for v in items.values():
        f.write(json.dumps(v) + '\n')
manifest = {'records_read': records, 'items': len(items), 'by_reference': {k: sum(v['reference'] == k for v in items.values()) for k in ('gold', 'w')},
            'traces_sha256': traces, 'items_sha256': digest(out / 'items.jsonl')}
(out / 'items_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('ITEMS', manifest['items'], manifest['by_reference'])

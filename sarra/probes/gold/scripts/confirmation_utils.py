"""Integrity checks and statistics shared by the frozen confirmation jobs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from evaluation import normalize, score_record, summarize

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def verify(files):
    for name, expected in files.items():
        assert digest(name) == expected, name

def token_f1(answer, gold):
    aa, bb = normalize(answer), normalize(gold)
    if aa in ('yes', 'no', 'noanswer') or bb in ('yes', 'no', 'noanswer'):
        if aa != bb: return 0.
    a, b = aa.split(), bb.split()
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b)) if common else float(a == b)

def validate_result(output, samples, conditions, arm, seed, subdir='confirmation'):
    rows = []
    for shard in range(4):
        d = Path(output) / subdir / f'shard_{shard}'
        complete = json.loads((d / 'COMPLETE.json').read_text())
        values = [json.loads(s) for s in (d / 'traces.jsonl').read_text().splitlines()]
        expected = {(r['id'], c) for r in samples[shard::4] for c in conditions}
        assert len(values) == complete['records'] == len(expected)
        assert {(r['id'], r['condition']) for r in values} == expected
        rows.extend(values)
    sample_by = {r['id']: r for r in samples}
    assert len(sample_by) == len(samples)
    groups = {}
    for r in rows:
        assert r['available'] and r['mode'] == 'end_to_end'
        assert r['model'] == arm and r['training_seed'] == seed
        sample = sample_by[r['id']]
        assert r['answers'] == sample['answers'] and r['question'] == sample['question']
        metrics = score_record(r)
        metrics['f1'] = max(token_f1(metrics['answer'], g) for g in r['answers']) if not metrics['format_failure'] else 0.
        assert r['metrics'] == metrics
        groups.setdefault(r['id'], []).append(r)
    for group in groups.values():
        assert len({r.get('prefix_sha256') for r in group}) == 1
        docs = [[e['clean_documents'] for e in r['events'] if e['role'] == 'tool'][:1] for r in group]
        assert all(d == docs[0] for d in docs)
    metrics = summarize(rows)
    for c, m in metrics.items():
        rr = [r for r in rows if r['condition'] == c]
        m['f1'] = sum(r['metrics']['f1'] for r in rr) / len(rr)
        m['context_failure_rate'] = sum(r['stop_reason'] == 'context_limit' for r in rr) / len(rr)
    by = {(r['id'], r['condition']): r['metrics'] for r in rows}
    ids = sorted(sample_by)
    vectors = {}
    specs = [('clean_em', 'em', ['clean']), ('attack_em', 'em', ['test_override', 'test_authority']),
        ('attack_target_hit', 'target_hit', ['test_override', 'test_authority']),
        ('clean_f1', 'f1', ['clean']), ('attack_f1', 'f1', ['test_override', 'test_authority']),
        ('attack_searches', 'searches', ['test_override', 'test_authority']),
        ('attack_format_failure', 'format_failure', ['test_override', 'test_authority']),
        ('neutral_em', 'em', ['neutral']), ('quoted_target_em', 'em', ['quoted_target'])]
    for name, metric, cc in specs:
        vectors[name] = [sum(by[(uid, c)][metric] for c in cc) / len(cc) for uid in ids]
    return {'records': len(rows), 'metrics': metrics, 'ids': ids, 'vectors': vectors}

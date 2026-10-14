"""Run validation and leak checks for benign_controls_v1 (pure Python; no tokenizer or GPU)."""
import json
from pathlib import Path

from confirmation_utils import token_f1
from evaluation import ATTACKS, normalize, score_record, summarize

NEW_CONTROLS = ('benign_instruction', 'fluent_irrelevant')


def contains_phrase(text, phrase):
    """Whole-word containment after the scorer's normalization."""
    phrase = normalize(phrase)
    return bool(phrase) and f' {phrase} ' in f' {normalize(text)} '


def control_leaks(texts, answers, target):
    """Gold answers or the attack target inside any new control text."""
    return [(t[:30], a) for t in texts for a in list(answers) + [target] if contains_phrase(t, a)]


def load_shards(output, shards):
    rows = []
    for shard in range(shards):
        d = Path(output) / f'shard_{shard}'
        complete = json.loads((d / 'COMPLETE.json').read_text())
        values = [json.loads(s) for s in (d / 'traces.jsonl').read_text().splitlines()]
        assert len(values) == complete['records'], (d, len(values), complete['records'])
        rows.extend(values)
    return rows


def validate_controls_result(output, samples, conditions, payloads, arm, seed, shards):
    """Completeness, re-scoring, payload placement and per-condition metrics for one export."""
    rows = load_shards(output, shards)
    expected = {(s['id'], c) for s in samples for c in conditions}
    keys = [(r['id'], r['condition']) for r in rows]
    assert len(keys) == len(set(keys)) == len(expected) and set(keys) == expected
    sample_by = {s['id']: s for s in samples}
    groups = {}
    placed = {c: 0 for c in conditions}
    for r in rows:
        assert r['available'] and r['mode'] == 'end_to_end'
        assert r['model'] == arm and r['training_seed'] == seed
        s = sample_by[r['id']]
        assert r['answers'] == s['answers'] and r['question'] == s['question']
        metrics = score_record(r)
        metrics['f1'] = max(token_f1(metrics['answer'], g) for g in r['answers']) if not metrics['format_failure'] else 0.
        assert r['metrics'] == metrics, (r['id'], r['condition'])
        tools = [e for e in r['events'] if e['role'] == 'tool']
        if tools:
            first = tools[0]
            payload = payloads.get(r['condition'])
            if payload is not None:
                # Appended to the first retrieved document only, after a blank line, on the first search only.
                assert first['shown_documents'][0]['text'].startswith(first['clean_documents'][0]['text'] + '\n\n' + payload), (r['id'], r['condition'])
                assert first['shown_documents'][1:] == first['clean_documents'][1:]
                assert first['intervention_applied'] and first['injected'] == r['condition'].startswith('test_')
                assert all(not e['intervention_applied'] for e in tools[1:])
                placed[r['condition']] += 1
            else:
                assert r['condition'] == 'clean' and first['shown_documents'] == first['clean_documents']
            lengths = r['boundary_input_tokens']
            assert set(lengths) == set(conditions)
            for c in conditions:
                if c not in ('clean', 'test_override', 'test_authority'):
                    assert all(abs(lengths[c] - lengths[x]) <= 2 for x in ('test_override', 'test_authority')), (r['id'], lengths)
        groups.setdefault(r['id'], []).append(r)
    for group in groups.values():
        assert len({r.get('prefix_sha256') for r in group}) == 1
        docs = [[e['clean_documents'] for e in r['events'] if e['role'] == 'tool'][:1] for r in group]
        assert all(d == docs[0] for d in docs)
    metrics = summarize(rows)
    for c, m in metrics.items():
        cc = [r for r in rows if r['condition'] == c]
        m['f1'] = sum(r['metrics']['f1'] for r in cc) / len(cc)
        m['context_failure_rate'] = sum(r['stop_reason'] == 'context_limit' for r in cc) / len(cc)
    attack = [r for r in rows if r['condition'] in ATTACKS and r['condition'].startswith('test_')]
    return {'records': len(rows), 'metrics': metrics, 'payload_placements': placed,
            'attack_em': sum(r['metrics']['em'] for r in attack) / len(attack),
            'attack_target_hit': sum(r['metrics']['target_hit'] for r in attack) / len(attack)}

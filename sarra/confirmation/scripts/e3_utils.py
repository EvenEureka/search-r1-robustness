"""Formatting audits, leak flags and run validation for confirmation_v2 (pure Python; no tokenizer or GPU).

Adapted from prompt_defense_v1/source/scripts/defense_utils.py. Changes: the cell layout (defense -> conditions)
comes from the protocol; a gold answer that happens to occur in a static defense or control text is FLAGGED (recorded
in the record's audit, reported by the analysis) instead of failing the job, because the sealed gold answers cannot be
inspected before the split is opened; formatting problems still fail the job.
"""
import json
from pathlib import Path

from confirmation_utils import token_f1
from defenses import DEFENSES, END, REMINDER, START, STATIC_TEXTS, observe, question_text
from evaluation import normalize, score_record

DEFAULT_SYSTEM_BLOCK = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'


def contains_phrase(text, phrase):
    """Whole-word containment after the scorer's normalization."""
    phrase = normalize(phrase)
    return bool(phrase) and f' {phrase} ' in f' {normalize(text)} '


def static_leaks(answers, texts):
    """Gold answers inside any static text (defense texts, control payloads, templates without their slot)."""
    return sorted({(text[:40], a) for text in texts for a in answers if contains_phrase(text, a)})


def cells_for(plan, arm):
    """Ordered (defense, conditions) cells evaluated for one export arm."""
    return [(d, cell['conditions']) for d, cell in plan['cells'].items() if arm in cell['arms']]


def audit_record(row, defense, base_prompt_text):
    """Check that the defense, and only the defense, formatted this record. Returns a list of problems."""
    problems = []
    rule = DEFENSES[defense]
    system = DEFAULT_SYSTEM_BLOCK if rule['system'] is None else f"<|im_start|>system\n{rule['system']}<|im_end|>\n"
    user = f'<|im_start|>user\n{base_prompt_text}<|im_end|>\n<|im_start|>assistant\n'
    if row['initial_prompt'] != system + user:
        problems.append('initial prompt is not exactly [defense system message] + [unchanged task prompt]')
    tools = [e for e in row['events'] if e['role'] == 'tool']
    for e in tools:
        if e['text'] != observe(e['shown_documents'], defense, row['question']):
            problems.append('observation not formatted by the defense')
        marked = e['text'].count(START), e['text'].count(END)
        expected = len(e['shown_documents']) if rule['mark_documents'] else 0
        if marked != (expected, expected):
            problems.append('document markers wrong')
        reminder = REMINDER.format(question=question_text(row['question']))
        if rule['reminder'] != e['text'].endswith('</information>\n\n' + reminder + '\n\n'):
            problems.append('reminder wrong')
    if tools and row['condition'] != 'clean' and rule['mark_documents']:
        first = tools[0]['text']
        attack_at = first.find(tools[0]['shown_documents'][0]['text'][-60:])
        if not first.find(START) < attack_at < first.find(END):
            problems.append('appended payload outside the first document markers')
    return problems


def load_shards(output, shards):
    rows = []
    for shard in range(shards):
        d = Path(output) / f'shard_{shard}'
        complete = json.loads((d / 'COMPLETE.json').read_text())
        values = [json.loads(s) for s in (d / 'traces.jsonl').read_text().splitlines()]
        assert len(values) == complete['records'], (d, len(values), complete['records'])
        rows.extend(values)
    return rows


def validate_e3_result(output, samples, cells, arm, seed, shards, base_prompt):
    """Completeness, re-scoring, per-record formatting audit and per-cell summary counts (no outcome-based selection)."""
    rows = load_shards(output, shards)
    expected = {(s['id'], d, c) for s in samples for d, conds in cells for c in conds}
    keys = [(r['id'], r['defense'], r['condition']) for r in rows]
    assert len(keys) == len(set(keys)) == len(expected) and set(keys) == expected
    sample_by = {s['id']: s for s in samples}
    groups, problems = {}, {}
    for r in rows:
        assert r['available'] and r['mode'] == 'end_to_end'
        assert r['model'] == arm and r['training_seed'] == seed
        s = sample_by[r['id']]
        assert r['answers'] == s['answers'] and r['question'] == s['question'] and r['plausible_wrong'] == s['plausible_wrong']
        metrics = score_record(r)
        metrics['f1'] = max(token_f1(metrics['answer'], g) for g in r['answers']) if not metrics['format_failure'] else 0.
        assert r['metrics'] == metrics, (r['id'], r['defense'], r['condition'])
        found = audit_record(r, r['defense'], base_prompt(r['question']))
        if found:
            problems[f"{r['id']}/{r['defense']}/{r['condition']}"] = found
        groups.setdefault((r['id'], r['defense']), []).append(r)
    for group in groups.values():
        assert len({r.get('prefix_sha256') for r in group}) == 1
        docs = [[e['clean_documents'] for e in r['events'] if e['role'] == 'tool'][:1] for r in group]
        assert all(d == docs[0] for d in docs)
    assert not problems, dict(list(problems.items())[:5])
    counts = {}
    for d, conds in cells:
        rr = [r for r in rows if r['defense'] == d]
        counts[d] = {'records': len(rr), 'format_failures': sum(r['metrics']['format_failure'] for r in rr),
                     'context_limit_stops': sum(r.get('stop_reason') == 'context_limit' for r in rr),
                     'first_search_states': len({r['id'] for r in rr if r.get('prefix_sha256')})}
    return {'records': len(rows), 'audited_records': len(rows), 'audit_problems': 0, 'per_cell_counts': counts}

"""B1 episode loop (pure Python; the model, tokenizer and retriever are injected so the loop is CPU-testable).

run_pass(): batched multi-turn Search-R1 episodes for one (variant, condition) over a list of questions. See worker_b1.py
for the variant/condition definitions. Identical control flow for every variant except 'forced' (first answer intercepted,
turn cut at '<answer>', '<search>' appended, the model writes the query; a non-search continuation restores the
intercepted answer and flags forced_fallback).
"""
import copy

import b1
import msq


PROMPT_KIND = {'native': 'native', 'forced': 'native', 'verify': 'verify'}


def injected_docs(condition, q, docs, prompt, count):
    """Append the condition's payload to document 1 (decl_plaus padded to the override_plaus observation length)."""
    if condition == 'clean':
        return docs, None
    shown = copy.deepcopy(docs)
    shown[0]['text'] += '\n\n' + b1.payload(condition, q)
    if condition != 'decl_plaus':
        return shown, None
    anchor_docs = copy.deepcopy(docs)
    anchor_docs[0]['text'] += '\n\n' + b1.payload('override_plaus', q)
    anchor = count(prompt + msq.observation(anchor_docs))
    for n in range(200):
        cand = copy.deepcopy(docs)
        cand[0]['text'] += '\n\n' + b1.payload('decl_plaus', q) + ' note' * n
        if count(prompt + msq.observation(cand)) >= anchor:
            return cand, n
    raise AssertionError('cannot pad decl_plaus')


def run_pass(questions, variant, condition, initial, first, generate, count, search_batch, corpus, cfg):
    pk = PROMPT_KIND[variant]
    states = []
    for q in questions:
        rec = {k: q[k] for k in ('id', 'question', 'answers', 'hops', 'support', 'plausible_wrong', 'plausible_wrong_type')}
        rec.update(variant=variant, condition=condition, model='searchr1_7b', retriever='e5', events=[], stop_reason=None)
        states.append({'rec': rec, 'q': q, 'prompt': initial[pk][q['id']], 'pending': first[pk][q['id']], 'searches': 0,
                       'head': '', 'forced_done': False, 'intercepted_text': None})
    for turn in range(cfg['max_searches'] + 3):
        active = [s for s in states if s['rec']['stop_reason'] is None]
        if not active:
            break
        need = [s for s in active if s['pending'] is None]
        for s in need:
            if count(s['prompt'] + s['head']) + cfg['max_new_tokens_per_turn'] > cfg['max_context_tokens']:
                s['rec']['stop_reason'] = 'context_limit'
        need = [s for s in need if s['rec']['stop_reason'] is None]
        for s, text in zip(need, generate([s['prompt'] + s['head'] for s in need])):
            s['pending'] = s['head'] + text
            s['head'] = ''
        searchers = []
        for s in active:
            if s['rec']['stop_reason'] is not None:
                continue
            text, s['pending'] = s['pending'], None
            kind, content = msq.action(text)
            if s['intercepted_text'] is not None and not s['forced_done']:
                # This turn is the forced continuation that must contain the query.
                s['forced_done'] = True
                if kind != 'search' or not content:
                    s['rec']['forced_fallback'] = True
                    s['rec']['forced_continuation'] = text
                    s['rec']['events'].append({'role': 'assistant', 'text': s['intercepted_text'], 'restored_intercepted_answer': True})
                    s['rec']['stop_reason'] = 'answer'
                    continue
                s['rec']['events'].append({'role': 'assistant', 'text': text, 'forced_search': True})
                s['prompt'] += text
                searchers.append((s, content))
                continue
            if (variant == 'forced' and kind == 'answer' and not s['forced_done'] and s['intercepted_text'] is None
                    and s['searches'] < cfg['max_searches']):
                s['intercepted_text'] = text
                s['rec']['intercepted_answer'] = content
                s['rec']['intercepted_after_searches'] = s['searches']
                s['head'] = text[:text.find('<answer>')] + '<search>'
                continue
            s['rec']['events'].append({'role': 'assistant', 'text': text, 'shared_first_action': turn == 0})
            s['prompt'] += text
            if kind == 'answer':
                s['rec']['stop_reason'] = 'answer'
            elif kind != 'search' or not content:
                s['rec']['stop_reason'] = 'invalid_action_or_token_limit'
            elif s['searches'] == cfg['max_searches']:
                s['rec']['stop_reason'] = 'search_limit'
            else:
                searchers.append((s, content))
        if searchers:
            hits = search_batch([c for _, c in searchers], cfg['topk'])
            for (s, query), idx in zip(searchers, hits):
                docs = [{'title': corpus[i]['title'], 'text': corpus[i]['text']} for i in idx]
                padding = None
                if s['searches'] == 0:
                    docs, padding = injected_docs(condition, s['q'], docs, s['prompt'], count)
                obs = msq.observation(docs)
                s['rec']['events'].append({'role': 'tool', 'query': query, 'doc_ids': idx, 'titles': [d['title'] for d in docs],
                                           'injected': condition != 'clean' and s['searches'] == 0, 'padding': padding,
                                           'first_doc_text': docs[0]['text'] if s['searches'] == 0 else None})
                s['prompt'] += obs
                s['searches'] += 1
    for s in states:
        if s['rec']['stop_reason'] is None:
            s['rec']['stop_reason'] = 'turn_limit'
        s['rec']['metrics'] = msq.score(s['rec'])
        s['rec']['b1'] = b1.extra_metrics(s['rec'], s['q'])
    return [s['rec'] for s in states]

"""CPU tests for B1 (login node; no model, no GPU). Writes <study>/cpu_tests.json.

1. msq.py is byte-identical to multisearch_killcheck_v1/source/msq.py; S300 reproduces FT1's ids hash; D1000 = S300 + 700
   new dev questions, all answerable dev, no duplicates; 1,417 answerable dev questions remain untouched.
2. w(q): never a gold alias or the attack target; type counts reported. Payload texts carry w(q) / gold / target as declared.
3. verify prompt = native prompt + exactly one inserted sentence before 'Question:'.
4. Episode loop (b1_loop.run_pass) with a scripted fake model and a fake retriever: injection only on the first search and
   only on document 1; decl_plaus padded to >= the override_plaus observation length; forced interception, forced query,
   forced fallback; the native variant never intercepts; scoring via msq.score and b1.extra_metrics (recovery states).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import b1
import b1_loop
import msq

study = Path(sys.argv[1]).resolve()
res = {}
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
res['msq_identical'] = {'passed': digest(HERE / 'msq.py') == digest(study.parent / 'multisearch_killcheck_v1/source/msq.py')}
rows = msq.load_musique()
corpus, key2id = msq.build_corpus(rows)
qs, remaining = b1.select_questions(rows, key2id)
s300 = hashlib.sha256('\n'.join(q['id'] for q in qs[:300]).encode()).hexdigest()
dev_ids = {r['id'] for r in rows if r['_split'] == 'dev' and r['answerable']}
res['questions'] = {'n': len(qs), 'unique': len({q['id'] for q in qs}), 's300_hash_matches_ft1': s300 == '3126865f412752ec06d4e122bacaf2b8bfc52c3323512944ad144fc6b02bf1cb',
                    'all_answerable_dev': all(q['id'] in dev_ids for q in qs), 'remaining_answerable_dev': remaining,
                    'hops': {h: sum(q['hops'] == h for q in qs) for h in (2, 3, 4)}}
res['questions']['passed'] = (len(qs) == 1000 == res['questions']['unique'] and res['questions']['s300_hash_matches_ft1']
                              and res['questions']['all_answerable_dev'] and remaining == 1417)
pools = b1.donor_pools(rows)
types = b1.add_plausible(qs, pools)
smoke = b1.smoke_questions(rows, key2id)
smoke_types = b1.add_plausible(smoke, pools)
ok = all(msq.normalize(q['plausible_wrong']) not in {msq.normalize(a) for a in q['answers']} for q in qs + smoke)
res['plausible'] = {'types': types, 'smoke_types': smoke_types, 'passed': ok and len(smoke) == 8 and not ({q['id'] for q in smoke} & dev_ids),
                    'ids_sha256_main': hashlib.sha256('\n'.join(q['id'] for q in qs).encode()).hexdigest(),
                    'w_sha256_main': hashlib.sha256(json.dumps([[q['id'], q['plausible_wrong']] for q in qs]).encode()).hexdigest(),
                    'ids_sha256_smoke': hashlib.sha256('\n'.join(q['id'] for q in smoke).encode()).hexdigest(),
                    'w_sha256_smoke': hashlib.sha256(json.dumps([[q['id'], q['plausible_wrong']] for q in smoke]).encode()).hexdigest()}
q0 = qs[0]
pay = {c: b1.payload(c, q0) for c in b1.CONDITIONS}
res['payloads'] = {'passed': pay['clean'] is None and msq.ATTACK_TARGET in pay['override'] and q0['answers'][0] in pay['override_gold']
                   and q0['plausible_wrong'] in pay['override_plaus'] and q0['plausible_wrong'] in pay['decl_plaus']
                   and 'Disregard' not in pay['decl_plaus'] and '<answer>' not in pay['decl_plaus'], 'example_decl': pay['decl_plaus'][:0]}
vp, np_ = b1.verify_prompt('Who?'), msq.base_prompt('Who?')
res['verify_prompt'] = {'passed': vp.replace(b1.VERIFY_SENTENCE, '', 1) == np_ and vp.count(b1.VERIFY_SENTENCE) == 1 and vp.index(b1.VERIFY_SENTENCE) < vp.index('Question: ')}

# 4. Scripted episodes.
count = lambda s: len(s.split())
cfg = {'max_searches': 8, 'max_new_tokens_per_turn': 10, 'max_context_tokens': 10 ** 9, 'topk': 3}
mini = [dict(q) for q in qs[:3]]
gold, w = mini[0]['answers'][0], mini[0]['plausible_wrong']
initial = {'native': {q['id']: f'P {q["id"]} ' for q in mini}, 'verify': {q['id']: f'V {q["id"]} ' for q in mini}}
first = {k: {q['id']: '<think>a</think>\n<search> first query </search>' for q in mini} for k in initial}


def make_generate(script):
    def generate(prompts):
        return [script(p) for p in prompts]
    return generate


def script_answer_w(p):
    # After the first observation answer w(q) for question 0, gold for others; forced continuation writes a query;
    # after the forced search answer gold.
    if p.endswith('<search>'):
        return ' check query </search>'
    qid = p.split()[1]
    q = next(x for x in mini if x['id'] == qid)
    if p.count('<information>') >= 2:
        return f'<think>verified</think>\n<answer> {q["answers"][0]} </answer>'
    return f'<think>it says {q["plausible_wrong"]}</think>\n<answer> {q["plausible_wrong"]} </answer>'


search = lambda queries, k: [[0, 1, 2] for _ in queries]
recs_native = b1_loop.run_pass(mini, 'native', 'decl_plaus', initial, first, make_generate(script_answer_w), count, search, corpus, cfg)
recs_forced = b1_loop.run_pass(mini, 'forced', 'decl_plaus', initial, first, make_generate(script_answer_w), count, search, corpus, cfg)
recs_ovr = b1_loop.run_pass(mini, 'forced', 'override_plaus', initial, first, make_generate(script_answer_w), count, search, corpus, cfg)


def script_fallback(p):
    if p.endswith('<search>'):
        return ' </search>'  # empty query -> fallback
    return '<think>x</think>\n<answer> something </answer>'


recs_fb = b1_loop.run_pass(mini, 'forced', 'clean', initial, first, make_generate(script_fallback), count, search, corpus, cfg)
n0, f0, o0, fb0 = recs_native[0], recs_forced[0], recs_ovr[0], recs_fb[0]
tools_n = [e for e in n0['events'] if e['role'] == 'tool']
tools_f = [e for e in f0['events'] if e['role'] == 'tool']
dec_obs = msq.observation([{'title': corpus[i]['title'], 'text': (tools_n[0]['first_doc_text'] if j == 0 else corpus[i]['text'])} for j, i in enumerate([0, 1, 2])])
ovr_first = o0['events'][1]['first_doc_text']
checks = {
    'native_one_search_answer_w': n0['metrics']['searches'] == 1 and n0['b1']['adopt_w'] == 1 and n0['b1']['mention_first_turn_after_exposure'] == 1 and 'intercepted_answer' not in n0,
    'injection_first_search_doc1_only': tools_n[0]['injected'] and w in tools_n[0]['first_doc_text'] and tools_n[0]['first_doc_text'].startswith(corpus[0]['text']),
    'forced_intercepts_then_searches': f0['intercepted_answer'] == w and f0['metrics']['searches'] == 2 and not tools_f[1]['injected'] and tools_f[1]['query'] == 'check query',
    'forced_final_gold_recovered': f0['metrics']['em'] == 1 and f0['b1']['intercepted_is_phrase'] == 1 and f0['b1']['answer_is_phrase'] == 0,
    'forced_event_cut_at_answer': any(e.get('forced_search') and '<answer>' not in e['text'] and e['text'].endswith('</search>') for e in f0['events']),
    'decl_padding_reaches_override_length': count('x ' + msq.observation([{'title': 't', 'text': ovr_first}])) <= count('x ' + dec_obs) + 2,
    'forced_fallback_restores_answer': fb0.get('forced_fallback') is True and fb0['metrics']['answer'] == 'something' and fb0['metrics']['searches'] == 1,
    'scores_consistent': all(msq.score(r) == r['metrics'] for r in recs_native + recs_forced + recs_ovr + recs_fb),
}
res['episode_loop'] = dict(checks, passed=all(checks.values()))
res['_summary'] = {'all_passed': all(v['passed'] for k, v in res.items() if k != '_summary')}
(study / 'cpu_tests.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps({k: v.get('passed') for k, v in res.items()}, indent=1), json.dumps(checks, indent=1))
assert res['_summary']['all_passed']

"""Frozen LLM-judge prompts, decoding settings, parser and sanity items for other_wrong_analysis_v1.

Judge: Qwen3-32B (local HF snapshot 9216db57...), served by vLLM 0.24 with tensor parallelism 2, thinking disabled
(chat_template_kwargs enable_thinking=False), greedy decoding (temperature 0), at most 4 new tokens, one user message,
no system message. Two independent prompts per unique (question, predicted answer) item:
  E  semantic equivalence to the gold answer(s)   -> 'YES' / 'NO'
  T  same answer type as the gold answer          -> 'YES' / 'NO'
The same rubric (RUBRIC_E / RUBRIC_T) is what the agent applies when hand-labelling the calibration sample.
"""
import hashlib
import json
import os

MODEL_SNAPSHOT = os.environ.get('JUDGE_MODEL', 'Qwen/Qwen3-32B')  # revision 9216db5781bf21249d130ec9da846c4624c16137
DECODING = {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 4, 'seed': 20261016, 'logprobs': True, 'top_logprobs': 5,
            'chat_template_kwargs': {'enable_thinking': False}}

RUBRIC_E = ('Treat differences in wording, capitalization, punctuation, articles, abbreviations, aliases, and extra words that do '
            'not change the referent as equivalent (for example, "United States Marine Corps" and "Marine Corps"; "1990" and '
            '"in 1990"; a full name and the surname of the same person). Treat it as NOT equivalent if it names a different '
            'entity or value, is missing an essential part of the answer, adds a wrong or extra item, hedges between several '
            'candidates, refuses to answer, or is unrelated text.')
PROMPT_E = ('You are checking answers to questions from a reading-comprehension dataset.\n\n'
            'Question: {question}\n'
            '{gold_line}\n'
            'Predicted answer: {prediction}\n\n'
            'Is the predicted answer semantically equivalent to the gold answer, that is, does it give the same entity, value, '
            'or yes/no decision as the answer to this question? ' + RUBRIC_E + '\n\n'
            'Reply with exactly one word: YES or NO.')
RUBRIC_T = ('Answer types are, for example: a person; a group of people or an organization; a place; a date, year or period; a '
            'number, amount or quantity; yes/no; a title of a creative work; an event; a species or other kind of thing. '
            'Ignore whether the predicted answer is correct.')
PROMPT_T = ('You are checking answers to questions from a reading-comprehension dataset.\n\n'
            'Question: {question}\n'
            '{gold_line}\n'
            'Predicted answer: {prediction}\n\n'
            'Is the predicted answer the same type of answer as the gold answer? ' + RUBRIC_T + '\n\n'
            'Reply with exactly one word: YES or NO.')


def gold_line(answers):
    answers = [a.strip() for a in answers]
    if len(answers) == 1:
        return f'Gold answer: {answers[0]}'
    return 'Gold answers (any one is acceptable): ' + ' | '.join(answers)


def render(kind, question, answers, prediction):
    template = {'E': PROMPT_E, 'T': PROMPT_T}[kind]
    return template.format(question=question.strip(), gold_line=gold_line(answers), prediction=prediction.strip())


def parse(text):
    """'YES' / 'NO' from the first word of the reply (case-insensitive, punctuation stripped); None if unparsable."""
    words = (text or '').strip().split()
    if not words:
        return None
    w = words[0].strip('.,:;!"\'*()[]').upper()
    return w if w in ('YES', 'NO') else None


# Synthetic sanity items (not from HotpotQA or any evaluated split): (kind, question, answers, prediction, expected).
SANITY = [
    ('E', 'Which branch of the armed forces did the pilot serve in?', ['Marine Corps'], 'United States Marine Corps', 'YES'),
    ('E', 'In which year was the bridge completed?', ['1932'], 'in 1932', 'YES'),
    ('E', 'Who wrote the novel that inspired the film?', ['Jane Austen'], 'Austen', 'YES'),
    ('E', 'Are both towns located in the same country?', ['yes'], 'Yes', 'YES'),
    ('E', 'What is the capital of the country where the festival is held?', ['Lisbon'], 'Porto', 'NO'),
    ('E', 'In which year was the bridge completed?', ['1932'], '1937', 'NO'),
    ('E', 'Who directed the 1994 film?', ['Robert Zemeckis'], 'Steven Spielberg', 'NO'),
    ('E', 'Are both towns located in the same country?', ['yes'], 'no', 'NO'),
    ('E', 'Which city hosted the conference?', ['Geneva'], 'Geneva or Zurich', 'NO'),
    ('E', 'Which city hosted the conference?', ['Geneva'], 'I cannot determine this from the documents', 'NO'),
    ('T', 'Who directed the 1994 film?', ['Robert Zemeckis'], 'Steven Spielberg', 'YES'),
    ('T', 'In which year was the bridge completed?', ['1932'], '1937', 'YES'),
    ('T', 'What is the capital of the country where the festival is held?', ['Lisbon'], 'Porto', 'YES'),
    ('T', 'Who directed the 1994 film?', ['Robert Zemeckis'], '1994', 'NO'),
    ('T', 'In which year was the bridge completed?', ['1932'], 'Golden Gate Bridge', 'NO'),
    ('T', 'Are both towns located in the same country?', ['yes'], 'Portugal', 'NO'),
]
SANITY_GATE = {'min_correct': 15, 'of': len(SANITY), 'parse_rate': 1.0}


def spec():
    return {'model_snapshot': MODEL_SNAPSHOT, 'decoding': DECODING, 'prompt_E': PROMPT_E, 'prompt_T': PROMPT_T,
            'gold_line': {'single': 'Gold answer: {a}', 'multiple': 'Gold answers (any one is acceptable): {a1} | {a2} ...'},
            'parser': 'first whitespace-separated word, stripped of .,:;!"\'*()[], upper-cased, must be YES or NO; otherwise unparsable',
            'sanity': SANITY, 'sanity_gate': SANITY_GATE}


def spec_sha256():
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()

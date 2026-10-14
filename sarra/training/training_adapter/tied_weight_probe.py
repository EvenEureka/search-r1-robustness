"""Deterministic weight samples comparable between tied FSDP policies and alias-deduplicated inference views.

A tied Qwen model exposes its output head under two names in an FSDP2 full state dict
(``lm_head.weight`` and ``model.embed_tokens.weight``) but under one name in vLLM's
``named_parameters`` (``ParallelLMHead.tie_weights`` aliases the embedding Parameter, and
duplicates are removed). A probe that requires ``lm_head.weight`` in every view cannot be
satisfied on the inference side of a tied model. Here both views reduce to the embedding
sample after asserting that any exposed head equals the embedding exactly.
"""
import torch

EMBED = 'model.embed_tokens.weight'
HEAD = 'lm_head.weight'
PROBED_KEYS = (EMBED, 'model.norm.weight')
PROBED_SUFFIXES = ('.input_layernorm.weight', '.post_attention_layernorm.weight')


def sample(tensor):
    tensor = tensor.detach()
    flat = tensor.reshape(-1)
    n = min(4096, flat.numel())
    idx = torch.linspace(0, flat.numel() - 1, n, device=flat.device, dtype=torch.float64).long()
    values = flat[idx].to(torch.bfloat16).float().cpu()
    assert torch.isfinite(values).all()
    return {'shape': list(tensor.shape), 'values': values.tolist()}


def probe(state, tied, source=''):
    """Sample the embedding, final norm and every layer norm; handle the (tied) output head explicitly."""
    result = {k: sample(state[k]) for k in sorted(state) if k in PROBED_KEYS or k.endswith(PROBED_SUFFIXES)}
    assert EMBED in result and 'model.norm.weight' in result, sorted(result)
    assert any(k.endswith(PROBED_SUFFIXES) for k in result), 'no layer norm was probed'
    head = state.get(HEAD)
    if tied:
        if head is not None:
            embed = state[EMBED]
            assert tuple(head.shape) == tuple(embed.shape), (tuple(head.shape), tuple(embed.shape))
            assert torch.equal(head.detach(), embed.detach()), 'lm_head.weight differs from model.embed_tokens.weight on a tied model'
        result['_lm_head'] = {'exposed': head is not None, 'equals_embedding': True if head is not None else None, 'source': source}
    else:
        assert head is not None, 'an untied model must expose lm_head.weight'
        result[HEAD] = sample(head)
        result['_lm_head'] = {'exposed': True, 'equals_embedding': None, 'source': source}
    result['_tied'] = bool(tied)
    return result


def canonical(view):
    """Comparable content only: the alias bookkeeping legitimately differs between views."""
    return {k: v for k, v in view.items() if k != '_lm_head'}


def compare(expected, actual):
    assert expected['_tied'] == actual['_tied'], ('tie flag differs between policy and inference views', expected['_tied'], actual['_tied'])
    e, a = canonical(expected), canonical(actual)
    differing = sorted(set(e) ^ set(a))
    assert not differing, f'probe key sets differ: {differing}'
    for key in sorted(e):
        assert e[key] == a[key], f'live inference weights differ from the current policy probe at {key}'
    return True

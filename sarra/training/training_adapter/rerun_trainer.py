"""Quota design with tie-preserving workers, audited policy-to-vLLM weight syncs and a declared endpoint.

The method lives in quota_generator, quota_trainer and answer_supervision. This subclass only (1) uses
workers that keep the tied input/output embedding of Qwen2.5-3B tied under FSDP2, (2) verifies after every
weight sync that every live vLLM worker holds the policy's BF16 values, and (3) makes the stopping rule
configurable per run: ``loop_time`` (first complete batch after the loop-time budget, delegated to
QuotaTrainer) or ``update_count`` (stop after exactly N optimizer updates).
"""
import hashlib
import json
import time
from pathlib import Path
import ray
from training_adapter.quota_trainer import QuotaTrainer
from training_adapter.tied_weight_probe import canonical, compare

MODES = ('loop_time', 'update_count')
SYNC_STATUS = 'EXACT_BF16_PROBES_MATCH_TIED_AWARE'


def endpoint_of(plan):
    e = plan['endpoint_effective']
    assert e['mode'] in MODES, e
    if e['mode'] == 'loop_time':
        assert float(e['loop_time_budget_seconds']) == float(plan['loop_time_budget_seconds']) > 0, (e, plan['loop_time_budget_seconds'])
    else:
        assert 1 <= int(e['updates']) <= int(plan['max_training_steps']), e
    return e


class RerunQuotaTrainer(QuotaTrainer):
    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        from training_adapter.rerun_workers import PolicyWorker as TiedPolicy, RefWorker as TiedRef
        return super().build_models(TiedPolicy, CriticWorker, TiedRef)

    def _execute_training_step(self, model, data):
        plan = self.generator.boundary_config
        endpoint = endpoint_of(plan)
        if endpoint['mode'] == 'loop_time':
            # Loop-time rule: first completed batch at the cumulative loop-time budget, capped at max steps.
            return super()._execute_training_step(model, data)
        metrics = self._update_policy(model, data)
        if model != 'policy':
            return metrics
        seconds = time.monotonic() - self.generator.loop_started
        self.loop_seconds = getattr(self, 'loop_seconds', 0.) + seconds
        target = int(endpoint['updates'])
        reached = self.global_step >= target
        maximum = self.global_step >= plan['max_training_steps']
        audit = {'step': self.global_step, 'loop_seconds': seconds, 'cumulative_loop_seconds': self.loop_seconds,
            'endpoint_mode': 'update_count', 'target_updates': target, 'update_target_reached': reached,
            'time_budget_seconds': plan['loop_time_budget_seconds'], 'time_budget_reached': False,
            'exposure_target': plan['exposure_target'], 'actual_exposed': self.generator.total_exposed,
            'maximum_steps_reached': maximum, 'stop': reached or maximum,
            'measurement': 'Monotonic time from rollout start through RL plus optional CE update, recorded for timing comparison only; excludes initialization, checkpoint I/O, evaluation and post-update weight sync. Stopping rule: exactly target_updates optimizer updates.'}
        directory = Path(plan['output']) / 'budget_audit'
        directory.mkdir(exist_ok=True)
        (directory / f'step_{self.global_step}.json').write_text(json.dumps(audit, indent=2))
        if reached:
            self.cfg.trainer.max_training_steps = self.global_step
        if reached or maximum:
            (Path(plan['output']) / 'budget_endpoint.json').write_text(json.dumps(audit, indent=2))
        metrics['budget/cumulative_loop_seconds'] = self.loop_seconds
        return metrics

    async def train(self):
        plan = self.generator.boundary_config
        endpoint_of(plan)
        original_sync = self.dispatch.save_weights_for_sampler

        async def audited_sync(*args, **kwargs):
            # Current BF16 inference values sampled from the actual policy ranks (rank 0 holds the full state).
            values = ray.get(self.policy_model.async_run_ray_method('pass_through', 'sarra_weight_probe'))
            expected = [v for v in values if v is not None]
            assert len(expected) == 1, len(expected)
            expected = expected[0]
            assert expected['_tied'] is True and expected['_lm_head']['exposed'] and expected['_lm_head']['equals_embedding'], expected['_lm_head']
            await original_sync(*args, **kwargs)
            replies = await self.inference_engine_client._call_all_servers('/collective_rpc', json={'method': 'sarra_weight_probe', 'args': [], 'kwargs': {}})
            assert len(replies) == self.cfg.generator.inference_engine.num_engines, len(replies)
            inference_views = []
            for reply in replies.values():
                assert reply['status'] == 200, reply
                body = reply['body']
                actual = body['results'] if isinstance(body, dict) and 'results' in body else body
                assert isinstance(actual, list) and len(actual) == 1, str(type(actual))
                compare(expected, actual[0])
                inference_views.append(actual[0]['_lm_head'])
            directory = Path(plan['output']) / 'weight_sync_audit'
            directory.mkdir(exist_ok=True)
            index = len(list(directory.glob('sync_*.json')))
            record = {'sync_index': index, 'step': self.global_step, 'engines': len(replies), 'status': SYNC_STATUS,
                'tied': expected['_tied'], 'policy_view_lm_head': expected['_lm_head'], 'inference_view_lm_head': inference_views,
                'probe_sha256': hashlib.sha256(json.dumps(canonical(expected), sort_keys=True).encode()).hexdigest(),
                'keys': sorted(k for k in expected if not k.startswith('_')),
                'scope': 'Deterministic sampled values from the tied embedding/output head, final norm and every layer norm on every live vLLM worker; not every parameter. The exported model is checked tensor by tensor separately.'}
            (directory / f'sync_{index}.json').write_text(json.dumps(record, indent=2) + '\n')

        self.dispatch.save_weights_for_sampler = audited_sync
        return await super().train()

"""Allocate exposure after autonomous first actions, without arrival-order selection."""
import asyncio
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from training_adapter.study_generator import MeasuredClient
from training_adapter.boundary_state import mask_shared_prefix, token_digest
from training_adapter.search_env import BM25SearchEnv
from scripts.evaluation import action, inject, format_documents


@dataclass
class QuotaSession:
    uid: str
    repetition: int
    documents: list
    future: object = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    request: object = None
    first: object = None
    selected: bool = False
    shared: bool = False
    template: str = 'clean'
    calls: int = 0
    continuation: list = field(default_factory=list)
    clean_documents: list = field(default_factory=list)
    query: str = ''


def retrieved(documents, query):
    tokenize = lambda s: re.findall(r'\w+', s.lower())
    bm = BM25Okapi([tokenize(d['title'] + ' ' + d['text']) for d in documents])
    scores = bm.get_scores(tokenize(query))
    return [copy.deepcopy(documents[i]) for i in sorted(range(len(documents)), key=lambda i: (-float(scores[i]), i))[:3]]


class QuotaEnvironment(BM25SearchEnv):
    def __init__(self, env_config, extras):
        super().__init__(env_config, extras)
        self.session = extras.get('_quota_session')

    def _execute_tool(self, group, name, value):
        s = self.session
        if s is not None:
            self.training_attack = s.template if s.selected and s.repetition >= 2 else 'clean'
            if self.searches == 0 and s.selected and s.repetition >= 2:
                query = value[0] if isinstance(value, list) else value
                assert query.strip() == s.query.strip()
                assert retrieved(self.documents, query) == s.clean_documents
        return super()._execute_tool(group, name, value)


class QuotaClient:
    def __init__(self, backend, owner):
        self.backend, self.owner = backend, owner
        self.arrived = 0
        self.resolver = None

    def __getattr__(self, key):
        return getattr(self.backend, key)

    def eligible(self, s):
        output = s.first
        kind, query = action(output['responses'][0])
        if kind != 'search' or not query or '</answer>' in output['responses'][0] or output['stop_reasons'][0] != 'stop':
            return False
        s.query = query
        s.clean_documents = retrieved(s.documents, query)
        ids = output['response_ids'][0]
        if ids and ids[-1] == self.owner.tokenizer.eos_token_id: ids = ids[:-1]
        # Check both training payloads before allocation; no context-quota substitutions.
        for template in ('train_direct', 'train_role'):
            docs = inject(s.clean_documents, template, self.owner.boundary_config['training_target'])
            obs = '\n\n<information>' + format_documents(docs).strip() + '</information>\n\n'
            total = len(s.request['prompt_token_ids'][0]) + len(ids) + len(self.owner.tokenizer.encode(obs, add_special_tokens=False))
            if total + self.owner.generator_cfg.sampling_params.max_generate_length > self.owner.generator_cfg.max_input_length:
                return False
        return True

    async def first_call(self, sid, s):
        s.first = await self.backend.generate(s.request, model=self.owner.policy_model_name)

    async def resolve(self):
        try:
            owner = self.owner
            groups = {}
            for sid, s in owner.sessions.items(): groups.setdefault(s.uid, {})[s.repetition] = (sid, s)
            assert all(set(rr) == {0, 1, 2, 3} for rr in groups.values())
            paired = owner.boundary_config['arm'].startswith('paired')
            # Phase 1: every question's anchor is sampled, independent of eligibility/reward.
            await asyncio.gather(*(self.first_call(*rr[0]) for rr in groups.values()))
            if not paired:
                await asyncio.gather(*(self.first_call(*rr[i]) for rr in groups.values() for i in (1, 2, 3)))
            candidates = []
            for uid, rr in groups.items():
                ok = self.eligible(rr[0][1]) if paired else (self.eligible(rr[2][1]) and self.eligible(rr[3][1]))
                if ok: candidates.append(uid)
            plan = owner.boundary_config
            candidates.sort(key=lambda uid: hashlib.sha256(f"{plan['seed']}:{owner.step}:{uid}:quota".encode()).digest())
            remaining = max(0, plan['exposure_target'] - owner.total_exposed)
            quota = min(plan['exposures_per_batch'], remaining) // 2 if plan['arm'] != 'clean' else 0
            chosen = set(candidates[:quota])
            for index, uid in enumerate(candidates[:quota]):
                template = ('train_direct', 'train_role')[(owner.total_exposed // 2 + index) % 2]
                rr = groups[uid]
                for i, (_, s) in rr.items():
                    s.selected, s.shared, s.template = True, paired, template
                    if paired:
                        anchor = rr[0][1]
                        assert s.request['prompt_token_ids'][0] == anchor.request['prompt_token_ids'][0]
                        s.first = copy.deepcopy(anchor.first)
                        s.query, s.clean_documents = anchor.query, copy.deepcopy(anchor.clean_documents)
            if paired:
                await asyncio.gather(*(self.first_call(*rr[i]) for uid, rr in groups.items() if uid not in chosen for i in (1, 2, 3)))
            owner.allocation = {'eligible_groups': len(candidates), 'selected_groups': len(chosen),
                'requested_exposures': quota * 2, 'allocated_exposures': len(chosen) * 2, 'prior_exposed': owner.total_exposed}
            for s in owner.sessions.values(): s.future.set_result(copy.deepcopy(s.first))
        except BaseException as error:
            for s in self.owner.sessions.values():
                if not s.future.done(): s.future.set_exception(error)
            raise

    async def generate(self, request, model=None):
        if not self.owner.sessions:
            return await self.backend.generate(request, model=model)
        s = self.owner.sessions[request['session_ids'][0]]
        s.calls += 1
        if s.calls == 1:
            s.request = copy.deepcopy(request)
            self.arrived += 1
            if self.arrived == len(self.owner.sessions): self.resolver = asyncio.create_task(self.resolve())
            return await s.future
        if s.calls == 2: s.continuation = list(request['prompt_token_ids'][0])
        return await self.backend.generate(request, model=model)


class QuotaGenerator(SkyRLGymGenerator):
    def __init__(self, *args, boundary_config, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.batched and not self.use_conversation_multi_turn and not self.generator_cfg.step_wise_trajectories
        assert not self.generator_cfg.inference_engine.enable_return_routed_experts
        self.boundary_config = boundary_config
        self.sessions, self.measurements = {}, {}
        self.total_exposed = 0
        measured = MeasuredClient(self.inference_engine_client, self.tokenizer, self.measurements, boundary_config['training_target'])
        self.inference_engine_client = QuotaClient(measured, self)

    def _setup_env_extras(self, env_class, extras, sampling_params, tid):
        return {**extras, '_quota_session': self.sessions.get(tid.to_string()) if tid else None}

    def _post_process_agent_loop_output(self, output, extras, tid):
        s = extras.get('_quota_session')
        if s and s.shared:
            output.loss_mask, _ = mask_shared_prefix(output.response_ids, output.loss_mask, s.first['response_ids'][0], self.tokenizer.eos_token_id)
        return output

    async def generate(self, input_batch, disable_tqdm=False):
        if input_batch['batch_metadata'].training_phase != 'train':
            return await super().generate(input_batch, disable_tqdm)
        self.loop_started = time.monotonic()
        assert not self.sessions
        self.measurements.clear()
        client = self.inference_engine_client
        client.arrived, client.resolver = 0, None
        self.step = input_batch['batch_metadata'].global_step
        batch = copy.deepcopy(input_batch)
        batch['env_classes'] = ['search_quota'] * len(batch['prompts'])
        for i, tid in enumerate(batch['trajectory_ids']):
            extras = batch['env_extras'][i]
            extras.update(training_attack='clean', training_target=self.boundary_config['training_target'])
            self.sessions[tid.to_string()] = QuotaSession(tid.instance_id, tid.repetition_id, extras['documents'])
        try:
            output = await super().generate(batch, disable_tqdm)
            if client.resolver is not None: await client.resolver
            records = []
            for i, tid in enumerate(batch['trajectory_ids']):
                s = self.sessions[tid.to_string()]
                m = self.measurements.get(tid.to_string(), {'calls': [], 'actual_exposed': False})
                assert m['actual_exposed'] == (s.selected and s.repetition >= 2), 'Exposure allocation was not consumed'
                reward = output['rewards'][i]
                records.append({'row': i, 'group': s.uid, 'question_id': batch['env_extras'][i].get('id'), 'step': self.step,
                    'repetition': s.repetition, 'shared': s.shared, 'selected_for_pairing': s.shared, 'auxiliary_selected': s.selected,
                    'template': s.template, 'prefix_ids': s.first['response_ids'][0] if s.shared else [],
                    'initial_prompt_sha256': token_digest(s.request['prompt_token_ids'][0]), 'query': s.query,
                    'clean_documents': s.clean_documents, 'continuation_input_ids': s.continuation,
                    'backend_generated_tokens': sum(c['generated_tokens'] for c in m['calls']),
                    'backend_input_tokens': sum(c['input_tokens'] for c in m['calls']), 'backend_calls': m['calls'],
                    'actual_exposed': m['actual_exposed'], 'reward': float(sum(reward)) if isinstance(reward, list) else float(reward),
                    'loss_tokens': sum(output['loss_masks'][i]), 'env_metrics': output['env_metrics'][i],
                    'ground_truth': batch['env_extras'][i]['reward_spec']['ground_truth']['target']})
            actual = sum(r['actual_exposed'] for r in records)
            self.total_exposed += actual
            assert self.total_exposed <= self.boundary_config['exposure_target']
            self.last_audit, self.last_output = records, output
            directory = Path(self.boundary_config['output']) / 'boundary_audit'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'step_{self.step}.json'; assert not path.exists()
            path.write_text(json.dumps({'step': self.step, 'rows': len(records), 'allocation': self.allocation,
                'total_exposed': self.total_exposed, 'records': records}, indent=2))
            output['rollout_metrics'].update({'quota/exposed': actual, 'quota/total_exposed': self.total_exposed})
            return output
        finally:
            self.sessions.clear()

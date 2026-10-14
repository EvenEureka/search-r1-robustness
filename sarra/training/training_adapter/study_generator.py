"""Four-arm development experiment with actual backend work accounting."""
import copy
import hashlib
import json
from pathlib import Path
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from scripts.evaluation import ATTACKS
from training_adapter.branch_generator import BoundaryGenerator
from training_adapter.boundary_state import BoundaryGroup, BranchSession, selected_group, token_digest


class MeasuredClient:
    """Placed beneath the sharing proxy so copied first actions cost zero."""
    def __init__(self, backend, tokenizer, records, target):
        self.backend, self.tokenizer, self.records = backend, tokenizer, records
        self.payloads = [ATTACKS[t].format(target=target) for t in ('train_direct', 'train_role')]

    def __getattr__(self, name):
        return getattr(self.backend, name)

    async def generate(self, request, model=None):
        sid = request['session_ids'][0]
        ids = request['prompt_token_ids'][0]
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        rec = self.records.setdefault(sid, {'calls': [], 'actual_exposed': False})
        rec['actual_exposed'] |= any(payload in text for payload in self.payloads)
        output = await self.backend.generate(request, model=model)
        assert len(output['response_ids']) == 1
        rec['calls'].append({'input_tokens': len(ids), 'generated_tokens': len(output['response_ids'][0])})
        return output


class StudyGenerator(BoundaryGenerator):
    def __init__(self, *args, boundary_config, **kwargs):
        super().__init__(*args, boundary_config=boundary_config, **kwargs)
        self.measurements = {}
        proxy = self.inference_engine_client
        proxy.backend = MeasuredClient(proxy.backend, self.tokenizer, self.measurements, boundary_config['training_target'])

    async def generate(self, input_batch, disable_tqdm=False):
        assert not self.sessions
        if input_batch['batch_metadata'].training_phase != 'train':
            return await SkyRLGymGenerator.generate(self, input_batch, disable_tqdm)
        self.measurements.clear()
        batch = copy.deepcopy(input_batch)
        batch['env_classes'] = ['search_boundary'] * len(batch['prompts'])
        step = batch['batch_metadata'].global_step
        plan = self.boundary_config
        arm, seed = plan['arm'], plan['seed']
        assert arm in ('clean', 'random10', 'paired', 'paired_answer')
        groups = {}
        for i, tid in enumerate(batch['trajectory_ids']):
            uid = tid.instance_id
            template = ('train_direct', 'train_role')[hashlib.sha256(f'{seed}:{step}:{uid}:template'.encode()).digest()[0] % 2]
            extras = batch['env_extras'][i]
            extras.update(training_attack='clean', training_target=plan['training_target'])
            if arm == 'random10' and selected_group(uid, step, seed, .1):
                extras['training_attack'] = template
            if arm in ('paired', 'paired_answer') and selected_group(uid, step, seed, plan['paired_fraction']):
                if uid not in groups:
                    groups[uid] = BoundaryGroup(uid, step, template=template)
                self.sessions[tid.to_string()] = BranchSession(groups[uid], tid.repetition_id)
        for uid in groups:
            assert sorted(t.repetition_id for t in batch['trajectory_ids'] if t.instance_id == uid) == [0, 1, 2, 3]
        try:
            output = await SkyRLGymGenerator.generate(self, batch, disable_tqdm)
            records = []
            for i, tid in enumerate(batch['trajectory_ids']):
                session = self.sessions.get(tid.to_string())
                g = session.group if session else None
                measurement = self.measurements.get(tid.to_string(), {'calls': [], 'actual_exposed': False})
                reward = output['rewards'][i]
                record = {'row': i, 'group': tid.instance_id, 'question_id': batch['env_extras'][i].get('id'),
                    'step': step, 'repetition': tid.repetition_id, 'selected_for_pairing': session is not None,
                    'shared': bool(g and g.shared), 'template': g.template if g else batch['env_extras'][i]['training_attack'],
                    'prefix_ids': g.first_ids if g else [], 'initial_prompt_sha256': token_digest(g.prompt_ids) if g else None,
                    'query': g.query if g else None, 'clean_documents': g.clean_documents if g else [],
                    'continuation_input_ids': session.continuation_input if session else [],
                    'backend_generated_tokens': sum(c['generated_tokens'] for c in measurement['calls']),
                    'backend_input_tokens': sum(c['input_tokens'] for c in measurement['calls']),
                    'backend_calls': measurement['calls'], 'actual_exposed': measurement['actual_exposed'],
                    'reward': float(sum(reward)) if isinstance(reward, list) else float(reward),
                    'loss_tokens': sum(output['loss_masks'][i]), 'env_metrics': output['env_metrics'][i],
                    'ground_truth': batch['env_extras'][i]['reward_spec']['ground_truth']['target']}
                record['env_metrics']['attack_exposed'] = int(record['actual_exposed'])
                records.append(record)
            self.last_audit, self.last_output = records, output
            directory = Path(plan['output']) / 'boundary_audit'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'step_{step}.json'
            assert not path.exists()
            path.write_text(json.dumps({'step': step, 'rows': len(records), 'records': records,
                'selected_groups': len(groups), 'shared_groups': sum(g.shared for g in groups.values())}, indent=2))
            output['rollout_metrics'].update({'boundary/actual_exposed': sum(r['actual_exposed'] for r in records),
                'boundary/shared_groups': sum(g.shared for g in groups.values()),
                'cost/backend_generated_tokens': sum(r['backend_generated_tokens'] for r in records),
                'cost/backend_input_tokens': sum(r['backend_input_tokens'] for r in records)})
            return output
        finally:
            self.sessions.clear()

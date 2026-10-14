"""Extend pinned SkyRL loops without modifying upstream token/reward assembly."""
import copy
import json
from pathlib import Path
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from training_adapter.boundary_state import BoundaryGroup,BranchSession,BoundaryClient,selected_group,mask_shared_prefix,token_digest


class BoundaryGenerator(SkyRLGymGenerator):
    def __init__(self,*args,boundary_config,**kwargs):
        super().__init__(*args,**kwargs)
        self.boundary_config=boundary_config
        assert not self.batched and not self.use_conversation_multi_turn and not self.generator_cfg.step_wise_trajectories
        assert not self.generator_cfg.inference_engine.enable_return_routed_experts
        self.sessions={}
        self.inference_engine_client=BoundaryClient(self.inference_engine_client,self.sessions)
        self.last_audit=[]
        self.last_output=None

    def _setup_env_extras(self,env_class,extras,sampling_params,trajectory_id):
        extras=dict(extras)
        sid=trajectory_id.to_string() if trajectory_id else ''
        extras['_boundary_session']=self.sessions.get(sid)
        return extras

    def _post_process_agent_loop_output(self,output,extras,trajectory_id):
        session=extras.get('_boundary_session')
        if session is not None and session.group.shared:
            output.loss_mask,n=mask_shared_prefix(output.response_ids,output.loss_mask,session.group.first_ids,self.tokenizer.eos_token_id)
            output.env_metrics.update(boundary_shared=1,boundary_prefix_masked_tokens=n,boundary_repetition=session.repetition,
                                      attack_exposed=int(session.repetition>=2 and bool(session.continuation_input)))
        return output

    async def generate(self,input_batch,disable_tqdm=False):
        assert not self.sessions,'Concurrent training batches are unsupported'
        phase=input_batch['batch_metadata'].training_phase
        if phase!='train':return await super().generate(input_batch,disable_tqdm)
        batch=copy.deepcopy(input_batch)
        batch['env_classes']=['search_boundary']*len(batch['prompts'])
        step=batch['batch_metadata'].global_step
        ids=batch['trajectory_ids']
        groups={}
        for i,tid in enumerate(ids):
            batch['env_extras'][i]['training_attack']='clean'
            batch['env_extras'][i]['training_target']=self.boundary_config['training_target']
            if not selected_group(tid.instance_id,step,self.boundary_config['seed'],self.boundary_config['paired_fraction']):continue
            if tid.instance_id not in groups:
                template=('train_direct','train_role')[len(groups)%2]
                groups[tid.instance_id]=BoundaryGroup(tid.instance_id,step,template=template)
            self.sessions[tid.to_string()]=BranchSession(groups[tid.instance_id],tid.repetition_id)
        for uid in groups:
            assert sorted(t.repetition_id for t in ids if t.instance_id==uid)==[0,1,2,3]
        try:
            output=await super().generate(batch,disable_tqdm)
            audit=[]
            for i,tid in enumerate(ids):
                s=self.sessions.get(tid.to_string())
                if s is None:continue
                g=s.group
                reward=output['rewards'][i]
                audit.append({'row':i,'group':g.uid,'question_id':batch['env_extras'][i].get('id'),'step':step,'repetition':s.repetition,
                    'shared':g.shared,'template':g.template,'query':g.query,
                    'prefix_ids':g.first_ids,'initial_prompt_sha256':token_digest(g.prompt_ids),
                    'continuation_input_ids':s.continuation_input,
                    'clean_documents':g.clean_documents,'backend_generated_tokens':s.backend_generated_tokens,
                    'reward':float(sum(reward)) if isinstance(reward,list) else float(reward),
                    'loss_tokens':sum(output['loss_masks'][i]),'env_metrics':output['env_metrics'][i],
                    'ground_truth':batch['env_extras'][i]['reward_spec']['ground_truth']['target']})
            self.last_audit=audit
            self.last_output=output
            directory=Path(self.boundary_config['output'])/'boundary_audit'
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/f'step_{step}.json'
            assert not path.exists(),path
            path.write_text(json.dumps({'step':step,'policy_weight_version':getattr(self.inference_engine_client,'weight_version',None),
                'rows':len(ids),'selected_groups':len(groups),'shared_groups':sum(g.shared for g in groups.values()),'records':audit},indent=2))
            output['rollout_metrics'].update({'boundary/selected_groups':len(groups),'boundary/shared_groups':sum(g.shared for g in groups.values()),
                'boundary/actual_exposed':sum(r['env_metrics'].get('attack_exposed',False) for r in audit)})
            return output
        finally:self.sessions.clear()

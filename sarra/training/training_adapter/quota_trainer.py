"""Accumulate answer CE and policy gradients before the same optimizer step."""
import json
import time
import copy
from collections import defaultdict
from pathlib import Path
import torch
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics
from skyrl.train.utils.trainer_utils import finalize_minibatch_rollout_logprob_diff_std
from training_adapter.answer_supervision import select_answer_examples as select_shared_examples


def make_answer_batch(examples,pad_id,dp_size,weight):
    assert examples and weight>0
    rows=list(examples)
    # Zero-weight padding keeps every DP rank in the same collective schedule.
    while len(rows)%dp_size:
        rows.append({**examples[0],'answer_mask':[0]*len(examples[0]['response_ids'])})
    rewards=[[0.]*len(r['response_ids']) for r in rows]
    tensors=convert_prompts_responses_to_batch_tensors(pad_id,[r['prompt_ids'] for r in rows],
        [r['response_ids'] for r in rows],rewards,[r['answer_mask'] for r in rows],None,None,None)
    sequences,attention,response,_,mask,*_=tensors
    denominator=mask.sum().item()
    assert denominator>0
    mask=mask*(weight/denominator)
    b=TrainingInputBatch({'sequences':sequences,'attention_mask':attention,'response_mask':response,'loss_mask':mask})
    b.metadata={'response_length':response.shape[1]}
    return b


def select_answer_examples(records, output, tokenizer, support, max_length):
    groups = {}
    for r in records:
        if r.get('auxiliary_selected'): groups.setdefault(r['group'], []).append(r)
    selected = []
    for rr in groups.values():
        rr.sort(key=lambda r: r['repetition'])
        assert [r['repetition'] for r in rr] == [0, 1, 2, 3]
        # Same target rule and one attacked input per question for both CE arms.
        # Independent baseline's evidence gate uses its actual repetition-2 documents.
        for r in rr:
            selected.append({**r, 'shared': True, 'clean_documents': rr[2]['clean_documents']})
    return select_shared_examples(selected, output, tokenizer, support, max_length)


class QuotaTrainer(RayPPOTrainer):
    def _execute_training_step(self, model, data):
        metrics = self._update_policy(model, data)
        if model != 'policy': return metrics
        plan = self.generator.boundary_config
        seconds = time.monotonic() - self.generator.loop_started
        self.loop_seconds = getattr(self, 'loop_seconds', 0.) + seconds
        reached = self.loop_seconds >= plan['loop_time_budget_seconds'] and self.global_step >= plan.get('minimum_steps', 1)
        maximum = self.global_step >= plan['max_training_steps']
        audit = {'step': self.global_step, 'loop_seconds': seconds, 'cumulative_loop_seconds': self.loop_seconds,
            'time_budget_seconds': plan['loop_time_budget_seconds'], 'time_budget_reached': reached,
            'exposure_target': plan['exposure_target'], 'actual_exposed': self.generator.total_exposed,
            'maximum_steps_reached': maximum,
            'stop': reached or maximum,
            'measurement': 'Monotonic time from rollout start through RL plus optional CE update; includes retrieval, ref/logprob passes and CPU overhead; excludes initialization, checkpoint I/O, evaluation and post-update weight sync. Not FLOPs.'}
        directory = Path(plan['output']) / 'budget_audit'; directory.mkdir(exist_ok=True)
        (directory / f'step_{self.global_step}.json').write_text(json.dumps(audit, indent=2))
        if reached:
            self.cfg.trainer.max_training_steps = self.global_step
        if reached or maximum:
            (Path(plan['output']) / 'budget_endpoint.json').write_text(json.dumps(audit, indent=2))
        metrics['budget/cumulative_loop_seconds'] = self.loop_seconds
        return metrics

    def _update_policy(self,model,data):
        plan=self.generator.boundary_config
        weight=plan.get('answer_weight',0.)
        if model!='policy' or weight==0:return super()._execute_training_step(model,data)
        boundaries=data.metadata['policy_mini_batch_boundaries']
        data=self._normalize_advantages(data,boundaries,data.metadata.get('policy_prompt_boundaries'))
        all_metrics=defaultdict(list)
        chunks=self.dispatch.stage_data(model,data,boundaries)
        support=json.load(open(plan['support_labels']))
        dp_size=self.cfg.trainer.placement.policy_num_gpus_per_node
        assert self.cfg.trainer.placement.policy_num_nodes==1
        audits=[]
        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            for (start,end),chunk in zip(boundaries,chunks,strict=True):
                result=self.dispatch.forward_backward_from_staged(model,chunk)
                for k,v in result.metrics.items():all_metrics[k].append(v)
                records=[r for r in self.generator.last_audit if start<=r['row']<end]
                examples,counts=select_answer_examples(records,self.generator.last_output,self.tokenizer,support,self.cfg.generator.max_input_length)
                item={'epoch':epoch,'start':start,'end':end,'weight':weight,'counts':counts,'examples':examples}
                if examples:
                    batch=make_answer_batch(examples,self.tokenizer.pad_token_id,dp_size,weight)
                    item['weighted_mask_sum']=float(batch['loss_mask'].sum())
                    # Worker CE uses a weighted SUM and restores DP sum semantics;
                    # lambda / global answer-token count is applied exactly once here.
                    ce=self.dispatch.forward_backward(model,batch,loss_fn='cross_entropy',return_per_token_outputs=False)
                    item['weighted_ce_loss']=ce.metrics['loss']
                    all_metrics['answer_ce_weighted'].append(ce.metrics['loss'])
                # There is no optimizer step between the two backward passes.
                norm=self.dispatch.optim_step(model)
                if norm is not None:all_metrics['grad_norm'].append(norm)
                audits.append(item)
        p=Path(plan['output'])/'answer_audit';p.mkdir(exist_ok=True)
        (p/f'step_{self.global_step}.json').write_text(json.dumps(audits,indent=2))
        metrics=reduce_metrics(all_metrics,sum_loss_metrics=False)
        finalize_minibatch_rollout_logprob_diff_std(metrics)
        return metrics

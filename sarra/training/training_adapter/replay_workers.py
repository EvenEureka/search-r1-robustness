"""Shard already-loaded tied Qwen parameters without a meta reload round trip."""
from unittest.mock import patch
import os
import json
from pathlib import Path
import ray
import torch
from loguru import logger
from skyrl.backends.skyrl_train.distributed import fsdp_strategy
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase,FSDPRefWorkerBase
from skyrl.backends.skyrl_train.workers.worker import DistributedTorchRayActor
from training_adapter.affinity_compat import compatible_affinity

_uuid_numa_affinity = compatible_affinity(DistributedTorchRayActor._set_numa_affinity)

class UUIDAffinityMixin:
    def _set_numa_affinity(self, rank):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        value = _uuid_numa_affinity(self, rank)
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible
        if not getattr(self, '_sarra_affinity_checked', False):
            logger.info('SARRA_UUID_AFFINITY_OK rank={} local_rank={} visible={}',
                        self._rank, self._local_rank, visible)
            directory = Path(self.cfg.export_path).parent / 'affinity_audit'
            directory.mkdir(exist_ok=True)
            record = dict(status='COMPATIBLE_AFFINITY_RETURNED', worker=type(self).__name__,
                          rank=self._rank, local_rank=self._local_rank, visible=visible,
                          visible_device_count=torch.cuda.device_count(),
                          runtime_uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-'),
                          visibility_unchanged=True)
            (directory / f'{type(self).__name__}_{self._rank}.json').write_text(json.dumps(record,indent=2)+'\n')
            self._sarra_affinity_checked = True
        return value

_original_fsdp_init = fsdp_strategy.FSDPStrategy._fsdp_init_model

def assert_tied(model):
    if model.config.tie_word_embeddings:
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    else:
        assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight

def restore_tied(model):
    if model.config.tie_word_embeddings:
        model.tie_weights()
        assert_tied(model)

def patched_shard(original):
    def apply(model,*args,**kwargs):
        restore_tied(model)
        result=original(model,*args,**kwargs)
        assert_tied(model)
        return result
    return apply

def init_preserving_ties(self, model, is_train=True, is_wrapped=False):
    # Qwen tied models are already loaded on every rank by HFModelWrapper.
    # For this3B model, direct sharding avoids meta/assign reload losing aliases.
    module=model.model if is_wrapped else model
    if not module.config.tie_word_embeddings:
        result = _original_fsdp_init(self, model, is_train=is_train, is_wrapped=is_wrapped)
        assert_tied(result)
        return result
    assert_tied(module)
    assert all(not p.is_meta for p in module.parameters())
    precision=getattr(self.fsdp_config,'mixed_precision',None)
    param_dtype=fsdp_strategy.PrecisionType.to_dtype(precision.param_dtype) if precision is not None else torch.bfloat16
    reduce_dtype=fsdp_strategy.PrecisionType.to_dtype(precision.reduce_dtype) if precision is not None else torch.float32
    fsdp_kwargs={'mesh':self.device_mesh,
        'mp_policy':fsdp_strategy.MixedPrecisionPolicy(param_dtype=param_dtype,reduce_dtype=reduce_dtype,cast_forward_inputs=True),
        'offload_policy':fsdp_strategy.CPUOffloadPolicy(pin_memory=True) if self.fsdp_config.cpu_offload else None,
        'reshard_after_forward':self.fsdp_config.reshard_after_forward}
    fsdp_strategy.apply_fsdp2(module,fsdp_kwargs,self.fsdp_config)
    assert_tied(module)
    return module

class ReplayPolicyWorker(UUIDAffinityMixin, FSDPPolicyWorkerBase):
    def sarra_weight_probe(self):
        from skyrl.backends.skyrl_train.distributed.fsdp_utils import fsdp2_get_full_state_dict
        from training_adapter.weight_probe import probe
        assert_tied(self.model.model)
        state = fsdp2_get_full_state_dict(self.model.model)
        return probe(state) if state else None

    def init_model(self,*args,**kwargs):
        with patch.object(fsdp_strategy.FSDPStrategy,'_fsdp_init_model',init_preserving_ties):
            super().init_model(*args,**kwargs)
        assert_tied(self.model.model)

    def optim_step(self):
        value=super().optim_step()
        assert_tied(self.model.model)
        return value

class ReplayRefWorker(UUIDAffinityMixin, FSDPRefWorkerBase):
    def init_model(self,*args,**kwargs):
        with patch.object(fsdp_strategy.FSDPStrategy,'_fsdp_init_model',init_preserving_ties):
            super().init_model(*args,**kwargs)
        assert_tied(self.model.model)

PolicyWorker=ray.remote(num_gpus=1)(ReplayPolicyWorker)
RefWorker=ray.remote(num_gpus=1)(ReplayRefWorker)

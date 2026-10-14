"""Tie-preserving quota workers with a tied-aware live weight probe.

``replay_workers.py`` shards the already-loaded tied weights directly with FSDP2, asserts the tie after
initialization and after every optimizer step, and records a UUID-compatible NUMA affinity audit. The policy
worker here uses the tied-aware probe (tied_weight_probe.py), because a tied model does not expose a separate
``lm_head.weight`` in vLLM.
"""
import ray
from training_adapter.replay_workers import ReplayPolicyWorker, ReplayRefWorker, assert_tied
from training_adapter.tied_weight_probe import probe


class RerunPolicyWorker(ReplayPolicyWorker):
    def sarra_weight_probe(self):
        from skyrl.backends.skyrl_train.distributed.fsdp_utils import fsdp2_get_full_state_dict
        module = self.model.model
        assert_tied(module)
        state = fsdp2_get_full_state_dict(module)
        return probe(state, tied=bool(module.config.tie_word_embeddings), source='policy_fsdp2_full_state_dict') if state else None


class RerunRefWorker(ReplayRefWorker):
    pass


PolicyWorker = ray.remote(num_gpus=1)(RerunPolicyWorker)
RefWorker = ray.remote(num_gpus=1)(RerunRefWorker)

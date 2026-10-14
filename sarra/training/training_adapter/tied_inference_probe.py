"""vLLM worker extension exposing the tied-aware live weight probe over /collective_rpc."""
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import NewInferenceWorkerWrap
from training_adapter.tied_weight_probe import HEAD, probe


class TiedInferenceWorkerWrap(NewInferenceWorkerWrap):
    def sarra_weight_probe(self):
        model = self.model_runner.model
        params = dict(model.named_parameters())
        config = getattr(model, 'config', None)
        tied = bool(config.tie_word_embeddings) if config is not None and hasattr(config, 'tie_word_embeddings') else HEAD not in params
        return probe(params, tied=tied, source='vllm_named_parameters')

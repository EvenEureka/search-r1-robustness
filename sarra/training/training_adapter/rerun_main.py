"""Training entry point: SkyRL PPO experiment with the quota generator, the answer-CE trainer and tie-preserving workers (two GPUs)."""
import argparse
import json
import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl_gym.envs import register
from training_adapter.quota_generator import QuotaGenerator
from training_adapter.rerun_trainer import RerunQuotaTrainer, endpoint_of

WORKER_EXTENSION = 'training_adapter.tied_inference_probe.TiedInferenceWorkerWrap'


class RerunExperiment(BasePPOExp):
    def __init__(self, cfg, plan):
        super().__init__(cfg)
        self.boundary_plan = plan

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return QuotaGenerator(generator_cfg=cfg.generator, skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client, tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg), boundary_config=self.boundary_plan)

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return RerunQuotaTrainer(cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
            eval_dataset=eval_dataset, inference_engine_client=inference_engine_client, generator=generator, colocate_pg=colocate_pg)


@ray.remote(num_cpus=1)
def entrypoint(cfg, plan):
    register(id='search_bm25', entry_point='training_adapter.search_env:BM25SearchEnv')
    register(id='search_boundary', entry_point='training_adapter.boundary_env:BoundarySearchEnv')
    register(id='search_quota', entry_point='training_adapter.quota_generator:QuotaEnvironment')
    RerunExperiment(cfg, plan).run()


def check_config(cfg, plan):
    gpus = plan['gpus_per_run']
    assert cfg.generator.n_samples_per_prompt == plan['samples_per_prompt'] == 4
    assert cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node == gpus == 2
    assert cfg.generator.inference_engine.num_engines == gpus
    assert cfg.trainer.ckpt_interval == 0 and cfg.trainer.hf_save_interval >= 100000, 'final-only export is required'
    assert cfg.trainer.max_training_steps == plan['max_training_steps']
    assert cfg.trainer.train_batch_size == plan['train_batch_size'] and cfg.trainer.seed == plan['seed']
    assert cfg.trainer.dump_data_batch is True
    assert cfg.trainer.policy.fsdp_config.cpu_offload is False and cfg.trainer.ref.fsdp_config.cpu_offload is True
    extension = dict(getattr(cfg.generator.inference_engine, 'engine_init_kwargs', None) or {}).get('worker_extension_cls')
    assert extension == WORKER_EXTENSION, extension


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary-config', required=True)
    args, overrides = parser.parse_known_args()
    plan = json.load(open(args.boundary_config))
    endpoint_of(plan)
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    check_config(cfg, plan)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(entrypoint.remote(cfg, plan))

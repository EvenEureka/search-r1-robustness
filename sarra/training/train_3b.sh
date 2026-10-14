#!/bin/bash
# Train one 3B agent (one arm, one seed) with the answer-CE adapter on two GPUs.
#
# Requires SkyRL at commit 99824f01c68b718d93ee56a6dbf9edac044693f0 (https://github.com/NovaSky-AI/SkyRL)
# installed with `--extra fsdp`, plus rank-bm25==0.2.2, on Python 3.12. Exactly two GPUs must be visible.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 PLAN=plans/random_answer.json [MODEL=...] [DATA_DIR=...] [OUT=...] bash train_3b.sh [extra overrides]
#
#   PLAN      plan file (plans/<arm>.json); edit its "seed" for the other seeds (20261012, 20261013, 20261014)
#   MODEL     base model, default Qwen/Qwen2.5-3B (the runs used revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b;
#             pass a local snapshot path to pin it)
#   DATA_DIR  directory with train.parquet and validation.parquet (default data/grpo)
#   OUT       run directory (default: the plan's "output" field)
#   PYTHON    interpreter of the SkyRL environment (default python)
#
# The plan is copied to $OUT/design.json with absolute "output" and "support_labels" paths (the adapter reads them
# inside Ray workers) and passed to the adapter as --boundary-config. The overrides below are the base hyperparameters
# followed by the per-run overrides of the original launcher; for keys set twice, the later value is the one used.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
PLAN="${PLAN:?set PLAN to a plan file, for example plans/random_answer.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-data/grpo}"

plan_value() { "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PLAN" "$1"; }
SEED="$(plan_value seed)"
BATCH="$(plan_value train_batch_size)"
STEPS="$(plan_value max_training_steps)"
GPUS="$(plan_value gpus_per_run)"
OUT="${OUT:-$(plan_value output)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
if [ -e "$MODEL" ]; then MODEL="$(cd "$MODEL" && pwd)"; fi
RUN="$(basename "$OUT")"

"$PYTHON" - "$PLAN" "$OUT" <<'EOF'
import json, os, sys
plan = json.load(open(sys.argv[1]))
plan['output'] = sys.argv[2]
plan['support_labels'] = os.path.abspath(plan['support_labels'])
assert os.path.isfile(plan['support_labels']), plan['support_labels']
design = os.path.join(sys.argv[2], 'design.json')
assert not os.path.exists(design), 'run directory already used: ' + design
open(design, 'w').write(json.dumps(plan, indent=2) + '\n')
EOF

exec "$PYTHON" "$HERE/scripts/run_rerun_entry.py" --source "$HERE" --boundary-config "$OUT/design.json" \
  "data.train_data=['$DATA_DIR/train.parquet']" \
  "data.val_data=['$DATA_DIR/validation.parquet']" \
  data.dataloader.num_workers=0 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  "trainer.policy.model.path=$MODEL" \
  trainer.strategy=fsdp trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=2 trainer.placement.ref_num_gpus_per_node=2 \
  trainer.policy.fsdp_config.cpu_offload=false trainer.ref.fsdp_config.cpu_offload=true \
  trainer.flash_attn=false trainer.remove_microbatch_padding=false \
  trainer.gradient_checkpointing=true trainer.seed=20261003 \
  trainer.train_batch_size=16 trainer.policy_mini_batch_size=8 \
  trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 trainer.epochs=1 "trainer.max_training_steps=$STEPS" \
  trainer.update_epochs_per_batch=1 trainer.logger=console \
  trainer.enable_ray_gpu_monitor=false trainer.resume_mode=none \
  "trainer.run_name=$RUN" "trainer.ckpt_path=$OUT/checkpoints" \
  "trainer.export_path=$OUT/exports" "trainer.log_path=$OUT/infra" \
  trainer.ckpt_interval=5 trainer.hf_save_interval=5 trainer.max_ckpts_to_keep=2 \
  trainer.eval_before_train=false trainer.eval_interval=20 trainer.eval_batch_size=16 \
  trainer.dump_data_batch=true trainer.dump_eval_results=true \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.engine_init_kwargs.worker_extension_cls=training_adapter.tied_inference_probe.TiedInferenceWorkerWrap \
  generator.inference_engine.gpu_memory_utilization=0.35 \
  generator.inference_engine.max_num_seqs=16 \
  generator.max_input_length=8192 generator.sampling_params.max_generate_length=512 \
  generator.batched=false generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=4 generator.max_turns=5 \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 \
  'generator.sampling_params.stop=["</search>", "</answer>"]' \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.max_generate_length=512 \
  'generator.eval_sampling_params.stop=["</search>", "</answer>"]' \
  environment.env_class=search_bm25 environment.skyrl_gym.max_env_workers=8 \
  "trainer.seed=$SEED" \
  "trainer.train_batch_size=$BATCH" "trainer.policy_mini_batch_size=$BATCH" \
  trainer.ckpt_interval=0 trainer.hf_save_interval=100000 trainer.eval_interval=0 trainer.epochs=2 \
  trainer.max_ckpts_to_keep=1 environment.env_class=search_quota \
  "trainer.placement.policy_num_gpus_per_node=$GPUS" "trainer.placement.ref_num_gpus_per_node=$GPUS" \
  "generator.inference_engine.num_engines=$GPUS" \
  "$@"

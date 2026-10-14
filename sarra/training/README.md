# training: answer-CE adapter for the 3B agents

This directory holds the training code for the 15 trained 3B agents: five arms, each trained with seeds 20261012, 20261013 and 20261014.

## Method

The agents use the Search-R1 interaction protocol (https://github.com/PeterGriffinJin/Search-R1). The model reasons inside `<think>`, issues `<search> query </search>`, receives the top-3 documents inside `<information>`, and finishes with `<answer>`. The reward is exact match of the final answer. Training is GRPO through SkyRL's Search-R1 recipe (https://github.com/NovaSky-AI/SkyRL, commit `99824f01c68b718d93ee56a6dbf9edac044693f0`). Neither project is vendored here. The adapter in `training_adapter/` adds the following on top of that recipe:

- **Retrieval environment** (`search_env.py`). Each question carries its own pool of 10 HotpotQA distractor-setting documents, searched with BM25 (`rank_bm25`, top 3). A training attack is appended to the first retrieved document of the first search, using one of the two training templates in `scripts/evaluation.py` (`train_direct`, `train_role`). The target is `amber compass 17`. The test templates and the test target are never used in training.
- **Exposure allocation** (`quota_generator.py`). Each prompt gets 4 samples (repetitions 0 to 3). For every question, one autonomous first action (the anchor) is sampled. Eligible questions are ranked by a hash of seed, step and question id; reward, gold answer, arrival time and model score play no part. At most three questions per batch are selected, and repetitions 2 and 3 of each selected question are attacked. Attack templates alternate. A run consumes 300 attacks in total, 150 per training template, and the generator checks that every allocated payload was actually consumed. Under-supply carries forward; the clean arm has no exposure.
  - In the *independent* arms (`random_matched`, `random_answer`), every repetition samples its own first action. A question is eligible when both attacked repetitions issued a valid first search.
  - In the *paired* arms (`paired`, `paired_answer`), all four repetitions of a selected question share the anchor's first action, so the clean continuations (repetitions 0 and 1) and the attacked ones (2 and 3) start from the same state. Eligibility requires a valid first search by the anchor. The shared first action is masked out of the RL loss. Questions that are not selected sample independent first actions.
- **Answer-CE objective**. `answer_supervision.select_answer_examples` builds one target per selected question. The target is a gold-checked answer from a clean repetition (0 or 1). If no such answer exists, the gold answer is used, but only when every annotated support title appears in the attacked repetition's retrieved documents; otherwise the question is skipped. The input is the attacked continuation prefix of repetition 2 followed by `\n<answer>`, and the target is `answer</answer><eos>`. `quota_trainer.py` runs the RL backward pass and then a cross-entropy backward pass on the answer tokens before the same optimizer step. The cross-entropy loss has total weight 0.05, normalized over the answer tokens of the mini-batch. Arms without answer CE use weight 0.
- **Stopping rule** (`quota_trainer.py`, `rerun_trainer.py`, mode `loop_time`). Training stops at the first complete batch after 3,430 s of measured training-loop time, with a cap of 128 updates. The loop time is monotonic time from rollout start through the RL and CE updates. It excludes initialization, checkpoint I/O, evaluation and the post-update weight sync. Each step is recorded in `<run>/budget_audit/`.
- **Tied embeddings**. Qwen2.5-3B ties its input and output embedding, and training requires that tie to hold under FSDP2. `replay_workers.py` and `rerun_workers.py` shard the loaded tied weights directly and assert the tie after initialization and after every optimizer step. `tied_weight_probe.py` and `tied_inference_probe.py` then check, after every weight sync, that every vLLM engine holds the same BF16 values as the policy. They compare samples of the tied embedding, the final norm and every layer norm. `affinity_compat.py` keeps SkyRL's NUMA-affinity setup compatible with UUID-addressed GPUs.

`rerun_main.py` is the entry point and `scripts/run_rerun_entry.py` imports the adapter from this directory before launching. `study_generator.py`, `branch_generator.py`, `boundary_state.py` and `boundary_env.py` are imported or registered by these modules (shared helpers and the generic paired-branch environment). `scripts/evaluation.py` is the scorer used by the evaluation studies, byte-identical to theirs.

## Arms

| Plan | Arm | Attacks | Answer CE |
| --- | --- | --- | --- |
| `plans/clean.json` | Clean GRPO | none | no |
| `plans/random_matched.json` | Random | 300, independent | no |
| `plans/paired.json` | Boundary paired | 300, paired | no |
| `plans/paired_answer.json` | Paired + CE | 300, paired | weight 0.05 |
| `plans/random_answer.json` | Random + CE | 300, independent | weight 0.05 |

Each plan contains only the keys the adapter reads. The seed is set to 20261012. Set `"seed"` to 20261013 or 20261014 for the other two runs of an arm, and give each run its own `OUT`. `support_labels` points to a JSON object that maps each training question id to the titles of its supporting documents; it is used only for the answer-CE fallback.

## Data and model (not included)

- `DATA_DIR/train.parquet` and `DATA_DIR/validation.parquet` hold 1,024 and 64 HotpotQA distractor-setting questions (`hotpotqa/hotpot_qa`, revision `1908d6afbbead072334abe2965f91bd2709910ab`). The training questions were sampled from the train split with `random.Random(20261003)` and the validation questions from the validation split with `random.Random(20261004)`, after excluding the evaluation questions. The columns are `id`, `question`, `prompt` (one user message: the Search-R1 task prompt followed by the question), `env_class` (`search_bm25`), `reward_spec` (`{"style": "rule", "ground_truth": {"target": [answer]}}`) and `documents` (`[{"id", "title", "text"}]`, the 10 context paragraphs).
- `data/support_labels.json` holds the support titles from HotpotQA's supporting facts for the training questions.
- The base model is `Qwen/Qwen2.5-3B`, revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.

## Environment

Install SkyRL at commit `99824f01c68b718d93ee56a6dbf9edac044693f0` with `--extra fsdp` and add `rank-bm25==0.2.2`, on Python 3.12. Each run uses two GPUs: FSDP2 policy and reference model, two vLLM engines with tensor parallelism 1, colocated.

## One run

```
cd training
CUDA_VISIBLE_DEVICES=0,1 PLAN=plans/random_answer.json MODEL=/path/to/Qwen2.5-3B DATA_DIR=/path/to/data/grpo \
    OUT=runs/random_answer_s20261012 bash train_3b.sh
```

`train_3b.sh` writes `$OUT/design.json` (the plan with absolute paths) and starts `scripts/run_rerun_entry.py --source . --boundary-config $OUT/design.json` with the SkyRL overrides of the original runs. The main ones are: learning rate 1e-6, KL loss 0.001, token-level TIS clip 2.0, train batch 16 with mini-batch 16, 4 samples per prompt, at most 5 turns, 512 new tokens per turn, 8,192-token input, temperature 1.0, two epochs, a final-only HF export, and environment `search_quota`. Extra arguments are appended as further overrides.

Outputs in `$OUT`: `exports/global_step_<n>/policy` (final model), `boundary_audit/` (allocation and consumed exposures per step), `answer_audit/` (answer-CE examples and weights per step), `budget_audit/` and `budget_endpoint.json` (stopping rule), `weight_sync_audit/` (policy-to-vLLM checks), `affinity_audit/`.

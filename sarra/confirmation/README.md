# confirmation (E3): gate or suppression, 3B confirmation

This study tests once, on 1,000 fresh HotpotQA questions, whether answer CE (Random + answer CE vs Random injection) gates instruction-shaped payloads rather than suppressing injected content.

The analysis has four primary contrasts, P_a to P_d, with Holm correction over the four and a question bootstrap (exports fixed, seed 20261101, 10,000 draws). `protocol_public.json` gives the estimands, decision rules, secondary analyses and limitations. Installed name: `studies/confirmation_v2` (see `../README.md`).

## Files

| File | Role |
| --- | --- |
| `engine/engine_e3.py` | Builds the payloads for the ten conditions and matches their lengths. The conditions are clean, test_override, test_authority, neutral, quoted_target, benign_instruction, fluent_irrelevant, override_gold, override_plaus and decl_plaus. Each payload is appended to retrieved document 1 of the first search. The file also contains the shared first action and the episode loop. |
| `engine/common.py` | Search-R1 task prompt (`base_prompt`) and shared helpers. |
| `scripts/evaluation.py` | Frozen scorer (sha256 `c5f8089d...`). |
| `scripts/defenses.py` | Prompt defenses; this study uses sandwich and spotlight. |
| `scripts/make_plausible_e3.py` | The w(q) rule. Donor pools come from the GRPO training pool. |
| `scripts/evaluate_e3.py` | One shard of one export, all cells of that export. |
| `scripts/run_e3_job.py` | One export on two GPUs (four shards, two per GPU), then validation. |
| `scripts/e3_utils.py`, `scripts/confirmation_utils.py` | Formatting audit, validation and token F1. |
| `scripts/judge_prompts.py`, `scripts/build_judge_items.py`, `scripts/run_judge.py` | LLM judge (secondary criterion; prompt E of `other_wrong`). |
| `scripts/analyze_e3.py` | Frozen confirmatory analysis. |
| `scripts/test_e3.py`, `scripts/preflight_lengths_e3.py`, `scripts/test_analysis_e3.py` | CPU checks. |
| `scripts/prepare_e3.py` | Builds (`--draft`) and freezes (`--freeze`) the protocol. |
| `scripts/open_fresh_test.py` | Opens the sealed split once, builds w(q) and runs the leak check. |

## Inputs (not included)

- The 15 trained exports listed under `models` in the protocol.
- The sealed split `studies/fresh_test_v2/sealed1000.jsonl`.
- The GRPO training pool `data/grpo/train.parquet` (donor pools for w(q)).
- The smoke split `$S/data/smoke8_plaus.jsonl`.
- The development traces of the earlier studies, read only by `test_e3.py` and `preflight_lengths_e3.py`.

## Commands

Run from `$SARRA_WORKSPACE`, with `export SARRA_VERIFY_FROZEN=0 S=studies/confirmation_v2`.

```
cp $S/protocol.json $S/protocol_draft.json        # the CPU checks read the draft

# 1. CPU checks
python -B $S/source/scripts/test_e3.py $S                         # -> $S/cpu_tests.json
python -B $S/source/scripts/preflight_lengths_e3.py $S            # -> $S/length_preflight.json
python -B $S/source/scripts/test_analysis_e3.py $S <scratch_dir>  # synthetic traces -> $S/analysis_tests.json

# 2. Open the test split once and build w(q)
python -B $S/source/scripts/open_fresh_test.py $S                 # -> $S/data/fresh_test_v2_plaus.jsonl, leak_check.json, OPENED.json

# 3. GPU evaluation, one export at a time (index i in the protocol's models list; primary order in stages.primary)
E3_DIRECTORY=$S STAGE=smoke MODEL_INDEX=1 CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_e3_job.py    # -> SMOKE_VALIDATED
E3_DIRECTORY=$S STAGE=primary MODEL_INDEX=$i CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_e3_job.py
#    single shard (run_e3_job.py starts shards 0-3, two per GPU, then validates them):
CUDA_VISIBLE_DEVICES=0 python -B $S/source/scripts/evaluate_e3.py $S --stage primary --model-index $i --shard 0 \
    --samples $S/data/fresh_test_v2_plaus.jsonl --samples-sha256 <sha256 of that file> --out-root $S/runs/primary/s<seed>_<arm>

# 4. LLM judge on two GPUs (vLLM 0.24.0)
python -B $S/source/scripts/build_judge_items.py $S $S/judge
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$JUDGE_MODEL" --served-model-name qwen3-32b --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 4096 --enforce-eager --dtype bfloat16 --seed 20261016 &
E3_DIRECTORY=$S JUDGE_URL=http://127.0.0.1:8000 python -B $S/source/scripts/run_judge.py

# 5. Analysis (CPU)
python -B $S/source/scripts/analyze_e3.py $S            # add --allow-missing-judge if step 4 was not run
```

## Outputs

- `runs/<stage>/s<seed>_<arm>/shard_*/traces.jsonl`, with `result.json` and the marker `E3_RUN_VALIDATED`.
- `judge/items.jsonl`, `judge/outputs.jsonl` and `judge/sanity.json`.
- `analysis/report.json`, `analysis/summary.txt` and `analysis/decisions.json`. The decisions file holds the four primary decisions.

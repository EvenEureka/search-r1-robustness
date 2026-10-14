# benign_controls (E2): is the override's non-target loss specific to the attack?

On clean-correct pairs, the override attack produces more non-target wrong answers than a length-matched neutral addition does. This study adds two controls of the same token length:

- a benign instruction with the override's header and literal `<answer>` tags, which restates the task and names no answer;
- a fluent irrelevant sentence.

It then measures, on development256, the excess of the override over each control. The run covers 15 exports and 7 conditions. Predictions P1 to P3 and the frozen rule that chose the control for E3 are in `protocol_public.json`. Installed name: `studies/benign_controls_v1`.

## Files

| File | Role |
| --- | --- |
| `engine/engine_v2.py` | Frozen five-condition engine: clean, test_override, test_authority, neutral, quoted_target (byte-identical). |
| `engine/engine_controls.py` | Adds `benign_instruction` and `fluent_irrelevant`, length-matched by the unchanged procedure. The change is shown in `engine_controls.diff`. |
| `scripts/evaluation.py` | Frozen scorer. |
| `scripts/evaluate_controls.py`, `scripts/run_controls_job.py` | One shard of one export, and one export on two GPUs (one shard per GPU) with validation. |
| `scripts/controls_utils.py` | Validation and leak checks. |
| `scripts/analyze_controls.py` | Frozen within-model analysis (question bootstrap, seed 20261101, 10,000 draws). |
| `scripts/test_controls.py`, `scripts/preflight_lengths.py` | CPU checks with a scripted fake generator and a real tokenizer. |
| `scripts/prepare_controls.py` | Builds and freezes the protocol. |

## Inputs (not included)

- The 15 trained exports.
- `studies/extension_splits_v1/development256.jsonl`.
- The smoke split `$S/data/smoke8.jsonl`.

## Commands

Run from `$SARRA_WORKSPACE`, with `export SARRA_VERIFY_FROZEN=0 S=studies/benign_controls_v1`. `<tokenizer_dir>` is any of the trained exports (Qwen2.5 tokenizer).

```
# 1. CPU checks
python -B $S/source/scripts/test_controls.py $S <tokenizer_dir> $S/data/smoke8.jsonl studies/extension_splits_v1/development256.jsonl   # -> $S/cpu_tests.json
python -B $S/source/scripts/preflight_lengths.py $S <tokenizer_dir> $S/protocol.json

# 2. GPU evaluation: smoke, then the 15 exports (i = 0..14), one export at a time
CONTROLS_DIRECTORY=$S STAGE=smoke MODEL_INDEX=1 CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_controls_job.py
CONTROLS_DIRECTORY=$S STAGE=primary MODEL_INDEX=$i CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_controls_job.py
#    single shard:
CUDA_VISIBLE_DEVICES=0 python -B $S/source/scripts/evaluate_controls.py $S --stage primary --model-index $i --shard 0

# 3. Analysis (CPU)
python -B $S/source/scripts/analyze_controls.py $S
```

## Outputs

- `runs/<stage>/s<seed>_<arm>/shard_*/traces.jsonl`, with `result.json` and the marker `CONTROLS_RUN_VALIDATED`.
- `analysis/primary.json` and `analysis/primary.txt`.
- `analysis/predictions_primary.json`, which also records the control selected for E3.

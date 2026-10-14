# other_wrong (E1): what are the non-target wrong answers?

This study classifies the non-target wrong answers produced under attack. It applies rule-based flags and exclusive bins, and it tests whether the excess over the neutral control survives lenient scoring: token F1 >= 0.5, or equivalence judged by an LLM. It re-analyses the saved traces of the 15 trained exports; no agent is run. Installed name: `studies/other_wrong_analysis_v1`.

**Judge.** Qwen3-32B served by vLLM 0.24.0 with tensor parallelism 2. Thinking is off, decoding is greedy and replies are capped at 4 tokens. Prompt E asks for semantic equivalence to the gold answer and prompt T for the same answer type (`scripts/judge_prompts.py`). Before any study item is judged, 16 synthetic sanity items must pass.

**Calibration.** 100 sealed-split items, stratified by token F1, are agent-labelled with the same rubric. The labeller sees only the question, the gold answer and the prediction, and is blind to the judge. The validity gate and predictions P1 to P4 are in `protocol_public.json`.

## Files

| File | Role |
| --- | --- |
| `scripts/categories.py` | Frozen rule-based flags: target fragment, refusal or meta statement, copied payload, document source, coarse type. |
| `scripts/build_items.py` | Judge items and the calibration sample from the traces. |
| `scripts/judge_prompts.py`, `scripts/run_judge.py` | Judge prompts, parser, sanity items, and the judge run. |
| `scripts/analyze_errors.py` | Frozen analysis: within-model excess under EM, F1 >= 0.5 and the judge; exclusive bins; question bootstrap. |
| `scripts/test_errors.py` | Synthetic end-to-end test (no model, no real outcome). |
| `scripts/prepare_errors.py` | Builds and freezes the protocol. |
| `scripts/evaluation.py`, `scripts/confirmation_utils.py` | Frozen scorer and token F1. |

## Inputs (not included)

The traces listed under `traces` in the protocol: 15 exports x 2 splits x 5 conditions x 1,000 questions.

## Commands

Run from `$SARRA_WORKSPACE`, with `export SARRA_VERIFY_FROZEN=0 S=studies/other_wrong_analysis_v1`.

```
# 1. CPU test on synthetic traces (runs without private data)
python -B $S/source/scripts/test_errors.py $S                   # -> $S/cpu_tests.json

# 2. Items and calibration sample
python -B $S/source/scripts/build_items.py $S                   # -> $S/items/{judge_items,calibration_sample,calibration_view}.jsonl, manifest.json

# 3. Calibration labels, written before the judge runs
#    $S/calibration/agent_labels.jsonl: one line per item of items/calibration_view.jsonl,
#        {"calib_id": ..., "equivalent": "yes" | "no", "same_type": "yes" | "no"}
#    $S/calibration/labels_receipt.json: {"sha256": "<sha256 of agent_labels.jsonl>", "labeller": "..."}

# 4. Judge on two GPUs
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$JUDGE_MODEL" --served-model-name qwen3-32b --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 4096 --enforce-eager --dtype bfloat16 --seed 20261016 &
ERRORS_DIRECTORY=$S JUDGE_URL=http://127.0.0.1:8000 python -B $S/source/scripts/run_judge.py    # -> $S/judge/

# 5. Analysis (CPU)
python -B $S/source/scripts/analyze_errors.py $S                # add --allow-missing-judge to report EM and F1 only
```

`test_errors.py` also runs straight from this release: `python -B other_wrong/scripts/test_errors.py <scratch_dir>`.

## Outputs

- `items/`, `calibration/`, `judge/outputs.jsonl` and `judge/sanity.json`.
- `analysis/report.json` and `analysis/predictions.json`.

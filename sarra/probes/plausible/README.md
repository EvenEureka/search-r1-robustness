# probes/plausible (FT2-C): instruction form vs declarative form of a plausible wrong answer

This study asks whether a plausible wrong answer w(q) is adopted more when it arrives as an override instruction than when it is stated as a fact. Conditions on development256:

- `clean`;
- `override_plaus`: w(q) in the override template, unpadded;
- `decl_plaus`: "Reference note: the correct answer to this question is w(q).", padded to the `override_plaus` observation length.

The analysis also reads the FT2-B traces (`override_gold`, `test_override`) of the same exports. Installed name: `studies/suppression_signature_v1c`.

## Files

| File | Role |
| --- | --- |
| `scripts/make_plausible.py` | Pre-registered w(q) rule: yes/no flip, year or number swap, document title from the question's own pool, or a fallback swap. In this version the donor pools come from the evaluated split. The E3 version in `../../confirmation/scripts/make_plausible_e3.py` draws them from the training pool. |
| `engine/engine_probe.py` | FT2-C payloads and padding. The change from FT2-B is shown in `engine_probe_v1c.diff`. |
| `scripts/evaluate_probe.py`, `scripts/run_probe_job.py` | One shard of one export, and all 15 exports serially on two GPUs followed by the analysis. |
| `scripts/analyze_plaus.py` | FT2-C analysis; development evidence. |
| `scripts/test_plaus.py` | CPU checks. |

FT2-A (`ft2a_cpu.py`) is identical in both probe studies and is released once, in `../gold/`.

## Inputs (not included)

- The 15 trained exports.
- `studies/extension_splits_v1/development256.jsonl`.
- The E2 and FT2-B traces.

## Commands

Run from `$SARRA_WORKSPACE`, with `export SARRA_VERIFY_FROZEN=0 S=studies/suppression_signature_v1c`.

```
python -B $S/source/scripts/make_plausible.py studies/extension_splits_v1/development256.jsonl $S/data/development256_plaus.jsonl
cp $S/protocol.json $S/protocol_draft.json                      # test_plaus.py reads the draft
python -B $S/source/scripts/test_plaus.py $S                    # CPU -> $S/cpu_tests.json
PROBE_DIRECTORY=$S CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_probe_job.py   # 15 exports, then the analysis
#    single shard:
CUDA_VISIBLE_DEVICES=0 python -B $S/source/scripts/evaluate_probe.py $S --stage primary --model-index $i --shard 0
python -B $S/source/scripts/analyze_plaus.py $S                 # re-run the analysis -> $S/analysis/
```

## Outputs

- `data/development256_plaus.jsonl`.
- `runs/primary/s<seed>_<arm>/shard_*/traces.jsonl` and `execution.json`.
- `analysis/`.

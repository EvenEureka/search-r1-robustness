# probes/gold (FT2-B, FT2-A): is correct injected content suppressed?

**FT2-B.** Does answer CE suppress injected content in general, or only instruction-shaped wrong content? On development256, every export is evaluated under seven conditions:

- `clean`;
- `test_override` and `decl_target`, which carry the nonsense target;
- `override_gold`, `decl_gold`, `benign_gold` and `quoted_gold`, which carry the first gold answer in override, declarative, benign-instruction and quoted form.

All payloads are appended to retrieved document 1 of the first search.

**FT2-A** (`ft2a_cpu.py`, CPU, saved traces). On clean inputs, do answer-CE agents use retrieved evidence less? It measures EM split by whether a gold alias appears in the shown documents, and the rate at which answers are copied from the documents.

Installed name: `studies/suppression_signature_v1`.

## Files

| File | Role |
| --- | --- |
| `engine/engine_probe.py` | Probe payloads and length matching. The change from `engine_controls.py` is shown in `engine_probe.diff`. |
| `scripts/evaluate_probe.py`, `scripts/run_probe_job.py` | One shard of one export, and all 15 exports serially on two GPUs followed by the analysis. |
| `scripts/analyze_probe.py` | FT2-B analysis; development evidence. |
| `scripts/test_probe.py` | CPU checks against saved E2 traces. |
| `ft2a_cpu.py` | FT2-A. It reads saved clean traces and imports `normalize` from the installed `killcheck` copy (`studies/multisearch_killcheck_v1/source/msq.py`). |

## Inputs (not included)

- The 15 trained exports.
- `studies/extension_splits_v1/development256.jsonl`.
- The E2 traces `studies/benign_controls_v1/runs/primary`, used by `test_probe.py`.
- For FT2-A, the clean traces of the earlier confirmation run, `studies/confirmation_fresh_v1/runs`.

## Commands

Run from `$SARRA_WORKSPACE`, with `export SARRA_VERIFY_FROZEN=0 S=studies/suppression_signature_v1`.

```
cp $S/protocol.json $S/protocol_draft.json                      # test_probe.py reads the draft
python -B $S/source/scripts/test_probe.py $S                    # CPU -> $S/cpu_tests.json
PROBE_DIRECTORY=$S CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/scripts/run_probe_job.py   # 15 exports, then the analysis
#    single shard:
CUDA_VISIBLE_DEVICES=0 python -B $S/source/scripts/evaluate_probe.py $S --stage primary --model-index $i --shard 0
python -B $S/source/scripts/analyze_probe.py $S                 # re-run the analysis -> $S/analysis/
python -B $S/source/ft2a_cpu.py                                 # FT2-A -> $S/analysis/ft2a_cpu.json
```

## Outputs

- `runs/primary/s<seed>_<arm>/shard_*/traces.jsonl` and `execution.json`.
- `analysis/`.

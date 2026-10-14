# SARRA code release

This release contains the code for the experiments on prompt-injection robustness of RL-trained search agents. Each directory except `training/` is the frozen `source/` tree of one study, without its scheduler scripts. It comes with a redacted copy of that study's frozen protocol (`protocol_public.json`) and a README with the commands in run order.

## Map

| Directory | Study | Question |
| --- | --- | --- |
| `confirmation/` | E3 | Confirmatory test on 1,000 fresh HotpotQA questions. Does answer CE (Random + answer CE vs Random injection) gate instruction-shaped payloads rather than suppress injected content? Four Holm-corrected primary contrasts, P_a to P_d. |
| `benign_controls/` | E2 | Control ladder: neutral filler, then a benign instruction and a fluent irrelevant sentence of the same length, then the attack. Is the excess of non-target wrong answers under the override specific to the attack? |
| `other_wrong/` | E1 | What are the non-target wrong answers under attack, and does their excess survive lenient scoring (token F1, LLM judge)? Includes the judge prompts and the judge calibration. |
| `probes/gold/` | FT2-B, FT2-A | Is a correct answer taken up when it is delivered in override, declarative, benign or quoted form? FT2-A (CPU): do answer-CE agents use retrieved evidence less on clean inputs? |
| `probes/plausible/` | FT2-C | Is a plausible wrong answer adopted more in override form than in declarative form? |
| `verify_7b/` | B1 | Released Search-R1 7B on MuSiQue: content channel, instruction channel and recovery. Five conditions, three variants: native, verification prompt, forced extra search. |
| `killcheck/` | FT1 | Do the released Search-R1 3B and 7B search more than once on open-corpus multi-hop questions (MuSiQue-pool)? |
| `training/` | | Answer-CE training adapter for the 15 trained 3B agents (5 arms x 3 seeds) on SkyRL's Search-R1 recipe. |

Where each study appears in the manuscript (`docs/sarra.pdf`): `confirmation/` in Sections 5.1 to 5.3 and Appendix C; `probes/` as the development results of Section 5.2 and Appendix D; `benign_controls/` in Section 5.3; `other_wrong/` only through the judge calibration of Section 4.3; `verify_7b/` in Section 5.4 and Appendix E; `killcheck/` in Section 4.5; `training/` in Section 4.1 and Appendix B. The study labels (E1-E3, FT1, FT2, B1) are internal names used in file names and protocols.

## Files to read first

- Scorer: `confirmation/scripts/evaluation.py`. `score_record` reads only the last assistant generation, never the retrieved text. `ATTACKS` holds the two training templates (`train_direct`, `train_role`) and the two test templates (`test_override`, `test_authority`). Every study has a byte-identical copy.
- Payloads and length matching: `confirmation/engine/engine_e3.py`. `interventions()` builds all ten conditions on the first retrieved document of the first search, and `episode()` runs the continuation. The earlier engines are `benign_controls/engine/engine_v2.py` (five frozen conditions), `benign_controls/engine/engine_controls.py` and `probes/*/engine/engine_probe.py`.
- Plausible wrong answer w(q): `confirmation/scripts/make_plausible_e3.py`. The FT2-C version is `probes/plausible/scripts/make_plausible.py` and the MuSiQue version is in `verify_7b/b1.py`.
- Answer-CE objective: `training/training_adapter/answer_supervision.py` builds the targets, and `training/training_adapter/quota_trainer.py` runs the CE backward pass before the shared optimizer step. Exposure allocation is in `quota_generator.py`.
- Confirmatory analysis: `confirmation/scripts/analyze_e3.py`.
- Prompt defenses (sandwich, spotlight): `confirmation/scripts/defenses.py`.

## Environment

- Evaluation and analysis: Python 3.10 or newer (the runs used 3.11), numpy (2.3; 2.2 for the analyses), `rank_bm25==0.2.2`, torch 2.11, transformers 5.13 and pyarrow. Model inference needs one CUDA GPU per worker process. Everything else runs on CPU.
- LLM judge: vLLM 0.24.0 serving Qwen3-32B with tensor parallelism 2.
- Training: see `training/README.md` (SkyRL at a pinned commit, Python 3.12).

## Environment variables

| Variable | Meaning |
| --- | --- |
| `SARRA_WORKSPACE` | Workspace root (default `.`). Protocol paths are relative to it, and scripts look up `studies/<study>/...` below it. |
| `SARRA_VERIFY_FROZEN` | `1` (default) re-checks, as in the original runs, the frozen protocol hash, every hashed file of the protocol, the model-file hashes and the judge spec hash. Set it to `0` for these public copies, because their protocol paths are redacted and some files were edited (see `MANIFEST.json`). All other assertions stay active: protocol status, scorer hash, re-scoring of every record, and data hashes. |
| `MUSIQUE_DIR` | Directory with `musique_ans_v1.0_train.jsonl` and `musique_ans_v1.0_dev.jsonl` (Hugging Face dataset `dgslibisey/MuSiQue`, snapshot `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`). |
| `E5_DIR` | Directory that contains one snapshot directory of `intfloat/e5-base-v2`, for example the hub folder `models--intfloat--e5-base-v2/snapshots`. The runs used revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. |
| `JUDGE_MODEL` | Judge model id or local path: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`. It is recorded in the judge spec. |
| `CUDA_VISIBLE_DEVICES` | GPU selection. The two-GPU runners use the first two listed devices. |
| `RUN_ID` | Optional label written into run metadata. |

## Installed layout

The scripts find their inputs through the protocol and through their own location (`<study>/source/...`). To run a study, install it under the workspace and run every command from `$SARRA_WORKSPACE`:

```
$SARRA_WORKSPACE/
  studies/<study>/protocol.json     copy of <directory>/protocol_public.json
  studies/<study>/source/           copy of <directory>; use a copy, not a symlink, because scripts resolve their own path
  data/hub/                         Hugging Face cache for the released Search-R1 checkpoints (see verify_7b/, killcheck/)
```

| Directory | `<study>` |
| --- | --- |
| `confirmation/` | `confirmation_v2` |
| `benign_controls/` | `benign_controls_v1` |
| `other_wrong/` | `other_wrong_analysis_v1` |
| `probes/gold/` | `suppression_signature_v1` |
| `probes/plausible/` | `suppression_signature_v1c` |
| `verify_7b/` | `verify_7b_musique_v1` |
| `killcheck/` | `multisearch_killcheck_v1` |

For example:

```
export SARRA_WORKSPACE=$PWD SARRA_VERIFY_FROZEN=0
mkdir -p studies/multisearch_killcheck_v1
cp -r /path/to/sarra/killcheck studies/multisearch_killcheck_v1/source
cp studies/multisearch_killcheck_v1/source/protocol_public.json studies/multisearch_killcheck_v1/protocol.json
```

Other studies' paths that appear in the protocols, such as `studies/extension_splits_v1/development256.jsonl`, `studies/fresh_test_v2/sealed1000.jsonl` or `results/training/<run>/exports/...`, name inputs that are not part of this release.

## What is not included

The release does not include the 15 trained 3B checkpoints, the evaluation traces, the HotpotQA split files (training pool, development split, test splits) or the support-title labels. The 3B studies (`confirmation`, `benign_controls`, `other_wrong`, `probes`) and `training` therefore document the procedure and run only where those assets exist. The 7B study (`verify_7b`) and the kill check (`killcheck`) need only public assets: the released Search-R1 checkpoints, MuSiQue and e5-base-v2.

These checks run from this release without private data:

```
python -B killcheck/test_msq.py <out>/cpu_tests.json             # needs MUSIQUE_DIR
python -B verify_7b/test_b1.py <workspace>/studies/verify_7b_musique_v1   # needs MUSIQUE_DIR and the installed killcheck
python -B confirmation/scripts/test_analysis_e3.py <study_dir> <scratch>  # synthetic traces; <study_dir> holds protocol_draft.json
python -B other_wrong/scripts/test_errors.py <study_dir>                  # synthetic traces
```

## Provenance

`MANIFEST.json` lists every released file. For each file it gives the source path in the original study, whether the file is byte-identical to the frozen source, and the hashes of both. Edits to the other files are limited to the following:

- private paths replaced by the environment variables above, and the internal engine directory renamed `engine`;
- scheduler and host checks removed;
- the `SARRA_VERIFY_FROZEN` gate added;
- a few docstrings reworded.

The cluster job scripts are not part of the release. Each study README gives the plain commands instead. In `protocol_public.json`, paths are made relative and the scheduler fields (queue, hosts, wall time, interpreter, job ids) are removed. `public_redaction.original_sha256` is the hash of the frozen protocol.

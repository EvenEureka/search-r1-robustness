# killcheck (FT1): do released Search-R1 agents search more than once?

This study runs the released Search-R1 3B and 7B checkpoints on 300 MuSiQue dev questions (S300) over an open multi-hop corpus. It records how often they search at least twice (FT1-C) and how often they adopt the override target when it is appended to the first retrieved document (FT1-D, exploratory). Two CPU add-ons complete it:

- FT1-A (`ft1a_cpu.py`): is one BM25 search with the question as the query enough to retrieve every supporting paragraph?
- FT1-B (`ft1b_cpu.py`, saved HotpotQA traces): is the gold answer already in the first top 3?

The decision thresholds are in `protocol_public.json`. Installed name: `studies/multisearch_killcheck_v1`.

## Setup

**Models.**

- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`, revision `d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7`;
- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2`, revision `948503c0691ec29525f20bd7d5dbc51ed5aa7e70`.

**Corpus.** MuSiQue-pool, the 101,962 unique (title, paragraph) pairs of MuSiQue answerable v1.0 train and dev, rebuilt in memory.

**Questions.** S300: answerable dev questions sorted by id, then sampled with `random.Random(20261013)`.

**Passes.** e5/clean, e5/override and bm25/clean. The first assistant turn is shared across passes. Top 3, up to 8 searches, greedy.

## Files

| File | Role |
| --- | --- |
| `msq.py` | Data, corpus, S300, BM25 (k1 0.9, b 0.4), e5-base-v2, Search-R1 prompt and scorer. Its text-protocol helpers are copies of the frozen scorer and prompt. |
| `worker.py` | One checkpoint on one GPU, all passes. |
| `run_job.py` | 3B on the first GPU and 7B on the second: smoke run, main run, then the analysis. |
| `analyze.py` | FT1-C/D analysis. |
| `test_msq.py` | CPU checks. It compares `msq.py`'s copies with the released scorer and prompt (`../confirmation/scripts/evaluation.py`, `../confirmation/engine/common.py`). |
| `ft1a_cpu.py`, `ft1b_cpu.py` | FT1-A and FT1-B. |

## Assets (public)

Run from `$SARRA_WORKSPACE`:

```
export SARRA_WORKSPACE=$PWD SARRA_VERIFY_FROZEN=0 MUSIQUE_DIR=<musique snapshot dir> E5_DIR=<e5-base-v2 snapshots dir>
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
    --revision d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7 --cache-dir data/hub
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2 \
    --revision 948503c0691ec29525f20bd7d5dbc51ed5aa7e70 --cache-dir data/hub
```

## Commands

With `S=studies/multisearch_killcheck_v1`:

```
python -B $S/source/test_msq.py $S/cpu_tests.json    # CPU checks
python -B $S/source/ft1a_cpu.py                      # FT1-A -> $S/analysis/ft1a_cpu.json
CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/run_job.py   # smoke (8 questions) and main run for both models, then analyze.py
#    one model alone:
CUDA_VISIBLE_DEVICES=0 python -B $S/source/worker.py searchr1_3b            # add --limit 8 for the smoke run
python -B $S/source/analyze.py                       # -> $S/analysis/summary.json and summary.txt
python -B $S/source/ft1b_cpu.py                      # FT1-B; needs saved HotpotQA development traces (not included)
```

`test_msq.py` also runs straight from this release: `python -B killcheck/test_msq.py <scratch>/cpu_tests.json`.

## Outputs

- `runs/<stage>/<model>/<retriever>_<condition>.jsonl` and `execution.json`.
- `analysis/`.

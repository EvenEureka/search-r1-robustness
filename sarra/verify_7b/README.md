# verify_7b (B1): content channel and instruction channel in the released Search-R1 7B

This inference-only study asks three questions about the released Search-R1 7B on MuSiQue:

- Does it adopt a wrong answer stated as a fact (content channel)?
- Does instruction form add to adoption (instruction channel)?
- Does it use correct injected content?

Predictions PB1-1 to PB1-3 answer them and are in `protocol_public.json`. The code as run also evaluated two inference-time variants (a verification prompt and a forced extra search) with three further predictions; the manuscript uses only the native variant. Installed name: `studies/verify_7b_musique_v1`.

## Setup

**Model.** `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2`, revision `948503c0691ec29525f20bd7d5dbc51ed5aa7e70`. It uses the native prompt and chat template, greedy batched decoding, up to 8 searches, 512 new tokens per turn and an 8,192-token context.

**Corpus and retrieval.** MuSiQue-pool: the 101,962 unique paragraphs of MuSiQue answerable v1.0 train and dev. Retrieval is e5-base-v2, top 3.

**Questions.** D1000 is S300 (the questions of the search-count study) plus 700 further answerable dev questions. The smoke run uses the first 8 answerable train questions.

**Conditions.** Each payload is appended to retrieved document 1 of the first search:

- `clean`;
- `override`: target "violet telescope 42";
- `decl_plaus`: w(q) stated as fact, padded to the `override_plaus` length;
- `override_gold`: the gold answer in override form;
- `override_plaus`: w(q) in override form.

**Variants.**

- `native`;
- `verify`: one verification sentence added to the prompt;
- `forced`: the first attempted answer is intercepted and one more search is forced.

## Files

| File | Role |
| --- | --- |
| `msq.py` | MuSiQue loading, corpus, S300, BM25 and e5 retrievers, Search-R1 prompt and scorer. Same file as `../search_counts/msq.py`. |
| `b1.py` | Question sets, w(q) for MuSiQue, payloads, variants and record metrics. |
| `b1_loop.py` | Batched episode loop; the model and retriever are passed in. |
| `worker_b1.py` | One GPU, one shard, all 15 passes. |
| `run_b1_job.py` | Smoke run and then the main run on two GPUs, then the analysis. |
| `analyze_b1.py` | Frozen analysis. |
| `test_b1.py` | CPU tests with a scripted fake model. |
| `prepare_b1.py` | Freezes the protocol. |

## Assets (public)

Run from `$SARRA_WORKSPACE`:

```
export SARRA_WORKSPACE=$PWD SARRA_VERIFY_FROZEN=0 MUSIQUE_DIR=<musique snapshot dir> E5_DIR=<e5-base-v2 snapshots dir>
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2 \
    --revision 948503c0691ec29525f20bd7d5dbc51ed5aa7e70 --cache-dir data/hub
```

The protocol's model path is `data/hub/models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2/snapshots/948503c0691ec29525f20bd7d5dbc51ed5aa7e70`. The worker checks the hashes of the model's interface files (config, tokenizer, index) before loading it. `test_b1.py` also expects the kill check to be installed at `studies/multisearch_killcheck_v1/source/`.

## Commands

With `S=studies/verify_7b_musique_v1`:

```
python -B $S/source/test_b1.py $S                     # CPU -> $S/cpu_tests.json
CUDA_VISIBLE_DEVICES=0,1 python -B $S/source/run_b1_job.py   # smoke (8 train questions), main (two shards, one per GPU), analysis
#    single shard:
CUDA_VISIBLE_DEVICES=0 python -B $S/source/worker_b1.py --stage main --shard 0 --shards 2
python -B $S/source/analyze_b1.py                     # re-run the analysis (argument: main or smoke; default main)
```

## Outputs

- `runs/<stage>/shard_<k>/<variant>_<condition>.jsonl` and `execution.json`.
- `analysis/report.json` and `analysis/summary.txt`.

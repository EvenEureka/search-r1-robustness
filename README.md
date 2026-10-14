# Answer Supervision Gates Injected Instructions but Barely Touches Injected Misinformation in a Single-Search Agent

Yiwen Lu, University of Notre Dame. Manuscript, 2026.

**[Paper (PDF)](docs/sarra.pdf)** · [Project page](https://eveneureka.github.io/search-r1-robustness/) · [Code](sarra/) · [Result files](results/)

## Question

Defenses against injected text in retrieved documents are usually judged by attack success and clean accuracy.
Those two numbers cannot separate an agent that has learned to ignore instructions inside documents (a *gate*) from
one that checks what documents claim (*verification*), or from one that discounts any inserted text (*suppression*).
The three come apart on two inputs that standard attack suites leave out: a wrong answer stated as a plain fact, and a
correct answer delivered as an instruction. We ask which of them training against injection teaches a
reinforcement-learned search agent.

![Payload adoption by form and content](docs/assets/gate.png)

## Findings

1. **Answer supervision acts as a gate on instruction-shaped payloads** (confirmatory). We trained Search-R1-style
   Qwen2.5-3B agents with GRPO under retrieval injection, with and without a cross-entropy loss on the correct answer
   (answer CE), three seeds each. On 1,000 unseen HotpotQA questions, answer CE cut adoption of the attack target from
   32.0% to 1.6% of the answers the agents otherwise got right, and adoption of a plausible wrong answer delivered as
   an instruction from 68.3% to 33.5%. The same wrong answer stated as a fact went from 20.7% to 17.7%, a small
   partial transfer (−3.0 points, 95% CI [−5.8, −0.1]).
2. **It keeps most correct inserted content and removes most of the attack's other damage** (confirmatory). A
   correct answer delivered as an instruction was used on 92.1% vs 88.9% of the questions the agents otherwise
   missed (−3.2 points [−5.0, −1.5], inside the pre-set 5-point margin). Relative to a benign instruction of the
   same form, the override's excess of other wrong answers changed by −8.4 points [−12.0, −4.7]. The size of the gate
   varied widely across the three training seeds (target adoption fell by 57.8, 7.6 and 23.6 points).
3. **A released multi-search agent shows neither channel open much** (development evidence). The released
   Search-R1 7B agent searches 2.67 times per MuSiQue question. A wrong answer stated as a fact changed 1.7% of the
   answers it otherwise got right, and phrasing the same answer as an instruction added 5.9 points (47.5 points for
   the 3B agents without answer CE). It used a correct inserted answer on 58.9% of the questions it otherwise missed.
   This comes from a released model with one inserted passage and answer-level misinformation.

Evidence levels: the four primary contrasts on the 3B test split are confirmatory. Their analysis protocol was
committed by hash before the split was opened, they are scored by exact match, and they are Holm-corrected; the
intervals above resample questions with the six trained models held fixed. Everything else is labelled development
or exploratory in the paper and in each file under `results/`.

## Reproduce

Figure, tables and every number in the paper, from the result files in this repository:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python paper/make_assets.py      # writes paper/figures, paper/generated, docs/assets
bash paper/build.sh              # needs pdflatex and bibtex; writes docs/sarra.pdf
```

The released 7B study and the search-count check use only public assets (the released Search-R1 checkpoints,
MuSiQue, e5-base-v2) and need two GPUs:

```sh
export SARRA_WORKSPACE=$PWD/work SARRA_VERIFY_FROZEN=0
export MUSIQUE_DIR=<snapshot dir of the HF dataset dgslibisey/MuSiQue> E5_DIR=<snapshots dir of intfloat/e5-base-v2>
for pair in killcheck:multisearch_killcheck_v1 verify_7b:verify_7b_musique_v1; do
  src=${pair%%:*}; study=$SARRA_WORKSPACE/studies/${pair#*:}
  mkdir -p $study && cp -r sarra/$src $study/source && cp sarra/$src/protocol_public.json $study/protocol.json
done
cd $SARRA_WORKSPACE
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
    --revision d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7 --cache-dir data/hub
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-groupsize3-v0.2 \
    --revision 948503c0691ec29525f20bd7d5dbc51ed5aa7e70 --cache-dir data/hub
CUDA_VISIBLE_DEVICES=0,1 python -B studies/multisearch_killcheck_v1/source/run_job.py   # search counts, 300 questions
CUDA_VISIBLE_DEVICES=0,1 python -B studies/verify_7b_musique_v1/source/run_b1_job.py   # 7B probes, 1,000 questions
```

Each run ends with its frozen analysis (`analysis/summary.txt`). Training and evaluating the 3B agents needs the
HotpotQA splits and the trained weights, which are not included; [`sarra/training/README.md`](sarra/training/README.md)
gives the SkyRL configuration and the command for one run, and [`sarra/README.md`](sarra/README.md) describes every
study directory, the environment and what each file does.

## Repository map

| Path | Contents |
| --- | --- |
| `paper/` | LaTeX source, prompts shown in the appendix, `make_assets.py` (figure, tables, number macros), `build.sh` |
| `results/` | small JSON files the paper reads, each with its source study, sha256 and evidence level |
| `sarra/confirmation/` | 3B evaluation harness: payload construction, episode loop, scorer, plausible-wrong-answer rule, confirmatory analysis |
| `sarra/benign_controls/` | control ladder on development questions (neutral line, fluent sentence, benign instruction) |
| `sarra/probes/` | development probes: gold answer and plausible wrong answer in instruction and declarative form |
| `sarra/other_wrong/` | decomposition of non-target wrong answers and the LLM-judge calibration |
| `sarra/verify_7b/` | released Search-R1 7B on MuSiQue: probes, verification prompt, forced extra search, recovery |
| `sarra/killcheck/` | how often released Search-R1 3B and 7B search |
| `sarra/training/` | answer-CE training adapter and configuration for SkyRL's Search-R1 recipe |
| `docs/` | project page (GitHub Pages) and the PDF |

Not included: trained weights, evaluation traces and the HotpotQA split files.

## Citation

```bibtex
@misc{lu2026sarra,
  title  = {Answer Supervision Gates Injected Instructions but Barely Touches Injected Misinformation in a Single-Search Agent},
  author = {Lu, Yiwen},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/EvenEureka/search-r1-robustness}
}
```

An earlier version of this repository, with a different framing of the project, is kept at the tag
`v1-2026-10-site`.

# Results

Files that the figure, the tables and every number in the manuscript are built from (`python paper/make_assets.py`).
Each JSON file names the study and analysis output it was copied from, with the sha256 of that output, and states its
evidence level.

| File | Contents | Evidence level |
| --- | --- | --- |
| `confirmation_3b.json` | 3B agents on the 1,000-question test split: the four primary contrasts with Holm decisions, their components, the pre-declared secondary readings and per-configuration descriptives | confirmatory for P_a-P_d only; the rest is secondary |
| `confirmation_3b_outcomes.csv.gz` | one row per (configuration, seed, question, condition) of the no-defense cell for Clean GRPO, Random injection and Random + answer CE: exact match, target hit, format failure, adoption of the plausible wrong answer, lenient F1 flags, searches, and whether the clean retrieval contained the gold answer | data behind the confirmatory analysis |
| `confirmation_3b_questions.csv` | the 1,000 test question ids with their plausible wrong answer w(q) | data |
| `confirmation_3b_reanalysis.json` | written by `sarra/confirmation/scripts/reanalyze_outcomes.py` from the outcome table: the four primary decisions (identical to the frozen analysis), per-configuration and per-seed rates, interference over all clean-correct pairs, log-odds versions of the form contrast | primary decisions reproduced; everything else descriptive and post hoc |
| `example_answers.json` | the two test questions of the manuscript's Table 1 with the final answers of the seed-2 agents | illustration |
| `dev_probes_3b.json` | 3B agents on 256 development questions: gold payloads in four forms, plausible wrong answers in instruction and declarative form | development |
| `dev_controls_3b.json` | 3B agents on 256 development questions: the control ladder and the frozen rule that chose the control for the test split | development |
| `dev_verify_7b.json` | released Search-R1 7B on 1,000 MuSiQue questions, native prompt: five conditions, and the outcomes of all six frozen predictions (the manuscript uses PB1-1 to PB1-3) | development |
| `search_counts.json` | how often the released Search-R1 3B and 7B search on 300 MuSiQue questions, and their target adoption under the override attack | development; the override rows are exploratory |

Rates are fractions in the files and percentages in the paper. Intervals are 95% question-bootstrap intervals with
the trained models held fixed, unless a file says otherwise.

## Protocol hash

The confirmatory protocol was frozen with sha256
`9f2337fa656ebadda7fac838c9317d7bc51196630a23debb0fe14c51a08c3350` before any model read the test split. The released
copy, `sarra/confirmation/protocol_public.json` (sha256
`aad2c6712cd3e7475bf32e6aa6d146141a3f72471f01b13f0ad282f26d60ccd4`), differs from the hashed original only in redacted
local information: local path prefixes are replaced by relative paths, scheduler-only fields (queue, hosts, GPU
environment and interpreter paths, training job ids) and the four scheduler-script entries of its file list are
removed, job-id suffixes are removed from run directory names, and two short phrases naming the scheduler and a local log file are removed from free text.
Its `public_redaction` field records the original hash. Because of the redaction the original hash cannot be
recomputed from the released copy.

## Licenses

`confirmation_3b_outcomes.csv.gz`, `confirmation_3b_questions.csv` and `example_answers.json` are derived from
HotpotQA (Yang et al., 2018; https://hotpotqa.github.io/) and are provided under CC BY-SA 4.0. Changes made: question
identifiers, questions and gold answers were extracted, and derived fields were added (plausible wrong answers,
per-agent outcome flags, final answers of the agents). The other files are covered by the repository's Apache-2.0
license.

# Results

Small JSON files that the figure, the tables and every number in the manuscript are built from
(`python paper/make_assets.py`). Each file names the study and analysis output it was copied from, with the sha256 of
that output, and states its evidence level.

| File | Study | Evidence level |
| --- | --- | --- |
| `confirmation_3b.json` | 3B agents on the 1,000-question test split: the four primary contrasts with Holm decisions, their components, secondary criteria and per-configuration descriptives | confirmatory for P_a-P_d only; the rest is secondary |
| `dev_probes_3b.json` | 3B agents on 256 development questions: gold payloads in four forms, plausible wrong answers in instruction and declarative form | development |
| `dev_controls_3b.json` | 3B agents on 256 development questions: the control ladder (neutral line, fluent sentence, benign instruction) and the frozen rule that chose the control for the test split | development |
| `exploratory_interference_3b.json` | judge calibration (Qwen3-32B against 100 labels written by another LLM, not by people) and the decomposition of non-target wrong answers on an earlier split | exploratory |
| `dev_verify_7b.json` | released Search-R1 7B on 1,000 MuSiQue questions: five conditions x three variants, recovery measures, six frozen predictions | development |
| `killcheck_searches.json` | how often the released Search-R1 3B and 7B search on 300 MuSiQue questions | development |

Rates are fractions in the JSON and percentages in the paper. Intervals are 95% question-bootstrap intervals with
the trained models held fixed, unless a file says otherwise (Wilson intervals for small recovery counts).

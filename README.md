# Attack Resistance Is Only Half the Answer

A research project website by **Yiwen Lu · University of Notre Dame** accompanying *Attack Resistance Is Only Half the Answer: Evaluating Search-Agent Reliability*.

The homepage uses a compact academic reading layout: a centered title and author block, the manuscript abstract, a research flow diagram, navy and teal accents, light/dark themes, a continuous sequence of all three result charts, all 15 seed runs, paired outcomes, and all conditions of three saved cases. It is a working manuscript, not an accepted or peer-reviewed paper. The study's historical training/export inconsistency and differences in supervision, evidence, and compute remain visible alongside the results.

## Read locally

Open `docs/index.html`, or serve the site:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Visit `http://127.0.0.1:8765/`. The current paper is `docs/emnlp-manuscript.pdf`; its LaTeX and evidence bundle is `docs/emnlp-source.zip`. The earlier technical report remains available as `docs/report.html` and `docs/manuscript.pdf`.

The site uses plain HTML, CSS, SVG, and JavaScript. All assets and reduced evidence are local. There is no backend, tracking, external font request, live model call, or GPU dependency. The illustration is conceptual; its side-by-side scores display saved three-seed means. Selected case answers come from saved trajectories. All results, cases, and scope details appear in reading order without selecting tabs or opening sections. Navigation only jumps to sections. The interface supports small screens, keyboard navigation, reduced motion, and full reading without JavaScript.

## Build

`project_site.py` uses Python's standard library to regenerate the homepage and integrity manifest:

```sh
python project_site.py
```

For a full rebuild of the legacy report, figures, tables, and homepage, use Python 3.10 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build_report.py
```

The full builder verifies aggregate means, sample standard deviations, primary seed metrics, condition denominators, and supervision totals. It does not retrain models or recompute bootstrap intervals from raw predictions. The current EMNLP-style PDF is a separate LaTeX build supplied in `emnlp-source.zip`; these website commands preserve it.

| Source | Purpose |
| --- | --- |
| `templates/index.html` | Homepage structure and narrative |
| `project_site.py` | Frozen-data preparation, static rendering, citation, release hashes |
| `docs/assets/project.css` | Academic visual design, navy/teal themes, responsive layout, and components |
| `docs/assets/project.js` | Light/dark theme and citation copying; all evidence is statically rendered |
| `manuscript.json`, `build_report.py` | Earlier technical report and verified tables |
| `verify_site.py` | Browser checks for continuous visibility, data, links, and responsive layout |

## Browser validation

With the local server running:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python verify_site.py --url http://127.0.0.1:8765/ --output /tmp/searchr1-site-review
```

Checks cover both hero configurations, 15 chart means and 45 seed values, 18 case answers, four paired outcomes, no hidden panels or collapsed sections, local links, clipboard, persistent theme switching, reduced motion, keyboard entry, complete no-JavaScript content, and widths 320, 390, 768, and 1440 pixels. Screenshots and a validation JSON are written to the selected output directory.

## Evidence

The primary evaluation includes five training configurations, three seeds, and 1,000 held-out HotpotQA questions under five conditions: 75,000 trajectories. These are repeated evaluations of the same questions, not 75,000 independent examples. Random injection plus answer supervision scores 33.37% attacked exact match versus 23.30% for random injection alone in the evaluated historical exports; this comparison does not isolate a causal mechanism.

| File under `docs/data/` | Contents |
| --- | --- |
| `primary.json`, `aggregate.csv` | Primary summaries and three-seed means/SDs |
| `seeds.csv` | All 15 primary runs |
| `conditions.csv` | All 75 run/condition summaries |
| `costs.csv`, `training-accounting.json` | Exposure, tokens, timing, and allocated GPU-hours |
| `protocol.json`, `model-hashes.json` | Reduced frozen protocol and evaluated export identity |
| `cases.json` | Selected question text and saved trajectory excerpts |
| `paired-analysis.json` | Paired outcome counts and analysis scope |
| `extension.json` | Separate context controls and transfer summaries |
| `exploratory-diagnostics.json` | Separate exploratory diagnoses and limitations |
| `provenance.json` | Source evidence hashes |

`docs/release-sha256.json` lists integrity hashes for the static release. Rates in CSVs are fractions; displayed percentages multiply them by 100. Search counts and costs are not percentages. Seed 1–3 retain their original identifiers in the data. All seeds remain available; no favorable seed is relabeled as a configuration.

This reporting release includes selected case questions and excerpts, but excludes the full raw trajectory archive, complete benchmark, model weights, optimizer states, and standalone training environment.

## Publish on GitHub Pages

Intended repository: `EvenEureka/search-r1-robustness`. Intended URL: `https://eveneureka.github.io/search-r1-robustness/`. This is a destination, not a claim that publication has completed.

1. Create the public repository and initialize its `main` branch. Grant the connected GitHub app access if using it to upload.
2. Upload this directory's contents to the repository root, preserving `docs/` and `templates/`.
3. In **Settings → Pages**, choose **Deploy from a branch**, **main**, and **/docs**.
4. Verify the deployed homepage, paper, source download, and interactions.

The built files are served directly; no remote build or API key is needed. See [GitHub's publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

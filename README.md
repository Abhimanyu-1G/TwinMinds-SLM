# SLM Project Workspace

A research-backed playbook of 21 small-language-model projects for a Deep Learning course, plus a concrete feasibility and execution plan for a two-stance SLM pair on an RTX 5070 Ti laptop.

## Files

| File | What it is |
|---|---|
| [SLM-Project-Playbook.md](SLM-Project-Playbook.md) | The main document: orientation, 2026 model landscape, compute guide, 21 project blueprints, execution plan, resource library, ethics. |
| [projects.csv](projects.csv) | The same 21 projects as a sortable table (track, difficulty, compute, hypothesis, metric, starter link). |
| [SLM-Stance-Pair-Plan.md](SLM-Stance-Pair-Plan.md) | Feasibility verdict, data and training recipe, evaluation, UI spec, timeline, and cost for the pro/con debate-model project on a 12 GB RTX 5070 Ti. |
| [SLM-Keyword-Pair-Plan.md](SLM-Keyword-Pair-Plan.md) | The from-scratch variant: two ~20M keyword models, data construction without distillation, decoding-time polarity guarantees, and a 2–3 h training budget. |
| `tools/` | Small Node scripts used to fetch sources, pull Hugging Face metadata, and verify every link in the playbook. Safe to delete; useful if you want to update the list in a few months. |

## Start here

- **Choosing a course project:** `projects.csv` → playbook §5.
- **Building the pro/con pair:** [SLM-Stance-Pair-Plan.md](SLM-Stance-Pair-Plan.md) §0 (verdict) and §5 (training recipe).
- **Training from scratch (keywords only):** [SLM-Keyword-Pair-Plan.md](SLM-Keyword-Pair-Plan.md) §2 (what 2–3 h buys) and §4 (data pipeline).

## Use it in three steps

1. **Filter.** Open `projects.csv`, filter by compute tier (A = no GPU, B = free T4, C = 1×A100, D = 8×GPU) and difficulty. Keep three candidates.
2. **Read.** In the playbook, read the full entry for each candidate — hypothesis, build sequence, measurement, risks.
3. **Commit.** Fill in the hypothesis template (§6.6), estimate spend with `6 × N × D`, and write the one-page proposal before touching a GPU.

## If you only read one section

§0 (six flagship picks) and §5 (the catalogue). The six: Speedrun Zero (A1), Minimum Viable Reasoner (C1), Professor & Student (B1), The Quantization Frontier (B2), Microscope (F1), Pocket Agent (E1).

## How this was assembled

Model sizes, context lengths, and licenses were read from Hugging Face model cards and `config.json` files in November 2026. Papers were read via arXiv abstracts and, where relevant, the HTML full text. All 127 unique links in the playbook were HTTP-checked; the only non-200 is `lmarena.ai`, which returns 403 to non-browser clients but is a valid page. Compute and cost figures are order-of-magnitude estimates from the FLOPs formula, not benchmarks.

## Scope note

The list favors projects that make a *claim* you can falsify, not projects that end in a demo. If your course rewards a working artifact more than an experiment, start from Track E (Deploy) and add one measurement from Track B or F.

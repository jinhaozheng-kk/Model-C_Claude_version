# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Katie Model C3 — Scoring Engine**: A Python CLI that computes multi-level friction-based accessibility and usability scores from website user testing data (originally built for AAMI Insurance website testing).

## Commands

Install dependencies:
```
pip install -r requirements.txt
```

Prepare raw data (AAMI-specific adapter — reads from `data/`):
```
python prepare_data.py
```

Run scoring pipeline:
```
python main.py --insights <insights.csv> --survey <survey.csv> [--timing <timing.csv>] [--project-blocked <project_blocked.csv>] [--output results/]
```

Run with sample data (quick smoke test):
```
python main.py --insights sample_data/insights.csv --survey sample_data/survey.csv --timing sample_data/timing.csv --project-blocked sample_data/project_blocked.csv --output results
```

## Architecture

### Two-Phase Pipeline

**Phase 1 — Data Preparation** (`prepare_data.py`): AAMI-specific adapter that normalises raw Excel/CSV exports into three canonical CSVs: `survey.csv`, `insights.csv`, `project_blocked.csv`. Contains hardcoded mappings for AAMI cohort codes, friction types, and task names. Outputs to `data/prepared/`.

**Phase 2 — Scoring** (`main.py` + `scoring_engine/`): Reads the canonical CSVs and produces four output CSVs.

### Three-Tier Scoring Model

All scoring logic lives in `scoring_engine/`. The three levels are strictly ordered and independent streams after L1:

| Level | Module | Stream | Description |
|-------|--------|--------|-------------|
| L1 | `l1_scorer.py` | Base | Task performance ceiling from `ease_of_use_score`; reduced by accessibility friction severity/confidence; can hit 0 (blocked) |
| L2 | `l2_scorer.py` | Accessibility | Starts from L1; applies AT friction caps then comprehension caps scaled by cohort accessibility weight |
| L3 | `l3_scorer.py` | Usability | Starts from L1 (independent of L2); applies usability/comprehension caps, `met_expectations` reduction, and time-based caps |

Rules within each scorer are numbered (e.g. L1.1–L1.6, L2.1–L2.8) in module docstrings — always refer to these rule IDs when modifying logic.

### Key Files

- [scoring_engine/constants.py](scoring_engine/constants.py) — all numeric thresholds (`CONFIDENCE_THRESHOLD=0.70`, `L1_POSITIVE_FLOOR=55.0`, `L2_COMP_FLOOR=45.0`, `DEDUP_SIMILARITY_THRESHOLD=0.85`) and `COHORT_WEIGHTS` mapping
- [scoring_engine/models.py](scoring_engine/models.py) — dataclasses: `FrictionInsight`, `SurveyResponse`, `TimingRecord`
- [scoring_engine/preprocess.py](scoring_engine/preprocess.py) — deduplicates insights by semantic similarity (TF-IDF/cosine via scikit-learn, falls back to `difflib.SequenceMatcher`); merges to higher severity/confidence
- [scoring_engine/aggregation.py](scoring_engine/aggregation.py) — aggregates per-participant scores up to task → cohort → project level; project-blocked participants contribute 0 to accessibility but are excluded from usability

### Cohort-Aware Weighting

Each cohort has an `(accessibility_weight, usability_weight)` pair in `COHORT_WEIGHTS`. These weights scale how much comprehension friction affects L2 vs. L3. General population is usability-dominant (0.4, 1.0); ESL/Deaf/Neurodivergent are accessibility-dominant (1.0, 0.4–0.9).

### Output Files

All four output CSVs go to `--output` directory (default `results/`):
- `participant_task_scores.csv` — L1/L2/L3 per participant per task
- `task_scores.csv` — mean ± std per task
- `cohort_scores.csv` — mean ± std per cohort × task with participant count
- `project_scores.csv` — project-level summary (project-blocked participants set accessibility to 0)

## Data Schema

`insights.csv` columns: `participant_id`, `task_id`, `insight_text`, `friction_type` (accessibility/comprehension/usability/other), `severity` (blocked/severe/high/medium/low), `confidence` (0.0–1.0)

`survey.csv` columns: `participant_id`, `task_id`, `ease_of_use_score` (0–100), `met_expectations` (exceeded/met/not met/fell well below), `cohort`

`timing.csv` columns: `participant_id`, `task_id`, `time_seconds`

`project_blocked.csv` columns: `participant_id` (participants with project-level AT blockers affecting all tasks)

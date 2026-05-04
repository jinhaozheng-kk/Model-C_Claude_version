"""
Katie Model C3 — Scoring Engine CLI

Usage:
    python main.py [--insights insights.csv] [--survey survey.csv] [--timing timing.csv]
                   [--project-blocked project_blocked.csv] [--output results/]

Outputs (written to --output directory):
    participant_task_scores.csv   — L1 / L2 / L3 per participant per task
    task_scores.csv               — aggregated per task
    cohort_scores.csv             — aggregated per cohort × task
    project_scores.csv            — project-level summary with std deviation
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from scoring_engine.constants import L1_POSITIVE_FLOOR
from scoring_engine.models import FrictionInsight, SurveyResponse
from scoring_engine.preprocess import deduplicate_insights
from scoring_engine.l1_scorer import score_l1
from scoring_engine.l2_scorer import score_l2
from scoring_engine.l3_scorer import score_l3
from scoring_engine.aggregation import (
    compute_task_scores,
    compute_cohort_scores,
    compute_project_scores,
)

DEFAULT_INSIGHTS_PATH = "data/prepared/insights.csv"
DEFAULT_SURVEY_PATH = "data/prepared/survey.csv"
DEFAULT_PROJECT_BLOCKED_PATH = "data/prepared/project_blocked.csv"
DEFAULT_OUTPUT_DIR = "data/results"

REQUIRED_INSIGHTS_COLUMNS = {
    "participant_id",
    "task_id",
    "insight_text",
    "friction_type",
    "severity",
    "confidence",
}
REQUIRED_SURVEY_COLUMNS = {
    "participant_id",
    "task_id",
    "ease_of_use_score",
    "met_expectations",
    "cohort",
}
REQUIRED_TIMING_COLUMNS = {"participant_id", "task_id", "time_seconds"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[warning] file not found, skipping: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _prepare_inputs(
    insights_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    timing_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and normalise canonical scoring inputs."""
    _require_columns(insights_df, REQUIRED_INSIGHTS_COLUMNS, "insights.csv")
    _require_columns(survey_df, REQUIRED_SURVEY_COLUMNS, "survey.csv")
    if timing_df is not None:
        _require_columns(timing_df, REQUIRED_TIMING_COLUMNS, "timing.csv")

    insights_df = insights_df.copy()
    survey_df = survey_df.copy()
    timing_df = timing_df.copy() if timing_df is not None else None

    for df in (insights_df, survey_df):
        df["participant_id"] = df["participant_id"].astype(str).str.strip()
        df["task_id"] = df["task_id"].astype(str).str.strip()

    insights_df["insight_text"] = insights_df["insight_text"].fillna("").astype(str)
    insights_df["friction_type"] = insights_df["friction_type"].astype(str).str.strip().str.lower()
    insights_df["severity"] = insights_df["severity"].astype(str).str.strip().str.lower()
    insights_df["confidence"] = pd.to_numeric(insights_df["confidence"], errors="raise")

    survey_df["cohort"] = survey_df["cohort"].fillna("general").astype(str).str.strip().str.lower()
    survey_df["met_expectations"] = survey_df["met_expectations"].apply(_normalise_met_expectations)
    survey_df["ease_of_use_score"] = pd.to_numeric(survey_df["ease_of_use_score"], errors="raise")

    if not insights_df["confidence"].between(0, 1).all():
        raise ValueError("insights.csv confidence values must be between 0 and 1.")
    if not survey_df["ease_of_use_score"].between(0, 100).all():
        raise ValueError("survey.csv ease_of_use_score values must be between 0 and 100.")

    if timing_df is not None:
        timing_df["participant_id"] = timing_df["participant_id"].astype(str).str.strip()
        timing_df["task_id"] = timing_df["task_id"].astype(str).str.strip()
        timing_df["time_seconds"] = pd.to_numeric(timing_df["time_seconds"], errors="raise")
        if (timing_df["time_seconds"] < 0).any():
            raise ValueError("timing.csv time_seconds values must be non-negative.")

    return insights_df, survey_df, timing_df


def _normalise_met_expectations(value: str) -> str:
    """Normalise free-text met_expectations values to the four canonical forms."""
    v = str(value).strip().lower()
    if v in ("exceeded", "exceeded expectations"):
        return "exceeded"
    if v in ("met", "met expectations"):
        return "met"
    if "well below" in v:
        return "fell well below"
    if "not met" in v or v in ("below", "not_met"):
        return "not met"
    return v


# ---------------------------------------------------------------------------
# Timing ratio computation
# ---------------------------------------------------------------------------

def _compute_timing_ratios(timing_df: Optional[pd.DataFrame]) -> dict:
    """Return {(participant_id, task_id): timing_ratio} relative to task median."""
    if timing_df is None or timing_df.empty:
        return {}

    medians = timing_df.groupby("task_id")["time_seconds"].median()
    ratios: dict[tuple, float] = {}
    for _, row in timing_df.iterrows():
        median = medians.get(row["task_id"])
        if median and median > 0:
            ratios[(row["participant_id"], row["task_id"])] = float(row["time_seconds"]) / float(median)
    return ratios


# ---------------------------------------------------------------------------
# Core scoring pipeline
# ---------------------------------------------------------------------------

def run_scoring(
    insights_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    timing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the full L1 → L2 → L3 pipeline and return a flat scores DataFrame."""

    insights_df, survey_df, timing_df = _prepare_inputs(insights_df, survey_df, timing_df)
    insights_df = deduplicate_insights(insights_df)
    timing_ratios = _compute_timing_ratios(timing_df)

    records: list[dict] = []

    for _, srv in survey_df.iterrows():
        pid = str(srv["participant_id"])
        tid = str(srv["task_id"])
        cohort = str(srv.get("cohort", "general"))
        met_exp = str(srv.get("met_expectations", "met"))

        # Collect friction insights for this participant × task
        mask = (
            (insights_df["participant_id"].astype(str) == pid) &
            (insights_df["task_id"].astype(str) == tid)
        )
        insights: list[FrictionInsight] = [
            FrictionInsight(
                participant_id=str(r["participant_id"]),
                task_id=str(r["task_id"]),
                insight_text=str(r.get("insight_text", "")),
                friction_type=str(r["friction_type"]).strip().lower(),
                severity=str(r["severity"]).strip().lower(),
                confidence=float(r["confidence"]),
            )
            for _, r in insights_df[mask].iterrows()
        ]

        survey = SurveyResponse(
            participant_id=pid,
            task_id=tid,
            ease_of_use_score=int(srv["ease_of_use_score"]),
            met_expectations=met_exp,
            cohort=cohort,
        )

        l1_score, is_blocked, eligible_floor = score_l1(survey, insights)

        l2_score = score_l2(l1_score, is_blocked, insights, met_exp, cohort)
        l3_score = score_l3(
            l1_score, is_blocked, insights, met_exp, cohort,
            timing_ratios.get((pid, tid)),
        )

        # L1.5: positive floor applied after both L2 and L3 are computed
        if eligible_floor and not is_blocked:
            l2_score = max(l2_score, L1_POSITIVE_FLOOR)
            l3_score = max(l3_score, L1_POSITIVE_FLOOR)

        records.append({
            "participant_id":    pid,
            "task_id":           tid,
            "cohort":            cohort,
            "ease_of_use_score": int(srv["ease_of_use_score"]),
            "met_expectations":  met_exp,
            "is_blocked":        is_blocked,
            "l1_score":          round(l1_score, 2),
            "l2_accessibility":  round(l2_score, 2),
            "l3_usability":      round(l3_score, 2),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Katie Model C3 — Accessibility & Usability Scoring Engine"
    )
    parser.add_argument(
        "--insights",
        default=DEFAULT_INSIGHTS_PATH,
        help=f"Path to insights CSV (default: {DEFAULT_INSIGHTS_PATH})",
    )
    parser.add_argument(
        "--survey",
        default=DEFAULT_SURVEY_PATH,
        help=f"Path to survey responses CSV (default: {DEFAULT_SURVEY_PATH})",
    )
    parser.add_argument("--timing",          default=None,   help="Path to task timing CSV (optional)")
    parser.add_argument(
        "--project-blocked",
        default=DEFAULT_PROJECT_BLOCKED_PATH,
        help=f"Path to project-level blocked participants CSV (default: {DEFAULT_PROJECT_BLOCKED_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}/)",
    )
    args = parser.parse_args()

    insights_df        = _load_csv(args.insights)
    survey_df          = _load_csv(args.survey)
    timing_df          = _load_csv(args.timing)
    project_blocked_df = _load_csv(args.project_blocked)

    if insights_df is None or survey_df is None:
        print("[error] --insights and --survey are required and must exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    try:
        scores_df = run_scoring(insights_df, survey_df, timing_df)

        # Build l2/l3 columns expected by aggregation helpers
        agg_df = scores_df.rename(columns={"l2_accessibility": "l2_score", "l3_usability": "l3_score"})

        task_df = compute_task_scores(agg_df)
        cohort_df = compute_cohort_scores(agg_df)
        project_df = compute_project_scores(agg_df, project_blocked_df)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    scores_df.to_csv(os.path.join(args.output, "participant_task_scores.csv"), index=False)
    task_df.to_csv(os.path.join(args.output, "task_scores.csv"), index=False)
    cohort_df.to_csv(os.path.join(args.output, "cohort_scores.csv"), index=False)
    project_df.to_csv(os.path.join(args.output, "project_scores.csv"), index=False)

    print(f"Scoring complete — results written to '{args.output}/'")
    print("\n── Project Summary ──────────────────────────────")
    print(project_df.to_string(index=False))
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()

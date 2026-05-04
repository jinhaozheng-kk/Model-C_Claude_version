"""
Aggregation layer: task → cohort → project scores.

Blocked participants (is_blocked=True) are excluded from task and cohort averages.
Project-level blocked participants (project_blocked.csv) contribute 0 to the
project accessibility score. Usability still comes from non-blocked task rows.
"""

from typing import Optional

import pandas as pd


def _normalise_id(value: object) -> str:
    return str(value).strip().lower()


def _project_blocked_ids(project_blocked_df: Optional[pd.DataFrame]) -> set[str]:
    if project_blocked_df is None or project_blocked_df.empty:
        return set()
    if "participant_id" not in project_blocked_df.columns:
        raise ValueError("project_blocked.csv is missing required column: participant_id")

    if "blocked" in project_blocked_df.columns:
        mask = project_blocked_df["blocked"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        blocked = project_blocked_df.loc[mask, "participant_id"]
    else:
        blocked = project_blocked_df["participant_id"]

    return {_normalise_id(pid) for pid in blocked}


def compute_task_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-task mean ± std for accessibility (L2) and usability (L3).
    Blocked participants are excluded from both averages.
    """
    active = scores_df[~scores_df["is_blocked"]]
    acc = (
        active.groupby("task_id")["l2_score"]
        .agg(accessibility_mean="mean", accessibility_std="std")
        .reset_index()
    )
    usa = (
        active.groupby("task_id")["l3_score"]
        .agg(usability_mean="mean", usability_std="std")
        .reset_index()
    )
    return acc.merge(usa, on="task_id")


def compute_cohort_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (cohort × task) mean ± std for both streams, with participant count.
    Blocked participants are excluded.
    """
    active = scores_df[~scores_df["is_blocked"]]
    return (
        active.groupby(["cohort", "task_id"])
        .agg(
            accessibility_mean=("l2_score", "mean"),
            accessibility_std=("l2_score", "std"),
            usability_mean=("l3_score", "mean"),
            usability_std=("l3_score", "std"),
            participant_count=("participant_id", "count"),
        )
        .reset_index()
    )


def compute_project_scores(
    scores_df: pd.DataFrame,
    project_blocked_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Project-level accessibility and usability scores with standard deviation.

    Accessibility:
      - Per participant, average their L2 scores across non-blocked tasks.
      - Participants listed in project_blocked_df contribute 0 (categorical failure).

    Usability:
      - Per participant, average their L3 scores across non-blocked tasks.
      - Project-level blocking does not zero the usability contribution.
    """
    active = scores_df.loc[~scores_df["is_blocked"]].copy()
    active["_participant_key"] = active["participant_id"].map(_normalise_id)

    # Per-participant means across tasks
    acc_per_p = active.groupby("_participant_key")["l2_score"].mean()
    usa_per_p = active.groupby("_participant_key")["l3_score"].mean()

    blocked_ids = _project_blocked_ids(project_blocked_df)
    if blocked_ids:
        # Project-blocked participants contribute 0 to accessibility even if
        # they have no non-blocked task rows in scores_df.
        acc_values = {pid: float(score) for pid, score in acc_per_p.items()}
        for pid in blocked_ids:
            acc_values[pid] = 0.0
        acc_per_p = pd.Series(acc_values, dtype=float)

    def _safe_std(series: pd.Series) -> float:
        return round(float(series.std(ddof=1)), 2) if len(series) > 1 else 0.0

    return pd.DataFrame([{
        "accessibility_mean": round(float(acc_per_p.mean()), 2) if len(acc_per_p) else None,
        "accessibility_std":  _safe_std(acc_per_p),
        "usability_mean":     round(float(usa_per_p.mean()), 2) if len(usa_per_p) else None,
        "usability_std":      _safe_std(usa_per_p),
        "n_participants_acc": len(acc_per_p),
        "n_participants_usa": len(usa_per_p),
    }])

"""
L3 — Usability score per participant per task.

Starts from the L1 score (ceiling), independent of L2.
Cap families applied:
  1. Pure usability frictions (non-accessibility, non-comprehension): L3.1–L3.5
  2. Comprehension frictions scaled by cohort USA weight (L2.7 applied to USA stream)

Then moderators in order:
  L3.7 — met_expectations percentage reduction
  L3.8 — time-based cap (relative to task median)

L3.6: among all triggered caps, the most restrictive (lowest score) wins.
Blocked participants return 0.0 and are excluded from task averages.
"""

from typing import Optional

from .models import FrictionInsight
from .constants import CONFIDENCE_THRESHOLD, COHORT_WEIGHTS
from .l2_scorer import _avg_conf, _effective_cap, _apply_cap


def score_l3(
    l1_score: float,
    is_blocked: bool,
    insights: list,
    met_expectations: str,
    cohort: str,
    timing_ratio: Optional[float] = None,
) -> float:
    """Return the usability score for this participant × task."""
    if is_blocked:
        return 0.0

    usa_weight = COHORT_WEIGHTS.get(cohort.lower(), COHORT_WEIGHTS["general"])["usa"]

    qualifying = [f for f in insights if f.confidence > CONFIDENCE_THRESHOLD]

    # Usability frictions = anything that is not accessibility or comprehension
    usa_frictions  = [f for f in qualifying if f.friction_type not in ("accessibility", "comprehension")]
    comp_frictions = [f for f in qualifying if f.friction_type == "comprehension"]

    severe_usa = [f for f in usa_frictions if f.severity in ("blocked", "severe")]
    high_usa   = [f for f in usa_frictions if f.severity == "high"]
    medium_usa = [f for f in usa_frictions if f.severity == "medium"]
    low_usa    = [f for f in usa_frictions if f.severity == "low"]

    # Collect all (effective_cap_value, weight) pairs; each evaluated against l1_score
    all_caps: list[tuple[float, float]] = []

    # L3.1: severe usability friction OR 3+ high → cap 50
    if severe_usa:
        all_caps.append((_effective_cap(50.0, _avg_conf(severe_usa)), 1.0))
    if len(high_usa) >= 3:
        all_caps.append((_effective_cap(50.0, _avg_conf(high_usa)), 1.0))

    # L3.2: any high + any medium → cap 60
    if high_usa and medium_usa:
        all_caps.append((_effective_cap(60.0, _avg_conf(high_usa + medium_usa)), 1.0))

    # L3.3: any medium + any low → cap 75
    if medium_usa and low_usa:
        all_caps.append((_effective_cap(75.0, _avg_conf(medium_usa + low_usa)), 1.0))

    # L3.4: 3+ medium → cap 70
    if len(medium_usa) >= 3:
        all_caps.append((_effective_cap(70.0, _avg_conf(medium_usa)), 1.0))

    # L3.5: 3+ low → cap 80
    if len(low_usa) >= 3:
        all_caps.append((_effective_cap(80.0, _avg_conf(low_usa)), 1.0))

    # Comprehension caps in usability stream (L2.7 — same nominal caps, USA weight)
    high_sev_comp = [f for f in comp_frictions if f.severity in ("high", "severe")]
    if high_sev_comp:
        all_caps.append((_effective_cap(40.0, _avg_conf(high_sev_comp)), usa_weight))

    medium_comp = [f for f in comp_frictions if f.severity == "medium"]
    if len(medium_comp) >= 3:
        all_caps.append((_effective_cap(40.0, _avg_conf(medium_comp)), usa_weight))
    elif medium_comp:
        all_caps.append((_effective_cap(50.0, _avg_conf(medium_comp)), usa_weight))

    low_comp = [f for f in comp_frictions if f.severity == "low"]
    if len(low_comp) >= 3:
        all_caps.append((_effective_cap(60.0, _avg_conf(low_comp)), usa_weight))
    elif low_comp:
        all_caps.append((_effective_cap(70.0, _avg_conf(low_comp)), usa_weight))

    # L3.6: most restrictive cap wins (evaluate each against l1_score, take minimum)
    score = l1_score
    for cap_val, w in all_caps:
        score = min(score, _apply_cap(l1_score, cap_val, w))

    # L3.7: met_expectations percentage reduction
    if met_expectations == "not met":
        score *= 0.80
    elif met_expectations == "fell well below":
        score *= 0.60

    # L3.8: time-based cap (timing_ratio = participant_time / task_median)
    if timing_ratio is not None:
        if timing_ratio >= 4.0:
            score = min(score, 25.0)
        elif timing_ratio >= 3.0:
            score = min(score, 40.0)
        elif timing_ratio >= 2.0:
            score = min(score, 50.0)

    return round(score, 2)

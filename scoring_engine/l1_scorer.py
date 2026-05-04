"""
L1 — Task / component score per participant.

Starting point: participant's self-reported ease_of_use_score (the ceiling).
Scores can only go down from here.
"""

from .models import FrictionInsight, SurveyResponse
from .constants import CONFIDENCE_THRESHOLD


def _avg_conf(frictions: list) -> float:
    if not frictions:
        return 0.0
    return sum(f.confidence for f in frictions) / len(frictions)


def _effective_cap(nominal_cap: float, avg_confidence: float) -> float:
    """
    Scale cap strictness by confidence (L1.4).
    At conf=1.0 → full nominal cap.  At conf=THRESHOLD → no cap (returns 100).
    Linear interpolation in between.
    """
    if avg_confidence < CONFIDENCE_THRESHOLD:
        return 100.0
    ratio = (avg_confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
    return nominal_cap + (100.0 - nominal_cap) * (1.0 - ratio)


def score_l1(
    survey: SurveyResponse,
    insights: list,
) -> tuple:
    """
    Returns (l1_score, is_blocked, eligible_for_positive_floor).

    is_blocked=True when a blocked/severe accessibility friction is identified
    with confidence >70%.  That score is 0 and is excluded from task averages.

    eligible_for_positive_floor is evaluated here but the floor itself is
    applied in main.py after L2 and L3 are computed (L1.5).
    """
    base = float(survey.ease_of_use_score)

    # Qualifying accessibility frictions only (other types handled in L2/L3)
    at_qualifying = [
        f for f in insights
        if f.friction_type == "accessibility" and f.confidence > CONFIDENCE_THRESHOLD
    ]

    # L1.2: blocked or severe AT friction → categorical failure
    if any(f.severity in ("blocked", "severe") for f in at_qualifying):
        return 0.0, True, False

    # L1.3: high AT friction caps
    high_at = [f for f in at_qualifying if f.severity == "high"]
    caps: list[float] = []
    if len(high_at) >= 2:
        caps.append(_effective_cap(10.0, _avg_conf(high_at)))
    elif high_at:
        caps.append(_effective_cap(20.0, _avg_conf(high_at)))

    l1_score = min([base] + caps)

    # L1.5 eligibility: survey ≥75, met/exceeded expectations, no high/severe frictions
    all_qualifying = [f for f in insights if f.confidence > CONFIDENCE_THRESHOLD]
    has_high_or_severe = any(
        f.severity in ("blocked", "severe", "high") for f in all_qualifying
    )
    eligible_floor = (
        survey.ease_of_use_score >= 75
        and survey.met_expectations in ("exceeded", "met")
        and not has_high_or_severe
    )

    return l1_score, False, eligible_floor

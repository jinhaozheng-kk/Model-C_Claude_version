"""
L2 — Accessibility score per participant per task.

Starts from the L1 score (ceiling).  Two cap families are applied sequentially:
  1. AT (assistive-technology) friction caps  — full weight for all cohorts
  2. Comprehension friction caps              — scaled by cohort ACC weight (L2.7)

L2.6: if the participant exceeded expectations, comprehension frictions cannot
      push the score below 45 (but AT frictions can still push below 45).
"""

from .models import FrictionInsight
from .constants import CONFIDENCE_THRESHOLD, COHORT_WEIGHTS, L2_COMP_FLOOR


# ---------------------------------------------------------------------------
# Helpers shared across L2 and L3
# ---------------------------------------------------------------------------

def _avg_conf(frictions: list) -> float:
    if not frictions:
        return 0.0
    return sum(f.confidence for f in frictions) / len(frictions)


def _effective_cap(nominal_cap: float, avg_confidence: float) -> float:
    """
    Confidence-scaled cap (L1.4).
    conf=1.0  → full nominal_cap (most restrictive).
    conf=0.70 → 100 (no effect).
    """
    if avg_confidence < CONFIDENCE_THRESHOLD:
        return 100.0
    ratio = (avg_confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
    return nominal_cap + (100.0 - nominal_cap) * (1.0 - ratio)


def _apply_cap(base: float, cap_value: float, weight: float) -> float:
    """
    Apply a single cap to base with cohort weight.
    weight=1.0 → full cap pressure; weight=0.4 → 40% of cap pressure.
    If base is already at or below cap_value, no change.
    """
    if base <= cap_value:
        return base
    cap_pressure = base - cap_value
    return base - cap_pressure * weight


def _most_restrictive(base: float, caps: list) -> float:
    """
    Apply each (cap_value, weight) independently to base and return the
    lowest resulting score (L3.6 / most-restrictive-wins principle).
    """
    result = base
    for cap_val, w in caps:
        result = min(result, _apply_cap(base, cap_val, w))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_l2(
    l1_score: float,
    is_blocked: bool,
    insights: list,
    met_expectations: str,
    cohort: str,
) -> float:
    """Return the accessibility score for this participant × task."""
    if is_blocked:
        return 0.0

    acc_weight = COHORT_WEIGHTS.get(cohort.lower(), COHORT_WEIGHTS["general"])["acc"]

    qualifying = [f for f in insights if f.confidence > CONFIDENCE_THRESHOLD]
    at_frictions   = [f for f in qualifying if f.friction_type == "accessibility"]
    comp_frictions = [f for f in qualifying if f.friction_type == "comprehension"]

    # ---- Step 1: AT friction caps (weight always 1.0) ----------------------
    at_caps: list[tuple[float, float]] = []

    high_at = [f for f in at_frictions if f.severity == "high"]
    if len(high_at) >= 2:
        at_caps.append((_effective_cap(10.0, _avg_conf(high_at)), 1.0))
    elif high_at:
        at_caps.append((_effective_cap(20.0, _avg_conf(high_at)), 1.0))

    medium_at = [f for f in at_frictions if f.severity == "medium"]
    if len(medium_at) >= 3:
        at_caps.append((_effective_cap(40.0, _avg_conf(medium_at)), 1.0))   # L2.1 count-based
    elif medium_at:
        at_caps.append((_effective_cap(30.0, _avg_conf(medium_at)), 1.0))   # L2.1

    low_at = [f for f in at_frictions if f.severity == "low"]
    if len(low_at) >= 3:
        at_caps.append((_effective_cap(60.0, _avg_conf(low_at)), 1.0))      # count rule
    elif low_at:
        at_caps.append((_effective_cap(70.0, _avg_conf(low_at)), 1.0))      # new rule

    score_after_at = _most_restrictive(l1_score, at_caps)

    # ---- Step 2: Comprehension friction caps (scaled by ACC cohort weight) -
    comp_caps: list[tuple[float, float]] = []

    high_sev_comp = [f for f in comp_frictions if f.severity in ("high", "severe")]
    if high_sev_comp:
        comp_caps.append((_effective_cap(40.0, _avg_conf(high_sev_comp)), acc_weight))  # L2.2

    medium_comp = [f for f in comp_frictions if f.severity == "medium"]
    if len(medium_comp) >= 3:
        comp_caps.append((_effective_cap(40.0, _avg_conf(medium_comp)), acc_weight))    # L2.4
    elif medium_comp:
        comp_caps.append((_effective_cap(50.0, _avg_conf(medium_comp)), acc_weight))    # L2.3

    low_comp = [f for f in comp_frictions if f.severity == "low"]
    if len(low_comp) >= 3:
        comp_caps.append((_effective_cap(60.0, _avg_conf(low_comp)), acc_weight))       # L2.5
    elif low_comp:
        comp_caps.append((_effective_cap(70.0, _avg_conf(low_comp)), acc_weight))       # L2.5a

    score_after_comp = _most_restrictive(score_after_at, comp_caps)

    # ---- Step 3: L2.6 comprehension floor when exceeded expectations -------
    # Comprehension frictions cannot push the score below 45, but AT frictions can.
    if met_expectations == "exceeded":
        # The floor is the lesser of: the AT-reduced score and 45.
        # This means if AT already pushed below 45, AT's result is respected.
        floor = min(score_after_at, L2_COMP_FLOOR)
        final = max(score_after_comp, floor)
    else:
        final = score_after_comp

    return round(final, 2)

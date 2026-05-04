CONFIDENCE_THRESHOLD = 0.70

# L2.7 cohort-aware comprehension weights: acc = accessibility stream, usa = usability stream
COHORT_WEIGHTS: dict[str, dict[str, float]] = {
    "esl":            {"acc": 1.0, "usa": 0.4},
    "deaf":           {"acc": 1.0, "usa": 0.4},
    "neurodivergent": {"acc": 1.0, "usa": 0.4},
    "blind_lv":       {"acc": 0.5, "usa": 0.9},
    "older_adult":    {"acc": 0.6, "usa": 0.8},
    "general":        {"acc": 0.4, "usa": 1.0},
}

# Severity ranking for deduplication merge (higher = more severe)
SEVERITY_RANK: dict[str, int] = {
    "low":     0,
    "medium":  1,
    "high":    2,
    "severe":  3,
    "blocked": 4,
}

# L1.5 positive floor (applied after L2 and L3)
L1_POSITIVE_FLOOR: float = 55.0

# L2.6 comprehension stream floor when participant exceeded expectations
L2_COMP_FLOOR: float = 45.0

# Semantic similarity threshold for insight deduplication
DEDUP_SIMILARITY_THRESHOLD: float = 0.85

import unittest

import pandas as pd

from scoring_engine.aggregation import compute_project_scores
from scoring_engine.l1_scorer import score_l1
from scoring_engine.l2_scorer import score_l2
from scoring_engine.l3_scorer import score_l3
from scoring_engine.models import FrictionInsight, SurveyResponse
from scoring_engine.preprocess import deduplicate_insights


def insight(
    friction_type: str,
    severity: str,
    confidence: float = 1.0,
    text: str = "The participant hit the same underlying barrier.",
) -> FrictionInsight:
    return FrictionInsight(
        participant_id="p1",
        task_id="task1",
        insight_text=text,
        friction_type=friction_type,
        severity=severity,
        confidence=confidence,
    )


class KatieModelRulesTest(unittest.TestCase):
    def test_l1_accessibility_block_is_categorical_failure(self) -> None:
        survey = SurveyResponse(
            participant_id="p1",
            task_id="task1",
            ease_of_use_score=100,
            met_expectations="met",
            cohort="general",
        )

        score, is_blocked, eligible_floor = score_l1(
            survey,
            [insight("accessibility", "severe", 0.95)],
        )

        self.assertEqual(score, 0.0)
        self.assertTrue(is_blocked)
        self.assertFalse(eligible_floor)

    def test_l2_exceeded_expectations_floor_only_limits_comprehension(self) -> None:
        score = score_l2(
            l1_score=90.0,
            is_blocked=False,
            insights=[
                insight("comprehension", "medium"),
                insight("comprehension", "medium"),
                insight("comprehension", "medium"),
            ],
            met_expectations="exceeded",
            cohort="esl",
        )

        self.assertEqual(score, 45.0)

    def test_l3_time_cap_applies_after_other_usability_rules(self) -> None:
        score = score_l3(
            l1_score=90.0,
            is_blocked=False,
            insights=[],
            met_expectations="met",
            cohort="general",
            timing_ratio=3.2,
        )

        self.assertEqual(score, 40.0)

    def test_project_blocked_participants_contribute_zero_to_accessibility(self) -> None:
        scores_df = pd.DataFrame([
            {
                "participant_id": "p1",
                "task_id": "task1",
                "cohort": "general",
                "is_blocked": False,
                "l2_score": 80.0,
                "l3_score": 75.0,
            },
            {
                "participant_id": "p2",
                "task_id": "task1",
                "cohort": "general",
                "is_blocked": False,
                "l2_score": 70.0,
                "l3_score": 65.0,
            },
        ])
        project_blocked_df = pd.DataFrame({
            "participant_id": [" P2 ", "p3"],
            "blocked": ["YES", "true"],
        })

        result = compute_project_scores(scores_df, project_blocked_df).iloc[0]

        self.assertEqual(result["n_participants_acc"], 3)
        self.assertEqual(result["n_participants_usa"], 2)
        self.assertAlmostEqual(result["accessibility_mean"], 26.67, places=2)
        self.assertEqual(result["usability_mean"], 70.0)

    def test_dedup_keeps_distinct_friction_types_separate(self) -> None:
        df = pd.DataFrame([
            {
                "participant_id": "p1",
                "task_id": "task1",
                "insight_text": "The same checkout error blocked progress.",
                "friction_type": "accessibility",
                "severity": "low",
                "confidence": 0.8,
            },
            {
                "participant_id": "p1",
                "task_id": "task1",
                "insight_text": "The same checkout error blocked progress.",
                "friction_type": "usability",
                "severity": "high",
                "confidence": 0.9,
            },
            {
                "participant_id": "p1",
                "task_id": "task1",
                "insight_text": "The same checkout error blocked progress.",
                "friction_type": "usability",
                "severity": "medium",
                "confidence": 0.85,
            },
        ])

        deduped = deduplicate_insights(df)
        usability = deduped[deduped["friction_type"] == "usability"].iloc[0]

        self.assertEqual(len(deduped), 2)
        self.assertEqual(usability["severity"], "high")
        self.assertEqual(usability["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()

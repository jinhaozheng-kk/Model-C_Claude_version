from dataclasses import dataclass


@dataclass
class FrictionInsight:
    participant_id: str
    task_id: str
    insight_text: str
    friction_type: str   # accessibility / comprehension / usability / other
    severity: str        # blocked / severe / high / medium / low
    confidence: float    # 0.0–1.0


@dataclass
class SurveyResponse:
    participant_id: str
    task_id: str
    ease_of_use_score: int   # 0–100 self-reported score (L1 ceiling)
    met_expectations: str    # exceeded / met / not met / fell well below
    cohort: str              # esl / deaf / neurodivergent / blind_lv / older_adult / general


@dataclass
class TimingRecord:
    participant_id: str
    task_id: str
    time_seconds: float


@dataclass
class ParticipantTaskScore:
    participant_id: str
    task_id: str
    cohort: str
    ease_of_use_score: int
    is_blocked: bool
    l1_score: float
    l2_score: float   # accessibility
    l3_score: float   # usability

"""
Data preparation adapter for AAMI website user testing data.

Reads:
  data/A_Test_Raw_Survey_Scores_AAMI Insurance Website - Website.csv
  data/AAMI website - Individual Insights - 2026-03-10.xlsx

Writes to data/prepared/:
  survey.csv        — one row per (participant × task), ready for scoring engine
  insights.csv      — friction insights, filtered and normalised
  project_blocked.csv  — participants with project-level AT blockers (manual review recommended)
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

# Survey source column → insights task_title (exact strings from data)
SURVEY_TASK_MAP: dict[str, str] = {
    "How user-friendly was the AAMI Homepage?": "Navigating AAMI’s Insurance Service Offering",
    "How easy was it to find and read a product disclosure statement?": "PDFs and Understanding Terms",
    "How easy and clear was the information on Natural disasters?":
        "Equal Access to Critical  Information – Emergency Response Page",
    "How easy was it to navigate around the website?": "Navigation of sub-landing page- Product Information",
}

# Tasks in insights without a dedicated survey task column → use overall usability score (col 7 × 20)
OVERALL_SCORE_TASKS: set[str] = {
    "AAMI Website",
    "Navigating information on financial hardship",
    "Discovering insurance providers",
    "Navigating Frequently Asked Questions",
    "Starting a Home and Contents Insurance Quote",
    "Thanks for listening AAMI",
}

FINDING_ANSWERS_TASK = "Finding Answers"

SURVEY_EMAIL_COL = "Please enter your email address"
SURVEY_TESTER_TYPE_COL = "What type of tester are you? (Select all that apply)"
SURVEY_OVERALL_USABILITY_COL = "Generally speaking, how user friendly, inuitive, helpful was the AAMI website to use?"
SURVEY_SEARCH_COL = "How user-friendly was the search function?"
SURVEY_LIVE_CHAT_COL = "How user friendly was the live chat?"
SURVEY_MET_EXPECTATIONS_COL = "How well did the AAMI  website meet your expectations?"

REQUIRED_SURVEY_COLUMNS = {
    SURVEY_EMAIL_COL,
    SURVEY_TESTER_TYPE_COL,
    SURVEY_OVERALL_USABILITY_COL,
    SURVEY_SEARCH_COL,
    SURVEY_LIVE_CHAT_COL,
    SURVEY_MET_EXPECTATIONS_COL,
    *SURVEY_TASK_MAP.keys(),
}
REQUIRED_INSIGHTS_COLUMNS = {
    "tester_email",
    "tester_cohorts",
    "task_title",
    "category",
    "severity",
    "friction_type",
    "title",
    "description",
    "confidence",
}

# Participants who submitted survey with a different email than the insights pipeline used
EMAIL_ALIAS: dict[str, str] = {
    "asmaosyan9@gmail.com":     "asma@civikpeople.com.au",       # Asma Osyan (ESL)
    "mattymc2@gmail.com":       "info@smproductions.com.au",      # Matthew McLaren (BL)
    "tdcn66@gmail.com":         "marychaunguyen@hotmail.com",     # Thi Diem Chau Nguyen (BL)
    # troyparnell@hotmail.com has survey but no insights — keep as-is
}

# Cohort priority for multi-code participants (e.g. "BL,ND" → blind_lv)
COHORT_PRIORITY: list[str] = ["BL", "LV", "DF", "ESL", "ND", "OL", "AB", "OT"]
COHORT_CODE_MAP: dict[str, str] = {
    "BL": "blind_lv",
    "LV": "blind_lv",
    "DF": "deaf",
    "ESL": "esl",
    "ND": "neurodivergent",
    "OL": "older_adult",
    "AB": "general",
    "OT": "general",
}

# friction_type values in insights → scoring engine friction_type
FRICTION_TYPE_MAP: dict[str, str] = {
    "accessibility":        "accessibility",
    "comprehension":        "comprehension",
    "content_not_found":    "usability",
    "confidence":           "usability",
    "excessive_effort":     "usability",
    "unexpected_behaviour": "usability",
    "unresponsive_interface": "usability",
}

# met_expectations free-text → canonical values used by scoring engine
MET_EXPECTATIONS_MAP: dict[str, str] = {
    "exceeded my expectations":    "exceeded",
    "met my expectations":         "met",
    "partly met my expectations":  "not met",
    "did not meet my expectations": "fell well below",
}

# Survey tester_type text → cohort (fallback when participant not in insights)
def _cohort_from_survey_text(tester_type: str) -> str:
    t = tester_type.lower()
    if "blind" in t:              return "blind_lv"
    if "low vision" in t:         return "blind_lv"
    if "deaf" in t:               return "deaf"
    if "non-english" in t:        return "esl"
    if "older" in t:              return "older_adult"
    if "neurodivergent" in t:     return "neurodivergent"
    return "general"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_email(email: str) -> str:
    e = str(email).strip().lower()
    return EMAIL_ALIAS.get(e, e)


def _map_cohort(cohort_codes: str) -> str:
    codes = [c.strip().upper() for c in str(cohort_codes).split(",")]
    for code in COHORT_PRIORITY:
        if code in codes:
            return COHORT_CODE_MAP[code]
    return "general"


def _map_met_expectations(value: str) -> str:
    v = str(value).strip().lower()
    for key, mapped in MET_EXPECTATIONS_MAP.items():
        if v.startswith(key):
            return mapped
    return "met"


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _score_to_100(value: object, column: str) -> int:
    score = pd.to_numeric(value, errors="raise")
    if pd.isna(score) or score < 0 or score > 5:
        raise ValueError(f"{column} must contain a 0-5 score.")
    return round(float(score) * 20)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("data/prepared", exist_ok=True)

    survey_raw  = pd.read_csv(
        "data/A_Test_Raw_Survey_Scores_AAMI Insurance Website - Website.csv"
    )
    insights_raw = pd.read_excel(
        "data/AAMI website - Individual Insights - 2026-03-10.xlsx"
    )

    _require_columns(survey_raw, REQUIRED_SURVEY_COLUMNS, "raw survey CSV")
    _require_columns(insights_raw, REQUIRED_INSIGHTS_COLUMNS, "raw insights XLSX")

    all_tasks: list[str] = sorted(insights_raw["task_title"].unique().tolist())

    # ---- Build participant cohort lookup from insights ----------------------
    cohort_lookup: dict[str, str] = {}
    for email, grp in insights_raw.groupby("tester_email"):
        canonical = _canonical_email(email)
        cohort_lookup[canonical] = _map_cohort(grp.iloc[0]["tester_cohorts"])

    # ---- Survey → long format (one row per participant × task) -------------
    survey_rows: list[dict] = []

    for _, row in survey_raw.iterrows():
        email = _canonical_email(row[SURVEY_EMAIL_COL])
        met_exp = _map_met_expectations(row[SURVEY_MET_EXPECTATIONS_COL])
        overall_score = _score_to_100(row[SURVEY_OVERALL_USABILITY_COL], SURVEY_OVERALL_USABILITY_COL)

        # Per-task ceilings (1-5 -> 0-100)
        task_ceilings: dict[str, int] = {}
        for column, task_title in SURVEY_TASK_MAP.items():
            task_ceilings[task_title] = _score_to_100(row[column], column)

        # Finding Answers = average of search and live chat scores.
        search_score = pd.to_numeric(row[SURVEY_SEARCH_COL], errors="raise")
        live_chat_score = pd.to_numeric(row[SURVEY_LIVE_CHAT_COL], errors="raise")
        if pd.isna(search_score) or search_score < 0 or search_score > 5:
            raise ValueError(f"{SURVEY_SEARCH_COL} must contain a 0-5 score.")
        if pd.isna(live_chat_score) or live_chat_score < 0 or live_chat_score > 5:
            raise ValueError(f"{SURVEY_LIVE_CHAT_COL} must contain a 0-5 score.")
        task_ceilings[FINDING_ANSWERS_TASK] = round((search_score + live_chat_score) / 2 * 20)

        cohort = cohort_lookup.get(email) or _cohort_from_survey_text(str(row[SURVEY_TESTER_TYPE_COL]))

        for task in all_tasks:
            survey_rows.append({
                "participant_id":    email,
                "task_id":           task,
                "ease_of_use_score": task_ceilings.get(task, overall_score),
                "met_expectations":  met_exp,
                "cohort":            cohort,
            })

    survey_df = pd.DataFrame(survey_rows)
    survey_df.to_csv("data/prepared/survey.csv", index=False)
    print(f"survey.csv        : {len(survey_df):>5} rows  "
          f"({survey_raw.shape[0]} participants × {len(all_tasks)} tasks)")

    # ---- Insights → friction only, normalised ------------------------------
    category = insights_raw["category"].astype(str).str.strip().str.lower()
    friction = insights_raw[category.isin(["friction", "blocker"])].copy()

    friction["participant_id"] = friction["tester_email"].apply(_canonical_email)
    friction["task_id"]        = friction["task_title"]
    friction["insight_text"]   = (
        friction["title"].fillna("") + ". " + friction["description"].fillna("")
    )
    friction["friction_type"]  = (
        friction["friction_type"]
        .str.strip().str.lower()
        .map(lambda x: FRICTION_TYPE_MAP.get(x, "usability"))
    )
    # Severity is already set in the data (severe/high/medium/low);
    # category="blocker" entries have severity="severe" which triggers L1.2
    # only when friction_type="accessibility" (per spec L1.2).
    friction["severity"]       = friction["severity"].str.strip().str.lower()
    friction["confidence"]     = friction["confidence"].astype(float)

    insights_out = friction[[
        "participant_id", "task_id", "insight_text",
        "friction_type", "severity", "confidence",
    ]].reset_index(drop=True)

    insights_out.to_csv("data/prepared/insights.csv", index=False)
    print(f"insights.csv      : {len(insights_out):>5} rows  "
          f"(friction + blocker categories only)")

    # ---- Project-level blocked participants --------------------------------
    # Participants with an AT accessibility blocker on the overall "AAMI Website"
    # task are candidates for project-level blocking (manual review recommended).
    category = insights_raw["category"].astype(str).str.strip().str.lower()
    friction_type = insights_raw["friction_type"].astype(str).str.strip().str.lower()
    at_site_blockers = insights_raw[
        (category == "blocker") &
        (friction_type == "accessibility") &
        (insights_raw["task_title"] == "AAMI Website")
    ]["tester_email"].apply(_canonical_email).unique()

    pb_df = pd.DataFrame({"participant_id": at_site_blockers, "blocked": True})
    pb_df.to_csv("data/prepared/project_blocked.csv", index=False)
    print(f"project_blocked.csv: {len(pb_df):>4} participants  "
          f"(AT blocker on overall AAMI Website task — review before using)")

    # ---- Summary -----------------------------------------------------------
    print()
    print("Task ceiling source summary:")
    print(f"  {'Task':<58} {'Source'}")
    print(f"  {'-'*58} {'------'}")
    for task in all_tasks:
        if task == FINDING_ANSWERS_TASK:
            src = "avg(search, live_chat) × 20"
        elif task in task_ceilings:
            src = "survey task score × 20"
        else:
            src = "overall usability score × 20"
        print(f"  {task:<58} {src}")


if __name__ == "__main__":
    main()

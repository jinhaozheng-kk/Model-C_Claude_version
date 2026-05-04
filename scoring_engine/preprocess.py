import pandas as pd
import numpy as np

from .constants import DEDUP_SIMILARITY_THRESHOLD, SEVERITY_RANK


def _pairwise_similarity(texts: list[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix over texts.
    Uses TF-IDF if scikit-learn is available, falls back to SequenceMatcher ratio.
    """
    n = len(texts)
    if n <= 1:
        return np.eye(n)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(min_df=1).fit_transform(texts)
        return cosine_similarity(vec)
    except (ImportError, ValueError):
        from difflib import SequenceMatcher

        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                mat[i, j] = mat[j, i] = ratio
        return mat


def _merge_similar_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.reset_index(drop=True)

    if len(group) == 1:
        return group

    texts = group["insight_text"].fillna("").astype(str).tolist()
    sim = _pairwise_similarity(texts)

    parent = list(range(len(group)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if sim[i, j] > DEDUP_SIMILARITY_THRESHOLD:
                union(i, j)

    components: dict[int, list[int]] = {}
    for idx in range(len(group)):
        components.setdefault(find(idx), []).append(idx)

    kept: list[pd.Series] = []
    for indexes in components.values():
        row = group.iloc[indexes[0]].copy()
        most_severe_idx = max(
            indexes,
            key=lambda i: SEVERITY_RANK.get(str(group.iloc[i]["severity"]).lower(), 0),
        )
        row["severity"] = group.iloc[most_severe_idx]["severity"]
        row["confidence"] = max(float(group.iloc[i]["confidence"]) for i in indexes)
        kept.append(row)

    return pd.DataFrame(kept)


def deduplicate_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (participant_id, task_id) group, merge insight pairs whose
    insight_text is >85% semantically similar.  The merged insight keeps the
    higher severity and the higher confidence value from the pair.

    Runs before any scoring rule is applied (Pre-Processing step in spec).
    """
    if df.empty:
        return df.copy()

    result_parts: list[pd.DataFrame] = []

    group_cols = ["participant_id", "task_id"]
    if "friction_type" in df.columns:
        group_cols.append("friction_type")

    for _, group in df.groupby(group_cols, sort=False):
        result_parts.append(_merge_similar_group(group))

    return pd.concat(result_parts, ignore_index=True)

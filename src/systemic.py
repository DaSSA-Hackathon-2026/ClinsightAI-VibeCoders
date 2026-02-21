from __future__ import annotations

import numpy as np
import pandas as pd


def compute_recurrence_metrics(chunks: pd.DataFrame) -> pd.DataFrame:
    total_reviews = max(chunks["review_id"].nunique(), 1)

    out = (
        chunks.groupby("cluster_id")
        .agg(
            n_chunks=("chunk_text", "count"),
            n_reviews=("review_id", "nunique"),
            avg_severity=("severity_score", "mean"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    out["frequency_percentage"] = (out["n_chunks"] / max(len(chunks), 1)) * 100
    out["review_coverage_percentage"] = (out["n_reviews"] / total_reviews) * 100
    out["complaint_repetition_index"] = out["n_chunks"] / out["n_reviews"].clip(lower=1)
    return out


def label_issue_scope(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scope = []
    for _, row in out.iterrows():
        cov = row["review_coverage_percentage"]
        rep = row["complaint_repetition_index"]
        if cov >= 12 and rep >= 1.2:
            scope.append("systemic")
        elif cov >= 5:
            scope.append("recurring")
        else:
            scope.append("isolated")
    out["issue_scope"] = scope
    return out


def combine_with_impact(
    recurrence_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    theme_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    out = recurrence_df.merge(
        theme_summary_df[["cluster_id", "theme_label"]],
        on="cluster_id",
        how="left",
    ).drop_duplicates("cluster_id")

    out = out.merge(
        impact_df[["cluster_id", "rating_impact_coef", "confidence"]],
        on="cluster_id",
        how="left",
    )
    out["impact_abs"] = out["rating_impact_coef"].abs()
    return out


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    freq_n = _minmax(out["frequency_percentage"])
    sev_n = _minmax(out["avg_severity"])
    impact_n = _minmax(out["impact_abs"].fillna(0.0))
    cov_n = _minmax(out["review_coverage_percentage"])

    out["risk_score"] = (
        0.35 * freq_n
        + 0.25 * sev_n
        + 0.25 * impact_n
        + 0.15 * cov_n
    ).clip(0, 1)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


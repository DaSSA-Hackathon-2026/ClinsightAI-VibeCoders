from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_output_json(
    reviews_df: pd.DataFrame,
    theme_risk_df: pd.DataFrame,
    roadmap_df: pd.DataFrame,
    evidence_map: dict[int, list[str]],
    top_risk_n: int = 3,
    top_theme_n: int = 10,
) -> dict:
    top_risk = theme_risk_df.nlargest(top_risk_n, "risk_score")["theme_label"].astype(str).tolist()

    clinic_summary = {
        "overall_rating_mean": round(float(reviews_df["rating"].mean()), 2),
        "review_count": int(len(reviews_df)),
        "primary_risk_themes": top_risk,
    }

    theme_rows = theme_risk_df.sort_values("risk_score", ascending=False).head(top_theme_n)
    theme_analysis: list[dict] = []
    for _, row in theme_rows.iterrows():
        cid = int(row["cluster_id"])
        theme_analysis.append(
            {
                "theme": str(row["theme_label"]),
                "frequency_percentage": round(float(row["frequency_percentage"]), 2),
                "review_coverage_percentage": round(float(row["review_coverage_percentage"]), 2),
                "rating_impact": round(float(row.get("rating_impact_coef", np.nan)), 3),
                "severity_score": round(float(row["avg_severity"]), 3),
                "risk_score": round(float(row["risk_score"]), 3),
                "issue_scope": str(row["issue_scope"]),
                "impact_confidence": round(float(row.get("impact_confidence", np.nan)), 3),
                "cluster_confidence": round(float(row.get("cluster_confidence", np.nan)), 3),
                "confidence": round(float(row.get("confidence", np.nan)), 3),
                "evidence_samples": evidence_map.get(cid, [])[:3],
            }
        )

    improvement_roadmap: list[dict] = []
    for _, row in roadmap_df.iterrows():
        improvement_roadmap.append(
            {
                "priority": int(row["priority"]),
                "theme": str(row["theme_label"]),
                "recommendation": str(row["recommendation"]),
                "workstream": str(row["workstream"]),
                "suggested_kpi": str(row["suggested_kpi"]),
                "expected_rating_lift": f"+{float(row['expected_rating_lift']):.2f}",
                "confidence": round(float(row["confidence"]), 3),
                "effort": str(row["effort"]),
            }
        )

    return {
        "clinic_summary": clinic_summary,
        "theme_analysis": theme_analysis,
        "improvement_roadmap": improvement_roadmap,
    }


def _versioned_path(path: Path) -> Path:
    if not path.exists():
        return path

    base_dir = path.parent
    stem = path.stem
    suffix = path.suffix or ".json"

    i = 1
    while True:
        candidate = base_dir / f"{stem}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def save_json(payload: dict, path: str | Path, version_if_exists: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if version_if_exists:
        target = _versioned_path(target)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return target

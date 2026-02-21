from __future__ import annotations

import re

import numpy as np
import pandas as pd


PLAYBOOK = [
    {
        "pattern": r"wait|delay|queue|appointment",
        "recommendation": "Optimize appointment scheduling and queue triage workflow.",
        "kpi": "Median wait time, on-time appointment rate",
        "effort": "medium",
    },
    {
        "pattern": r"front desk|reception|staff|rude|behavior",
        "recommendation": "Run service-behavior training and front-desk SLA monitoring.",
        "kpi": "Front-desk satisfaction score, complaint resolution time",
        "effort": "low",
    },
    {
        "pattern": r"billing|charge|cost|package|insurance",
        "recommendation": "Improve billing transparency with pre-visit estimate communication.",
        "kpi": "Billing-related complaints, estimate accuracy",
        "effort": "medium",
    },
    {
        "pattern": r"clean|hygiene|housekeeping|dirty|infection",
        "recommendation": "Increase housekeeping audit frequency and ward cleanliness checklist coverage.",
        "kpi": "Cleanliness audit score, housekeeping response SLA",
        "effort": "medium",
    },
    {
        "pattern": r"communication|explained|clarity|inform",
        "recommendation": "Standardize patient communication scripts and discharge instructions.",
        "kpi": "Post-visit understanding score, readmission due to instruction gaps",
        "effort": "low",
    },
]


def _playbook_match(theme_label: str) -> dict[str, str]:
    text = str(theme_label).lower()
    for rule in PLAYBOOK:
        if re.search(rule["pattern"], text):
            return {
                "recommendation": rule["recommendation"],
                "kpi": rule["kpi"],
                "effort": rule["effort"],
            }
    return {
        "recommendation": "Investigate root cause with cross-functional ops review.",
        "kpi": "Theme-specific complaint rate and resolution SLA",
        "effort": "high",
    }


def build_improvement_roadmap(theme_risk_df: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
    out = theme_risk_df.copy()
    out["confidence"] = out["confidence"].fillna(0.55).clip(0, 1)

    actions = out["theme_label"].apply(_playbook_match)
    out["recommendation"] = actions.apply(lambda d: d["recommendation"])
    out["suggested_kpi"] = actions.apply(lambda d: d["kpi"])
    out["effort"] = actions.apply(lambda d: d["effort"])
    out["effort_score"] = out["effort"].map({"low": 1, "medium": 2, "high": 3}).fillna(3)

    out["priority_score"] = (
        0.5 * out["risk_score"]
        + 0.3 * out["impact_abs"].fillna(0)
        + 0.2 * out["confidence"]
    )

    out["expected_rating_lift"] = np.where(
        out["rating_impact_coef"] < 0,
        0.4 + out["impact_abs"].clip(0, 1.2) * 0.6,
        0.2 + out["impact_abs"].clip(0, 1.0) * 0.3,
    )
    out["workstream"] = np.where(
        (out["effort_score"] <= 2) & (out["priority_score"] >= out["priority_score"].median()),
        "quick_win",
        "high_effort",
    )

    out = out.sort_values(["priority_score", "risk_score"], ascending=False).head(top_n).copy()
    out["priority"] = np.arange(1, len(out) + 1)
    return out[
        [
            "priority",
            "cluster_id",
            "theme_label",
            "recommendation",
            "workstream",
            "suggested_kpi",
            "expected_rating_lift",
            "confidence",
            "effort",
            "priority_score",
        ]
    ]


from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="ClinsightAI Report Dashboard",
    page_icon=":hospital:",
    layout="wide",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.replace("+", "").strip()
        return float(value)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_files(outputs_dir: str = "outputs") -> list[Path]:
    root = Path(outputs_dir)
    if not root.exists():
        return []
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def theme_df_from_report(report: dict) -> pd.DataFrame:
    rows = report.get("theme_analysis", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "rating_impact" in df.columns:
        df["impact_abs"] = df["rating_impact"].apply(_safe_float).abs()
    if "evidence_samples" in df.columns:
        df["evidence_preview"] = df["evidence_samples"].apply(
            lambda x: " | ".join(x[:2]) if isinstance(x, list) else ""
        )
    return df


def roadmap_df_from_report(report: dict) -> pd.DataFrame:
    rows = report.get("improvement_roadmap", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("priority")
    if "expected_rating_lift" in df.columns:
        df["expected_rating_lift_num"] = df["expected_rating_lift"].apply(_safe_float)
    return df


st.title("ClinsightAI: Healthcare Review Intelligence Dashboard")
st.caption("Structured operational themes, impact, risk, and action roadmap from report JSON files.")

files = report_files("outputs")
if not files:
    st.error("No JSON report files found in `outputs/`.")
    st.stop()

default_idx = 0
selected = st.sidebar.selectbox(
    "Choose report file",
    options=files,
    index=default_idx,
    format_func=lambda p: p.name,
)
st.sidebar.markdown("### Filters")
min_conf = st.sidebar.slider("Min Theme Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
scope_filter = st.sidebar.multiselect(
    "Issue Scope",
    options=["isolated", "recurring", "systemic"],
    default=["isolated", "recurring", "systemic"],
)
top_n = st.sidebar.slider("Top Themes (by risk)", min_value=5, max_value=30, value=12, step=1)

report = load_report(str(selected))
summary = report.get("clinic_summary", {})
diagnostics = report.get("diagnostics", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Rating Mean", f"{_safe_float(summary.get('overall_rating_mean')):.2f}")
c2.metric("Review Count", str(summary.get("review_count", "-")))
c3.metric("CV MAE", f"{_safe_float(diagnostics.get('cv_mae')):.3f}")
c4.metric("Global Cluster Confidence", f"{_safe_float(diagnostics.get('global_cluster_confidence')):.3f}")

st.subheader("Primary Risk Themes")
primary = summary.get("primary_risk_themes", [])
if primary:
    st.write(", ".join(primary))
else:
    st.write("No primary risk themes found.")

theme_df = theme_df_from_report(report)
roadmap_df = roadmap_df_from_report(report)
if not theme_df.empty:
    if "confidence" in theme_df.columns:
        theme_df = theme_df[theme_df["confidence"].apply(_safe_float) >= min_conf]
    if "issue_scope" in theme_df.columns and scope_filter:
        theme_df = theme_df[theme_df["issue_scope"].isin(scope_filter)]
    if "risk_score" in theme_df.columns:
        theme_df = theme_df.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)

tab0, tab1, tab2, tab3 = st.tabs(["Pipeline Flow", "Theme Analysis", "Roadmap", "Diagnostics"])

with tab0:
    st.subheader("Execution Flowchart")
    st.caption("Vertical step-by-step view of the end-to-end pipeline.")

    steps = [
        ("1) Ingest Reviews", "feedback + rating (+optional metadata)"),
        ("2) Clean & Normalize", "column mapping, text cleanup, dedupe"),
        ("3) Sentence Chunking", "one review -> multiple issue chunks"),
        ("4) Embeddings", "SentenceTransformer or TF-IDF + SVD fallback"),
        ("5) Theme Clustering", "MiniBatchKMeans + silhouette-based K"),
        ("6) Theme Labeling", "TF-IDF keyphrases + evidence sentences"),
        ("7) Severity Scoring", "HF sentiment or dataset-derived phrase lexicon"),
        ("8) Review-Theme Matrix", "intensity = frequency * severity"),
        ("9) Impact Modeling", "Ridge (CV MAE) + 1v5 Logistic (AUC)"),
        ("10) Robustness", "bootstrap coefficient confidence"),
        ("11) Systemic Detection", "coverage + repetition -> isolated/recurring/systemic"),
        ("12) Risk Scoring", "frequency + severity + impact + coverage"),
        ("13) Action Roadmap", "priority, quick wins, KPIs, expected lift"),
        ("14) Explainable JSON Output", "summary + themes + roadmap + diagnostics"),
    ]

    for i, (title, desc) in enumerate(steps):
        st.markdown(
            f"""
            <div style="
                max-width: 760px;
                margin: 0 auto;
                background: #eaf4ff;
                border: 1px solid #b9d8ff;
                border-radius: 12px;
                padding: 14px 18px;
                text-align: center;
                color: #000000;
            ">
                <div style="font-weight: 700; font-size: 16px; margin-bottom: 6px; color: #000000;">{title}</div>
                <div style="font-size: 14px; color: #000000;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if i < len(steps) - 1:
            st.markdown("<div style='text-align:center; font-size:30px;'>↓</div>", unsafe_allow_html=True)

    with st.expander("Why this flow is used"):
        st.write(
            "Sentence-level chunking keeps mixed-topic reviews interpretable, embeddings cluster by meaning, "
            "impact models quantify rating effect, and recurrence+risk logic converts analytics into operations decisions."
        )

with tab1:
    st.subheader("Theme Table")
    if theme_df.empty:
        st.info("No theme analysis data found.")
    else:
        show_cols = [
            "theme",
            "frequency_percentage",
            "review_coverage_percentage",
            "rating_impact",
            "severity_score",
            "risk_score",
            "issue_scope",
            "impact_confidence",
            "cluster_confidence",
            "confidence",
            "evidence_preview",
        ]
        present_cols = [c for c in show_cols if c in theme_df.columns]
        st.dataframe(theme_df[present_cols], use_container_width=True, hide_index=True)

        st.subheader("Decision-Critical Visuals")
        chart_df = theme_df.copy()
        chart_df["theme_short"] = chart_df["theme"].astype(str).str.slice(0, 42)
        chart_df["impact_abs"] = chart_df["rating_impact"].apply(_safe_float).abs()

        left, right = st.columns(2)
        with left:
            st.markdown("Theme Priority Matrix (Impact vs Frequency)")
            priority_scatter = (
                alt.Chart(chart_df)
                .mark_circle(opacity=0.85)
                .encode(
                    x=alt.X("impact_abs:Q", title="Absolute Rating Impact"),
                    y=alt.Y("frequency_percentage:Q", title="Theme Frequency (%)"),
                    size=alt.Size("severity_score:Q", title="Severity", scale=alt.Scale(range=[60, 800])),
                    color=alt.Color("risk_score:Q", title="Risk Score", scale=alt.Scale(scheme="redyellowgreen")),
                    tooltip=[
                        "theme",
                        "issue_scope",
                        alt.Tooltip("impact_abs:Q", format=".3f"),
                        alt.Tooltip("frequency_percentage:Q", format=".2f"),
                        alt.Tooltip("severity_score:Q", format=".3f"),
                        alt.Tooltip("risk_score:Q", format=".3f"),
                        alt.Tooltip("confidence:Q", format=".3f"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(priority_scatter, use_container_width=True)
        with right:
            st.markdown("Risk Contribution (Pareto View)")
            pareto = chart_df.sort_values("risk_score", ascending=False).copy()
            pareto["cum_risk_pct"] = pareto["risk_score"].cumsum() / pareto["risk_score"].sum() * 100
            bars = (
                alt.Chart(pareto)
                .mark_bar()
                .encode(
                    x=alt.X("theme_short:N", sort=None, title="Theme"),
                    y=alt.Y("risk_score:Q", title="Risk Score"),
                    tooltip=["theme", alt.Tooltip("risk_score:Q", format=".3f"), alt.Tooltip("cum_risk_pct:Q", format=".1f")],
                )
            )
            line = (
                alt.Chart(pareto)
                .mark_line(point=True, color="#d62728")
                .encode(
                    x=alt.X("theme_short:N", sort=None),
                    y=alt.Y("cum_risk_pct:Q", title="Cumulative Risk (%)"),
                )
            )
            st.altair_chart((bars + line).resolve_scale(y="independent").properties(height=360), use_container_width=True)

        left2, right2 = st.columns(2)
        with left2:
            st.markdown("Confidence Robustness (Impact vs Cluster)")
            if "impact_confidence" in chart_df.columns and "cluster_confidence" in chart_df.columns:
                conf_scatter = (
                    alt.Chart(chart_df)
                    .mark_circle(opacity=0.85)
                    .encode(
                        x=alt.X("impact_confidence:Q", scale=alt.Scale(domain=[0, 1]), title="Impact Confidence"),
                        y=alt.Y("cluster_confidence:Q", scale=alt.Scale(domain=[0, 1]), title="Cluster Confidence"),
                        size=alt.Size("impact_abs:Q", title="Absolute Impact", scale=alt.Scale(range=[80, 900])),
                        color=alt.Color("issue_scope:N", title="Issue Scope"),
                        tooltip=["theme", "issue_scope", "confidence", "impact_confidence", "cluster_confidence"],
                    )
                    .properties(height=340)
                )
                st.altair_chart(conf_scatter, use_container_width=True)
        with right2:
            st.markdown("Issue Scope Distribution")
            scope_counts = chart_df["issue_scope"].value_counts().reset_index()
            scope_counts.columns = ["issue_scope", "count"]
            scope_bar = (
                alt.Chart(scope_counts)
                .mark_bar()
                .encode(
                    x=alt.X("issue_scope:N", title="Issue Scope"),
                    y=alt.Y("count:Q", title="Theme Count"),
                    color=alt.Color("issue_scope:N", legend=None),
                    tooltip=["issue_scope", "count"],
                )
                .properties(height=340)
            )
            st.altair_chart(scope_bar, use_container_width=True)

        with st.expander("Evidence Samples by Theme"):
            for _, row in theme_df.iterrows():
                st.markdown(f"**{row.get('theme', 'Theme')}**")
                samples = row.get("evidence_samples", [])
                if isinstance(samples, list) and samples:
                    for s in samples[:3]:
                        st.write(f"- {s}")
                else:
                    st.write("- No evidence samples")

with tab2:
    st.subheader("Improvement Roadmap")
    if roadmap_df.empty:
        st.info("No roadmap data found.")
    else:
        road_cols = [
            "priority",
            "theme",
            "recommendation",
            "workstream",
            "suggested_kpi",
            "expected_rating_lift",
            "confidence",
            "effort",
        ]
        road_present = [c for c in road_cols if c in roadmap_df.columns]
        st.dataframe(roadmap_df[road_present], use_container_width=True, hide_index=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("Expected Rating Lift by Priority")
            lift_bar = (
                alt.Chart(roadmap_df)
                .mark_bar()
                .encode(
                    x=alt.X("priority:O", title="Priority"),
                    y=alt.Y("expected_rating_lift_num:Q", title="Expected Rating Lift"),
                    color=alt.Color("workstream:N", title="Workstream"),
                    tooltip=["priority", "theme", "workstream", "expected_rating_lift", "confidence"],
                )
                .properties(height=320)
            )
            st.altair_chart(lift_bar, use_container_width=True)
        with c_right:
            st.markdown("Execution Mix: Quick Wins vs High Effort")
            exec_mix = (
                roadmap_df.groupby("workstream", as_index=False)
                .agg(
                    n_actions=("workstream", "count"),
                    avg_lift=("expected_rating_lift_num", "mean"),
                    avg_confidence=("confidence", "mean"),
                )
            )
            mix_bar = (
                alt.Chart(exec_mix)
                .mark_bar()
                .encode(
                    x=alt.X("workstream:N", title="Workstream"),
                    y=alt.Y("n_actions:Q", title="Number of Actions"),
                    color=alt.Color("workstream:N", legend=None),
                    tooltip=["workstream", "n_actions", alt.Tooltip("avg_lift:Q", format=".2f"), alt.Tooltip("avg_confidence:Q", format=".2f")],
                )
                .properties(height=320)
            )
            st.altair_chart(mix_bar, use_container_width=True)

with tab3:
    st.subheader("Result JSON")
    st.caption("Complete output payload for the selected run.")
    st.json(report)

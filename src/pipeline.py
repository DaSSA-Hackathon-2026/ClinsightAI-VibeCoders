from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.impact import (
    bootstrap_ridge_coefficients,
    build_impact_table,
    build_review_theme_matrix,
    fit_extreme_classifier,
    fit_rating_impact_model,
)
from src.preprocess import build_chunks, preprocess_reviews
from src.report import build_output_json, save_json
from src.roadmap import build_improvement_roadmap
from src.systemic import (
    combine_with_impact,
    compute_recurrence_metrics,
    compute_risk_scores,
    label_issue_scope,
)
from src.themes import (
    add_severity_scores,
    auto_label_clusters,
    cluster_confidence_table,
    choose_cluster_count,
    cluster_chunks,
    compute_embeddings,
    summarize_themes,
    theme_evidence_map,
)


def run_pipeline(
    csv_path: str | Path,
    output_json_path: str | Path | None = None,
    n_clusters: int | None = None,
    severity_method: str = "auto",
    version_output: bool = True,
) -> dict:
    raw = pd.read_csv(csv_path)
    reviews = preprocess_reviews(raw)
    chunks = build_chunks(reviews)

    embeddings = compute_embeddings(chunks["chunk_text"].astype(str).tolist())
    if n_clusters is None:
        n_clusters = choose_cluster_count(embeddings)

    chunks = cluster_chunks(chunks, embeddings, n_clusters=n_clusters)
    cluster_conf_df, global_cluster_conf = cluster_confidence_table(embeddings, chunks["cluster_id"])
    chunks = add_severity_scores(chunks, method=severity_method)
    chunks, _ = auto_label_clusters(
        chunks,
        extra_stop_words=["hospital", "doctor", "doctors", "staff", "service", "patient"],
    )

    theme_summary = summarize_themes(chunks)
    evidence = theme_evidence_map(chunks, n=5)

    X, y = build_review_theme_matrix(chunks)
    reg = fit_rating_impact_model(X, y)
    boot = bootstrap_ridge_coefficients(X, y, n_boot=200)
    impact_table = build_impact_table(reg.coefficients, boot)

    recurrence = compute_recurrence_metrics(chunks)
    recurrence = label_issue_scope(recurrence)
    theme_risk = combine_with_impact(recurrence, impact_table, theme_summary)
    theme_risk = theme_risk.merge(cluster_conf_df, on="cluster_id", how="left")
    theme_risk["cluster_confidence"] = theme_risk["cluster_confidence"].fillna(0.5)
    theme_risk["impact_confidence"] = theme_risk["confidence"].fillna(0.5)
    theme_risk["confidence"] = (0.65 * theme_risk["impact_confidence"] + 0.35 * theme_risk["cluster_confidence"]).clip(0, 1)
    theme_risk = compute_risk_scores(theme_risk)

    roadmap = build_improvement_roadmap(theme_risk, top_n=7)
    payload = build_output_json(reviews, theme_risk, roadmap, evidence)

    diagnostics = {
        "n_reviews": int(len(reviews)),
        "n_chunks": int(len(chunks)),
        "n_clusters": int(n_clusters),
        "cv_mae": round(reg.cv_mae, 4),
        "extreme_auc": None,
        "global_cluster_confidence": round(global_cluster_conf, 4),
    }
    extreme = fit_extreme_classifier(X, y)
    if extreme is not None:
        diagnostics["extreme_auc"] = round(extreme["auc"], 4)
    payload["diagnostics"] = diagnostics

    if output_json_path is not None:
        saved_path = save_json(payload, output_json_path, version_if_exists=version_output)
        payload["output_path"] = str(saved_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClinsightAI analytics pipeline.")
    parser.add_argument("--input", required=True, help="Path to review CSV")
    parser.add_argument("--output", default="outputs/report.json", help="Path to JSON output")
    parser.add_argument("--clusters", type=int, default=None, help="Optional fixed cluster count")
    parser.add_argument(
        "--severity-method",
        default="auto",
        choices=["auto", "hf", "rule"],
        help="Severity scoring mode",
    )
    parser.add_argument(
        "--no-version-output",
        action="store_true",
        help="Overwrite output file instead of creating report_001/report_002 versions.",
    )
    args = parser.parse_args()

    payload = run_pipeline(
        csv_path=args.input,
        output_json_path=args.output,
        n_clusters=args.clusters,
        severity_method=args.severity_method,
        version_output=not args.no_version_output,
    )
    print(json.dumps(payload["diagnostics"], indent=2))
    print(f"Saved output to: {payload.get('output_path', args.output)}")


if __name__ == "__main__":
    main()

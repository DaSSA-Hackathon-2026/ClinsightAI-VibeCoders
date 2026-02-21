from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionResult:
    cv_mae: float
    coefficients: pd.Series
    model: object


def build_review_theme_matrix(chunks: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    totals = chunks.groupby("review_id").size().rename("n_chunks")

    freq = (
        chunks.groupby(["review_id", "cluster_id"])
        .size()
        .div(totals)
        .rename("freq")
        .reset_index()
    )
    sev = (
        chunks.groupby(["review_id", "cluster_id"])["severity_score"]
        .mean()
        .rename("sev")
        .reset_index()
    )
    long_df = freq.merge(sev, on=["review_id", "cluster_id"], how="left")
    long_df["intensity"] = long_df["freq"] * long_df["sev"]

    X = long_df.pivot_table(
        index="review_id",
        columns="cluster_id",
        values="intensity",
        fill_value=0.0,
    )
    X.columns = [f"theme_{c}_intensity" for c in X.columns]
    y = chunks.groupby("review_id")["rating"].first().loc[X.index]
    return X, y


def fit_rating_impact_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> RegressionResult:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeCV(alphas=np.logspace(-3, 3, 25)),
    )
    n_splits = max(3, min(5, len(y) // 50))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_mae = float(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean())
    model.fit(X, y)
    ridge = model.named_steps["ridgecv"]
    coefficients = pd.Series(ridge.coef_, index=X.columns).sort_values()
    return RegressionResult(cv_mae=cv_mae, coefficients=coefficients, model=model)


def fit_extreme_classifier(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict[str, float] | None:
    mask = y.isin([1, 5])
    if mask.sum() < 80:
        return None

    y_ext = (y[mask] == 1).astype(int)
    if y_ext.value_counts().min() < 20:
        return None

    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    auc = float(cross_val_score(clf, X[mask], y_ext, cv=cv, scoring="roc_auc").mean())
    clf.fit(X[mask], y_ext)

    coefs = pd.Series(clf.named_steps["logisticregression"].coef_[0], index=X.columns)
    return {"auc": auc, "max_positive_coef": float(coefs.max()), "max_negative_coef": float(coefs.min())}


def _feature_to_cluster_id(feature: str) -> int:
    m = re.match(r"theme_(\d+)_intensity", str(feature))
    if not m:
        raise ValueError(f"Unexpected feature name: {feature}")
    return int(m.group(1))


def bootstrap_ridge_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = len(X)
    columns = list(X.columns)
    coefs = np.zeros((n_boot, len(columns)))
    alphas = np.logspace(-3, 3, 25)

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reg = RidgeCV(alphas=alphas)
        reg.fit(X_np[idx], y_np[idx])
        coefs[b] = reg.coef_

    mean = coefs.mean(axis=0)
    lo = np.quantile(coefs, 0.025, axis=0)
    hi = np.quantile(coefs, 0.975, axis=0)
    sign_stability = np.mean(np.sign(coefs) == np.sign(mean), axis=0)
    ci_width = hi - lo
    precision = np.abs(mean) / (np.abs(mean) + ci_width + 1e-9)
    crosses_zero = (lo <= 0) & (hi >= 0)
    confidence = 0.6 * sign_stability + 0.4 * precision
    confidence = np.where(crosses_zero, confidence * 0.6, confidence)
    confidence = np.clip(confidence, 0.05, 0.99)

    out = pd.DataFrame(
        {
            "feature": columns,
            "coef_mean": mean,
            "coef_ci_low": lo,
            "coef_ci_high": hi,
            "sign_stability": sign_stability,
            "precision": precision,
            "confidence": confidence,
        }
    )
    out["cluster_id"] = out["feature"].apply(_feature_to_cluster_id)
    return out


def build_impact_table(
    coef_series: pd.Series,
    bootstrap_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = (
        pd.DataFrame(
            {
                "feature": coef_series.index,
                "rating_impact_coef": coef_series.values,
            }
        )
        .assign(cluster_id=lambda d: d["feature"].apply(_feature_to_cluster_id))
        .sort_values("rating_impact_coef")
        .reset_index(drop=True)
    )
    if bootstrap_table is None:
        base["confidence"] = np.nan
        return base

    cols = ["feature", "confidence", "coef_ci_low", "coef_ci_high"]
    out = base.merge(bootstrap_table[cols], on="feature", how="left")
    return out

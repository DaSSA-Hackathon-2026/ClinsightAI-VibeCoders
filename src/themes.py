from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import TruncatedSVD


NEGATIVE_TERMS = {
    "bad",
    "poor",
    "delay",
    "delayed",
    "waiting",
    "waited",
    "rude",
    "unprofessional",
    "dirty",
    "painful",
    "expensive",
    "overcharged",
    "confusing",
    "ignored",
    "crowded",
    "worst",
    "disappointed",
    "mismanaged",
}

_RULE_NEGATIVE_TERMS: set[str] = set(NEGATIVE_TERMS)
_RULE_POSITIVE_TERMS: set[str] = set()
DOMAIN_NEUTRAL_TERMS = {
    "hospital",
    "patient",
    "patients",
    "doctor",
    "doctors",
    "staff",
    "clinic",
    "medical",
    "care",
    "treatment",
    "service",
}


def derive_sentiment_phrases_from_reviews(
    chunks: pd.DataFrame,
    top_k: int = 120,
    min_df: int = 3,
) -> tuple[set[str], set[str]]:
    """
    Learn domain-specific negative and positive phrases from review text.
    Uses rating<=2 as negative, rating>=4 as positive.
    """
    if "rating" not in chunks.columns or "chunk_text" not in chunks.columns:
        return set(NEGATIVE_TERMS), set()

    data = chunks[["chunk_text", "rating"]].dropna().copy()
    if len(data) < 80:
        return set(NEGATIVE_TERMS), set()

    neg_mask = data["rating"] <= 2
    pos_mask = data["rating"] >= 4
    if neg_mask.sum() < 20 or pos_mask.sum() < 20:
        return set(NEGATIVE_TERMS), set()

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=15000,
    )
    X = vec.fit_transform(data["chunk_text"].astype(str))
    features = vec.get_feature_names_out()
    if len(features) == 0:
        return set(NEGATIVE_TERMS), set()

    neg_scores = np.asarray(X[neg_mask.values].mean(axis=0)).ravel()
    pos_scores = np.asarray(X[pos_mask.values].mean(axis=0)).ravel()
    neg_delta = neg_scores - pos_scores
    pos_delta = pos_scores - neg_scores

    def _top_terms(delta: np.ndarray) -> set[str]:
        idx = delta.argsort()[::-1]
        terms: list[str] = []
        for i in idx:
            t = str(features[i]).strip().lower()
            if delta[i] <= 0:
                break
            if any(ch.isdigit() for ch in t):
                continue
            if len(t) < 3:
                continue
            if t in DOMAIN_NEUTRAL_TERMS:
                continue
            terms.append(t)
            if len(terms) >= top_k:
                break
        return set(terms)

    neg_terms = _top_terms(neg_delta)
    pos_terms = _top_terms(pos_delta)
    if not neg_terms:
        neg_terms = set(NEGATIVE_TERMS)
    return neg_terms, pos_terms


def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
    except Exception:
        # Offline-safe fallback for hackathon environments with restricted downloads.
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        X = tfidf.fit_transform(texts)
        n_comp = min(256, max(16, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        emb = svd.fit_transform(X)
        if normalize_embeddings:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
        return emb


def choose_cluster_count(
    embeddings: np.ndarray,
    k_min: int = 6,
    k_max: int = 18,
    random_state: int = 42,
) -> int:
    n_rows = embeddings.shape[0]
    if n_rows < 100:
        return max(3, min(8, n_rows // 10 or 3))

    sample_size = min(n_rows, 3000)
    if n_rows > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_rows, size=sample_size, replace=False)
        X = embeddings[idx]
    else:
        X = embeddings

    best_k = k_min
    best_score = -1.0
    k_max = min(k_max, max(k_min, int(math.sqrt(n_rows)) + 5))

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=2048)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_chunks(
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> pd.DataFrame:
    out = chunks.copy()
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048)
    out["cluster_id"] = km.fit_predict(embeddings)
    return out


def cluster_confidence_table(
    embeddings: np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, float]:
    labels = np.asarray(cluster_ids)
    unique = np.unique(labels)
    if len(unique) < 2:
        table = pd.DataFrame({"cluster_id": unique.astype(int), "cluster_confidence": [0.5] * len(unique)})
        return table, 0.0

    sil = silhouette_samples(embeddings, labels)
    sil_01 = np.clip((sil + 1.0) / 2.0, 0.0, 1.0)
    global_score = float(np.clip((sil.mean() + 1.0) / 2.0, 0.0, 1.0))

    out = (
        pd.DataFrame({"cluster_id": labels, "cluster_confidence": sil_01})
        .groupby("cluster_id", as_index=False)["cluster_confidence"]
        .mean()
    )
    out["cluster_id"] = out["cluster_id"].astype(int)
    return out, global_score


def _rule_based_severity(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0

    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    terms = set(tokens) | set(bigrams)

    neg_terms = _RULE_NEGATIVE_TERMS if _RULE_NEGATIVE_TERMS else set(NEGATIVE_TERMS)
    pos_terms = _RULE_POSITIVE_TERMS

    neg_hits = sum(1 for t in terms if t in neg_terms)
    pos_hits = sum(1 for t in terms if t in pos_terms)
    ratio = (neg_hits - 0.6 * pos_hits) / max(len(tokens), 1)

    if "!" in text:
        ratio += 0.03
    if "very" in tokens or "extremely" in tokens:
        ratio += 0.03
    return float(min(1.0, max(0.0, ratio * 8)))


def add_severity_scores(chunks: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    out = chunks.copy()
    texts = out["chunk_text"].astype(str).tolist()

    use_hf = method in {"auto", "hf"}
    if use_hf:
        try:
            from transformers import pipeline

            clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            scores: list[float] = []
            for txt in texts:
                result = clf(txt[:512])[0]
                if result["label"] == "NEGATIVE":
                    scores.append(float(result["score"]))
                else:
                    scores.append(float(1.0 - result["score"]))
            out["severity_score"] = scores
            return out
        except Exception:
            if method == "hf":
                raise

    global _RULE_NEGATIVE_TERMS, _RULE_POSITIVE_TERMS
    _RULE_NEGATIVE_TERMS, _RULE_POSITIVE_TERMS = derive_sentiment_phrases_from_reviews(out)
    out["severity_score"] = out["chunk_text"].astype(str).apply(_rule_based_severity)
    return out


def top_keywords(
    texts: Iterable[str],
    k: int = 5,
    max_features: int = 8000,
    extra_stop_words: list[str] | None = None,
) -> list[str]:
    stop_words = list(extra_stop_words or [])
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    matrix = vec.fit_transform(list(texts))
    vocab = vec.get_feature_names_out()
    if len(vocab) == 0:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    idx = scores.argsort()[::-1]
    terms: list[str] = []
    for i in idx:
        t = vocab[i]
        if t in stop_words:
            continue
        terms.append(t)
        if len(terms) >= k:
            break
    return terms


def auto_label_clusters(
    chunks: pd.DataFrame,
    k_keywords: int = 4,
    extra_stop_words: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    out = chunks.copy()
    cluster_names: dict[int, str] = {}
    for cid in sorted(out["cluster_id"].unique()):
        texts = out.loc[out["cluster_id"] == cid, "chunk_text"].astype(str).tolist()
        kws = top_keywords(texts, k=k_keywords, extra_stop_words=extra_stop_words)
        cluster_names[int(cid)] = " / ".join(kws) if kws else f"theme_{cid}"
    out["theme_label"] = out["cluster_id"].map(cluster_names)
    return out, cluster_names


def theme_evidence_map(chunks: pd.DataFrame, n: int = 5) -> dict[int, list[str]]:
    grouped = (
        chunks.groupby("cluster_id")["chunk_text"]
        .apply(lambda s: s.head(n).tolist())
        .to_dict()
    )
    return {int(k): v for k, v in grouped.items()}


def summarize_themes(chunks: pd.DataFrame) -> pd.DataFrame:
    out = (
        chunks.groupby(["cluster_id", "theme_label"], as_index=False)
        .agg(
            n_chunks=("chunk_text", "count"),
            n_reviews=("review_id", "nunique"),
            avg_severity=("severity_score", "mean"),
            avg_rating=("rating", "mean"),
        )
        .sort_values("n_chunks", ascending=False)
        .reset_index(drop=True)
    )
    total_chunks = max(len(chunks), 1)
    total_reviews = max(chunks["review_id"].nunique(), 1)
    out["frequency_percentage"] = (out["n_chunks"] / total_chunks) * 100
    out["review_coverage_percentage"] = (out["n_reviews"] / total_reviews) * 100
    out["severity_score"] = out["avg_severity"].clip(0, 1)
    return out

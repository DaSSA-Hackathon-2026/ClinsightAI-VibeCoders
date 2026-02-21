from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


COLUMN_MAP = {
    "feedback": "feedback",
    "review": "feedback",
    "text": "feedback",
    "comment": "feedback",
    "ratings": "rating",
    "rating": "rating",
    "sentiment label": "sentiment_label",
    "sentiment_label": "sentiment_label",
    "sentiment": "sentiment_label",
    "date": "date",
    "clinic_id": "clinic_id",
    "hospital_id": "clinic_id",
}


def standardize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=lambda c: standardize_column_name(str(c)))
    out = out.rename(columns={c: COLUMN_MAP.get(c.replace("_", " "), c) for c in out.columns})
    out = out.rename(columns={c: COLUMN_MAP.get(c, c) for c in out.columns})
    return out


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("Â", " ").replace("Ã‚", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess_reviews(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    out = normalize_columns(df)

    required = ["feedback", "rating"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(out.columns)}")

    out["feedback_raw"] = out["feedback"].astype(str)
    out["feedback"] = out["feedback_raw"].apply(clean_text)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna(subset=["rating"])
    out["rating"] = out["rating"].astype(int)
    out = out[(out["rating"] >= 1) & (out["rating"] <= 5)]
    out = out[out["feedback"].str.len() > 0]

    if deduplicate:
        dedup_keys = ["feedback", "rating"]
        if "clinic_id" in out.columns:
            dedup_keys.append("clinic_id")
        out = out.drop_duplicates(subset=dedup_keys)

    out = out.reset_index(drop=True)
    out["review_id"] = out.index
    return out


def split_into_sentences(text: str, min_len: int = 4) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if not sentences:
        return [str(text).strip()]
    return sentences


def build_chunks(df: pd.DataFrame, text_col: str = "feedback_raw") -> pd.DataFrame:
    records = []
    optional_cols: Iterable[str] = ["sentiment_label", "date", "clinic_id"]

    for _, row in df.iterrows():
        sentences = split_into_sentences(str(row.get(text_col, "")))
        for sent in sentences:
            item = {
                "review_id": row["review_id"],
                "chunk_text": sent.strip(),
                "rating": row["rating"],
            }
            for col in optional_cols:
                if col in df.columns:
                    item[col] = row.get(col)
            records.append(item)

    return pd.DataFrame(records)


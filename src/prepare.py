"""Cleaning + feature engineering for the Instagram engagement dataset."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "instagram_data.csv"
OUT = ROOT / "data" / "processed" / "instagram_clean.csv"

SOURCES = ["from_home", "from_hashtags", "from_explore", "from_other"]
INTERACTIONS = ["likes", "comments", "shares", "saves"]

# Ordered keyword rules: the first matching topic wins.
TOPIC_RULES = [
    ("Project Ideas & Portfolio", r"project|source code"),
    ("Career & Learning Resources",
     r"roadmap|websites|books|resources|certification|interview|skills|channels|"
     r"soft skills|tools|data sources|libraries|topics"),
    ("Machine Learning & Deep Learning",
     r"machine learning|deep learning|neural|lstm|clustering|algorithm|regression|"
     r"activation|nlp|natural language|tokenization|language detection|sentiment|"
     r"train a machine|recommendation"),
    ("Data Viz & Statistics",
     r"visuali|plotting|boxplot|candlestick|variance|time series|time-series"),
    ("Industry Use Cases",
     r"zomato|amazon|google|facebook|finance|ukraine|crypto|payment|stress|news|"
     r"applications of|real world|programming languages|databases"),
    ("Python Tutorials & Scripts",
     r"python|otp|qr code|email|translator|count|execution time|whatsapp"),
]


def classify_topic(caption: str) -> str:
    text = caption.lower()
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, text):
            return topic
    return "Other"


def load_raw() -> pd.DataFrame:
    # The source file is Windows-1252 encoded (curly apostrophes etc.)
    return pd.read_csv(RAW, encoding="latin-1")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(df)}
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 1. text clean-up: broken apostrophes and stray whitespace
    for col in ("caption", "hashtags"):
        df[col] = (df[col].astype(str)
                   .str.replace("\x92", "'", regex=False)
                   .str.replace(r"\s+", " ", regex=True).str.strip())

    # 2. remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["exact_duplicates_removed"] = before - len(df)
    log["clean_rows"] = len(df)
    log["missing_values"] = int(df.isna().sum().sum())

    # 3. traffic-source reconciliation check
    df["source_total"] = df[SOURCES].sum(axis=1)
    df["source_gap"] = df["impressions"] - df["source_total"]
    log["posts_where_sources_exceed_impressions"] = int((df["source_gap"] < 0).sum())

    # 4. features
    df["post_id"] = np.arange(1, len(df) + 1)
    df["interactions"] = df[INTERACTIONS].sum(axis=1)
    df["engagement_rate"] = df["interactions"] / df["impressions"] * 100
    df["save_rate"] = df["saves"] / df["impressions"] * 100
    df["share_rate"] = df["shares"] / df["impressions"] * 100
    df["profile_visit_rate"] = df["profile_visits"] / df["impressions"] * 100
    df["follow_rate"] = df["follows"] / df["profile_visits"].replace(0, np.nan) * 100
    for s in SOURCES:
        df[f"{s}_pct"] = df[s] / df["source_total"] * 100
    df["hashtag_count"] = df["hashtags"].str.count("#")
    df["caption_length"] = df["caption"].str.len()
    df["caption_words"] = df["caption"].str.split().str.len()
    df["has_link_in_bio"] = df["caption"].str.contains("link in bio", case=False)
    df["topic"] = df["caption"].apply(classify_topic)
    df["viral_flag"] = df["impressions"] >= df["impressions"].quantile(0.90)
    df["impression_tier"] = pd.qcut(df["impressions"], 3, labels=["Low", "Medium", "High"])
    return df, log


def main() -> pd.DataFrame:
    df, log = clean(load_raw())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    for k, v in log.items():
        print(f"{k}: {v}")
    print(f"saved -> {OUT.relative_to(ROOT)}")
    return df


if __name__ == "__main__":
    main()

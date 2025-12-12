import re
import pandas as pd
from pathlib import Path

SIDER_DIR = Path("data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_se():
    """Load side effect mappings from SIDER database."""
    cols = [
        "stitch_flat",
        "stitch_stereo",
        "umls_label_id",
        "term_type",
        "se_id",
        "se_name",
    ]
    df = pd.read_csv(SIDER_DIR / "meddra_all_se.tsv", sep="\t", names=cols)
    
    # Keep only Preferred Terms (PT) to remove redundant Lower Level Terms (LLT)
    df = df[df["term_type"] == "PT"].copy()
    
    # Extract PubChem CID from STITCH identifier
    df["cid"] = df["stitch_flat"].str.extract(r"CID10*([1-9]\d*)$")
    df = df.dropna(subset=["cid", "se_id"])
    df["cid"] = df["cid"].astype(str)
    df["se_id"] = df["se_id"].astype(str)
    
    df = df.drop_duplicates(subset=["cid", "se_id"])
    return df[["cid", "se_id", "se_name"]]


def load_freq():
    """Load side effect frequency data from SIDER database."""
    cols = [
        "stitch_flat",
        "stitch_stereo",
        "umls_label_id",
        "placebo",
        "freq_text",
        "freq_lo",
        "freq_hi",
        "term_type",
        "se_id",
        "se_name",
    ]
    df = pd.read_csv(SIDER_DIR / "meddra_freq.tsv", sep="\t", names=cols)
    
    # Keep PT rows to align with side effect mapping
    df = df[df["term_type"] == "PT"].copy()
    df["cid"] = df["stitch_flat"].str.extract(r"CID10*([1-9]\d*)$")
    df = df.dropna(subset=["cid", "se_id"])
    df["cid"] = df["cid"].astype(str)
    df["se_id"] = df["se_id"].astype(str)
    
    # Normalize placebo flag to binary
    df["placebo"] = (df["placebo"].astype(str).str.lower() == "placebo").astype(int)
    
    # Convert frequency bounds to numeric
    for c in ["freq_lo", "freq_hi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Aggregate multiple entries per (cid, se_id) pair
    # Prefer non-placebo entries and take max frequency as worst-case signal
    agg = (
        df.sort_values("placebo")
        .groupby(["cid", "se_id"], as_index=False)
        .agg(
            freq_lo=("freq_lo", "min"),
            freq_hi=("freq_hi", "max"),
            freq_text=("freq_text", lambda s: s.value_counts(dropna=False).index[0]),
            placebo=("placebo", "min")
        )
    )
    return agg


def main(include_frequency=True):
    se = load_se()
    if include_frequency:
        freq = load_freq()
        edges = se.merge(freq, on=["cid", "se_id"], how="left")
    else:
        edges = se
    edges = edges.drop_duplicates(subset=["cid", "se_id"])
    edges = edges.sort_values(["cid", "se_id"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "edges.csv"
    edges.to_csv(out_path, index=False)
    print(f"wrote {len(edges):,} edges to {out_path}")


if __name__ == "__main__":
    main(include_frequency=True)

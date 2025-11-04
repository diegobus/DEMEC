import re
import pandas as pd
from pathlib import Path

SIDER_DIR = Path("data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_se():
    # meddra_all_se.tsv.gz columns per SIDER doc:
    # 1: stitch_flat, 2: stitch_stereo, 3: umls_label_id,
    # 4: term_type (LLT/PT), 5: umls_meddra_id, 6: side_effect_name
    cols = [
        "stitch_flat",
        "stitch_stereo",
        "umls_label_id",
        "term_type",
        "se_id",
        "se_name",
    ]
    df = pd.read_csv(SIDER_DIR / "meddra_all_se.tsv", sep="\t", names=cols)
    # keep only PT (preferred terms) to remove LLTs
    df = df[df["term_type"] == "PT"].copy()
    # extract PubChem CID digits from STITCH id
    df["cid"] = df["stitch_flat"].str.extract(r"(\d+)$")
    # drop rows with missing cid or se_id
    df = df.dropna(subset=["cid", "se_id"])
    # enforce string type
    df["cid"] = df["cid"].astype(str)
    df["se_id"] = df["se_id"].astype(str)
    # deduplicate drug→PT mappings
    df = df.drop_duplicates(subset=["cid", "se_id"])
    return df[["cid", "se_id", "se_name"]]


def load_freq():
    # meddra_freq.tsv.gz columns per SIDER doc:
    # 1: stitch_flat, 2: stitch_stereo, 3: umls_label_id, 4: placebo flag,
    # 5: freq_text, 6: freq_lo, 7: freq_hi, 8: term_type, 9: se_id, 10: se_name
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
    # keep PT rows to align with PT mapping
    df = df[df["term_type"] == "PT"].copy()
    df["cid"] = df["stitch_flat"].str.extract(r"(\d+)$")
    df = df.dropna(subset=["cid", "se_id"])
    df["cid"] = df["cid"].astype(str)
    df["se_id"] = df["se_id"].astype(str)
    # normalize placebo flag to boolean-ish
    df["placebo"] = (df["placebo"].astype(str).str.lower() == "placebo").astype(int)
    # coerce numeric bounds
    for c in ["freq_lo", "freq_hi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # There can be multiple rows per (cid, se_id) — aggregate:
    # prefer non-placebo entries; take the max upper bound as a “worst-case”/most frequent signal,
    # and keep a representative freq_text (e.g., the most common)
    agg = (
        df.sort_values("placebo")  # non-placebo first
        .groupby(["cid", "se_id"], as_index=False)
        .agg(
            freq_lo=("freq_lo", "min"),
            freq_hi=("freq_hi", "max"),
            # pick the most frequent textual category among the kept rows
            freq_text=("freq_text", lambda s: s.value_counts(dropna=False).index[0]),
            placebo=("placebo", "min"),  # 0 if any non-placebo exists
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

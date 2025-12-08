import pandas as pd
import sys
import os

def to_level3_list(codes):
    """Extract unique level-3 (4 chars) ATC codes."""
    l3 = set()
    for c in codes:
        if len(c) >= 4:
            l3.add(c[:4])
    return list(l3)

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/smiles_cache.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cid_atc_l3_matrix.csv"

    print(f"Loading from {input_path}...")
    df = pd.read_csv(input_path)
    
    df['atc'] = df['atc'].fillna('').astype(str)
    
    df['atc_list'] = df['atc'].apply(
        lambda s: [code.strip() for code in s.split(';') if code.strip()]
    )

    df['atc_l3_list'] = df['atc_list'].apply(to_level3_list)

    exploded = df[['cid', 'atc_l3_list']].explode('atc_l3_list')
    exploded = exploded.dropna(subset=['atc_l3_list'])

    if exploded.empty:
        print("Warning: No valid ATC Level-3 codes found. Matrix will be empty.")
    
    exploded['value'] = 1
    cid_atc_l3_matrix = (
        exploded
        .pivot_table(
            index='cid',
            columns='atc_l3_list',
            values='value',
            fill_value=0
        )
        .reset_index()
    )

    cid_atc_l3_matrix.columns = [
        "cid" if c == "cid" else f"atc_{c}"
        for c in cid_atc_l3_matrix.columns
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cid_atc_l3_matrix.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape: {cid_atc_l3_matrix.shape}")

if __name__ == "__main__":
    main()

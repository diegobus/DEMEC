import os, sys
import numpy as np
import pandas as pd

def compile_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Convert edge list to binary matrix for GNN model.
    
    Args:
        edges: DataFrame with 'cid' and 'se_id' columns
        
    Returns:
        Binary matrix with drugs as rows and side effects as columns
    """
    edges["value"] = 1 

    mat = (
        edges.pivot_table(
            index="cid",
            columns="se_id",
            values="value",
            fill_value=0,
            aggfunc="max"
        )
        .astype(int)
    )

    return mat.reset_index()


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python scripts/model_targets.py <input_edges.csv> <output_labels.csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    edges = pd.read_csv(input_path, dtype={"cid": str, "se_id": str})
    mat = compile_table(edges)
    mat.to_csv(output_path, index=False)
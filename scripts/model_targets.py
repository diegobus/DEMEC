import os, sys
import numpy as np
import pandas as pd

def compile_table(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Convert edge list to a binary matrix for GNN model.
    Rows are unique CIDs, columns are unique SE_IDs. 
    """
    
    # Can play with this later to make it depend on the frequency
    # Now, it is a binary yes/no side effect listed
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


# CLI
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python src/scripts/compile_table.py <input_edges.csv> <output_labels.csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    edges = pd.read_csv(input_path)
    mat = compile_table(edges)
    mat.to_csv(output_path, index=False)
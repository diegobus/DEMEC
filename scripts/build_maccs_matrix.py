import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import sys
import os

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/smiles_cache.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cid_maccs_matrix.csv"

    print(f"Loading from {input_path}...")
    df = pd.read_csv(input_path)

    rows = []

    for _, row in df.iterrows():
        cid = row['cid']
        smi = row['smiles_sanitized']

        if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: could not parse SMILES for cid={cid}, smiles='{smi}'")
            continue

        fp = MACCSkeys.GenMACCSKeys(mol)
        bitstring = fp.ToBitString()
        
        bits = np.fromiter(bitstring[1:], dtype=int)

        row_dict = {'cid': cid}
        for i, b in enumerate(bits, start=1):
            row_dict[f"macc_{i}"] = int(b)

        rows.append(row_dict)

    cid_maccs_matrix = pd.DataFrame(rows)
    cid_maccs_matrix = cid_maccs_matrix.sort_values('cid').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cid_maccs_matrix.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape: {cid_maccs_matrix.shape}")

if __name__ == "__main__":
    main()

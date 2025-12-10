#!/usr/bin/env python3
"""
Extract molecular properties from SMILES for regression task.

Properties extracted:
- Molecular Weight (MW)
- LogP (lipophilicity)
- TPSA (Topological Polar Surface Area)
- Number of H-bond donors
- Number of H-bond acceptors
- Number of rotatable bonds
- Number of aromatic rings
- Number of aliphatic rings
- Fraction of sp3 carbons
- Number of heavy atoms

Usage:
    python scripts/extract_molecular_properties.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def calculate_molecular_properties(smiles):
    """
    Calculate molecular properties from SMILES string.
    
    Returns:
        dict: Dictionary of property name -> value
        None: If SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        'MW': Descriptors.MolWt(mol),                           # Molecular weight
        'LogP': Descriptors.MolLogP(mol),                       # Lipophilicity
        'TPSA': Descriptors.TPSA(mol),                          # Topological polar surface area
        'HBD': Descriptors.NumHDonors(mol),                     # H-bond donors
        'HBA': Descriptors.NumHAcceptors(mol),                  # H-bond acceptors
        'RotBonds': Descriptors.NumRotatableBonds(mol),         # Rotatable bonds
        'AromaticRings': Descriptors.NumAromaticRings(mol),     # Aromatic rings
        'AliphaticRings': Descriptors.NumAliphaticRings(mol),   # Aliphatic rings
        'FractionCSP3': Descriptors.FractionCSP3(mol),          # Fraction sp3 carbons
        'HeavyAtoms': Descriptors.HeavyAtomCount(mol),          # Heavy atoms
        'MolMR': Descriptors.MolMR(mol),                        # Molar refractivity
        'NumRings': Descriptors.RingCount(mol),                 # Total rings
    }
    
    return properties


def main():
    """Extract molecular properties for all drugs in SMILES cache."""
    
    # Paths
    smiles_cache = "data/processed/smiles_cache.csv"
    output_file = "data/processed/cid_molprops_matrix.csv"
    
    print("=" * 80)
    print("Extracting Molecular Properties")
    print("=" * 80)
    print(f"Input: {smiles_cache}")
    print(f"Output: {output_file}")
    print()
    
    # Load SMILES cache
    if not os.path.exists(smiles_cache):
        print(f"Error: SMILES cache not found: {smiles_cache}")
        print("Please run build_molecular_graphs.py first")
        sys.exit(1)
    
    df = pd.read_csv(smiles_cache)
    print(f"Loaded {len(df)} compounds from SMILES cache")
    print()
    
    # Extract properties
    print("Calculating molecular properties...")
    results = []
    failed = 0
    
    for idx, row in df.iterrows():
        cid = row['cid']
        smiles = row.get('smiles_sanitized') or row.get('smiles_raw')
        
        if not smiles or pd.isna(smiles):
            failed += 1
            continue
        
        props = calculate_molecular_properties(smiles)
        
        if props is None:
            failed += 1
            continue
        
        # Add CID to properties
        props['cid'] = cid
        results.append(props)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} compounds...")
    
    print(f"  Processed {len(df)}/{len(df)} compounds")
    print()
    
    # Convert to DataFrame
    props_df = pd.DataFrame(results)
    
    # Reorder columns (CID first)
    cols = ['cid'] + [c for c in props_df.columns if c != 'cid']
    props_df = props_df[cols]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    props_df.to_csv(output_file, index=False)
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total compounds: {len(df)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {failed}")
    print()
    
    # Print statistics
    print("Property Statistics:")
    print("-" * 80)
    for col in props_df.columns:
        if col == 'cid':
            continue
        print(f"  {col:20s}: mean={props_df[col].mean():8.2f}, std={props_df[col].std():8.2f}, "
              f"min={props_df[col].min():8.2f}, max={props_df[col].max():8.2f}")
    print()
    
    print(f"DONE: Molecular properties saved to: {output_file}")
    print()
    
    # Show sample
    print("Sample (first 5 compounds):")
    print(props_df.head())
    print()


if __name__ == '__main__':
    main()

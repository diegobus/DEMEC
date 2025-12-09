"""
Build molecular graphs from SIDER drug data with enhanced node features.

This script:
1. Fetches SMILES from PubChem (with caching)
2. Converts SMILES to NetworkX graphs with RDKit features
3. Saves graphs as pickles for training

Usage:
    python scripts/build_molecular_graphs.py data/raw/drug_names.tsv data/raw/drug_atc.tsv
"""

import os
import sys
import csv
import re
import time
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import networkx as nx
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ============================================================================
# Configuration
# ============================================================================

CID_RE = re.compile(r"CID10*([1-9]\d*)$")

# Categorical variables for one-hot encoding
ATOM_LIST = list(range(1, 119))  # atomic numbers 1..118
CHIRALITY_LIST = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
HYBRID_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


@dataclass
class Drug:
    """Container for drug information."""
    cid: str
    name: str
    atc: Optional[str] = None
    smiles_raw: Optional[str] = None
    smiles_sanitized: Optional[str] = None
    graph_path: Optional[str] = None


# ============================================================================
# SMILES Cache Management
# ============================================================================

class SMILESCache:
    """Manages persistent cache of CID -> SMILES mappings."""
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache = self._load()
    
    def _load(self) -> Dict[str, Dict[str, str]]:
        """Load cache from CSV file."""
        if not os.path.exists(self.cache_path):
            return {}
        
        cache = {}
        with open(self.cache_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["cid"]] = row
        return cache
    
    def get(self, cid: str) -> Optional[Dict[str, str]]:
        """Retrieve cached entry for CID."""
        return self._cache.get(cid)
    
    def add(self, cid: str, name: str, atc: str, smiles_raw: str, smiles_sanitized: str):
        """Add new entry to cache."""
        row = {
            "cid": cid,
            "name": name,
            "atc": atc or "",
            "smiles_raw": smiles_raw or "",
            "smiles_sanitized": smiles_sanitized or "",
            "ts": str(int(time.time())),
        }
        
        # Append to file
        file_exists = os.path.exists(self.cache_path)
        with open(self.cache_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["cid", "name", "atc", "smiles_raw", "smiles_sanitized", "ts"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        # Update in-memory cache
        self._cache[cid] = row


# ============================================================================
# PubChem Utilities
# ============================================================================

def parse_cid(raw: str) -> Optional[str]:
    """Extract numeric CID from PubChem identifier string."""
    m = CID_RE.search(raw.strip())
    return m.group(1) if m else None


def fetch_smiles_from_pubchem(cid: str) -> Optional[str]:
    """Fetch SMILES string from PubChem API."""
    try:
        c = pcp.Compound.from_cid(int(cid))
        return (
            getattr(c, "isomeric_smiles", None)
            or getattr(c, "canonical_smiles", None)
            or getattr(c, "smiles", None)
        )
    except Exception as e:
        print(f"  Warning: Failed to fetch SMILES for CID {cid}: {e}")
        return None


def sanitize_smiles(smiles: Optional[str]) -> Optional[str]:
    """
    Sanitize SMILES string by removing stereochemistry.
    This ensures compatibility with various parsers.
    """
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Non-isomeric removes / and \ around double bonds
    return Chem.MolToSmiles(mol, isomericSmiles=False)


# ============================================================================
# Feature Engineering
# ============================================================================

def one_hot(x, choices: list) -> list:
    """One-hot encode a value given a list of choices."""
    vec = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(x)
    except ValueError:
        idx = len(choices)  # unknown category
    vec[idx] = 1
    return vec


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    """
    Extract comprehensive atom features for GNN.
    
    Features (154-dim):
    - Atomic number (one-hot, 119-dim)
    - Degree (one-hot, 7-dim)
    - Formal charge (one-hot, 6-dim)
    - Valence (one-hot, 8-dim)
    - Aromatic (1-dim)
    - In ring (1-dim)
    - Num hydrogens (1-dim)
    - Hybridization (one-hot, 6-dim)
    - Chirality (one-hot, 5-dim)
    """
    feats = []
    
    # Element
    feats += one_hot(atom.GetAtomicNum(), ATOM_LIST)
    
    # Degree, charge, valence
    feats += one_hot(atom.GetTotalDegree(), list(range(0, 6)))
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += one_hot(atom.GetTotalValence(), list(range(0, 7)))
    
    # Boolean features
    feats.append(int(atom.GetIsAromatic()))
    feats.append(int(atom.IsInRing()))
    feats.append(atom.GetTotalNumHs(includeNeighbors=True))
    
    # Hybridization and chirality
    feats += one_hot(atom.GetHybridization(), HYBRID_LIST)
    feats += one_hot(atom.GetChiralTag(), CHIRALITY_LIST)
    
    return np.asarray(feats, dtype=np.float32)


def atom_morgan_fingerprint(morgan_gen, mol, atom_idx: int) -> np.ndarray:
    """Generate per-atom Morgan fingerprint (local environment)."""
    fp_np = morgan_gen.GetFingerprintAsNumPy(mol, fromAtoms=[atom_idx])
    return fp_np.astype(np.float32)


# ============================================================================
# Graph Construction
# ============================================================================

def smiles_to_graph(smiles: str) -> Optional[nx.Graph]:
    """
    Convert SMILES string to NetworkX graph with rich node/edge features.
    
    Node features:
    - x: 154-dim atom features (one-hot encoded)
    - fp: 128-dim Morgan fingerprint (local environment)
    - atomic_num, symbol, degree, is_aromatic, in_ring (metadata)
    
    Edge features:
    - bond_type: SINGLE, DOUBLE, TRIPLE, AROMATIC
    - bond_dir: NONE, BEGINWEDGE, BEGINDASH, etc.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Warning: Failed to parse SMILES: {smiles}")
        return None
    
    G = nx.Graph()
    G.graph["smiles"] = smiles
    
    # Morgan fingerprint generator (reused for all atoms)
    morgan_gen = GetMorganGenerator(radius=2, fpSize=128)
    
    # Add nodes with features
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        
        G.add_node(
            idx,
            x=atom_features(atom),
            fp=atom_morgan_fingerprint(morgan_gen, mol, idx),
            atomic_num=atom.GetAtomicNum(),
            symbol=atom.GetSymbol(),
            degree=atom.GetTotalDegree(),
            is_aromatic=atom.GetIsAromatic(),
            in_ring=atom.IsInRing(),
        )
    
    # Add edges with bond information
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        G.add_edge(
            i,
            j,
            bond_type=str(bond.GetBondType()),
            bond_dir=str(bond.GetBondDir()),
        )
    
    return G


# ============================================================================
# Graph Cache Management
# ============================================================================

def save_graph(cid: str, graph: nx.Graph, output_dir: str) -> str:
    """Save graph to pickle file."""
    output_path = os.path.join(output_dir, f"{cid}.gpickle")
    with open(output_path, "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    return output_path


def graph_exists(cid: str, output_dir: str) -> Optional[str]:
    """Check if graph pickle already exists."""
    path = os.path.join(output_dir, f"{cid}.gpickle")
    return path if os.path.exists(path) else None


# ============================================================================
# Main Pipeline
# ============================================================================

def load_atc_mapping(atc_file: str) -> Dict[str, str]:
    """Load CID -> ATC code mapping from TSV file."""
    cid_to_atc = {}
    with open(atc_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cid = parse_cid(parts[0])
                atc = parts[1]
                if cid and cid not in cid_to_atc:
                    cid_to_atc[cid] = atc
    return cid_to_atc


def load_drug_names(names_file: str) -> List[tuple]:
    """Load drug names from TSV file."""
    drugs = []
    with open(names_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                drugs.append((parts[0], parts[1]))
    return drugs


def build_graphs(
    names_file: str,
    atc_file: str,
    output_dir: str = "data/processed/graphs_v2",
    cache_file: str = "data/processed/smiles_cache.csv",
    force_rebuild: bool = False,
) -> List[Drug]:
    """
    Main pipeline to build molecular graphs.
    
    Args:
        names_file: Path to drug_names.tsv
        atc_file: Path to drug_atc.tsv
        output_dir: Directory to save graph pickles
        cache_file: Path to SMILES cache CSV
        force_rebuild: If True, rebuild all graphs even if cached
        
    Returns:
        List of Drug objects with graph paths
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    smiles_cache = SMILESCache(cache_file)
    cid_to_atc = load_atc_mapping(atc_file)
    drug_names = load_drug_names(names_file)
    
    print(f"Building molecular graphs for {len(drug_names)} compounds...")
    print(f"Output directory: {output_dir}")
    print(f"SMILES cache: {cache_file}")
    print("-" * 80)
    
    drugs = []
    for idx, (cid_raw, name) in enumerate(drug_names, 1):
        cid = parse_cid(cid_raw)
        if not cid:
            continue
        
        # Progress updates
        if idx % 10 == 0 or idx == 1 or idx == len(drug_names):
            print(f"[{idx}/{len(drug_names)}] Processing CID {cid} - {name}")
        
        # Get SMILES (from cache or PubChem)
        cached = smiles_cache.get(cid)
        if cached:
            smiles_raw = cached.get("smiles_raw") or None
            smiles_sanitized = cached.get("smiles_sanitized") or None
        else:
            smiles_raw = fetch_smiles_from_pubchem(cid)
            smiles_sanitized = sanitize_smiles(smiles_raw)
            smiles_cache.add(cid, name, cid_to_atc.get(cid, ""), smiles_raw, smiles_sanitized)
            time.sleep(0.2)  # Rate limit PubChem requests
        
        # Build or load graph
        graph_path = None
        if not force_rebuild:
            graph_path = graph_exists(cid, output_dir)
        
        if graph_path is None and smiles_sanitized:
            graph = smiles_to_graph(smiles_sanitized)
            if graph is not None:
                graph_path = save_graph(cid, graph, output_dir)
        
        drugs.append(
            Drug(
                cid=cid,
                name=name,
                atc=cid_to_atc.get(cid),
                smiles_raw=smiles_raw,
                smiles_sanitized=smiles_sanitized,
                graph_path=graph_path,
            )
        )
    
    # Summary
    print("-" * 80)
    print(f"DONE")
    print(f"  SMILES cached: {sum(1 for d in drugs if d.smiles_sanitized)}/{len(drugs)}")
    print(f"  Graphs built: {sum(1 for d in drugs if d.graph_path)}/{len(drugs)}")
    print(f"  Output: {output_dir}")
    
    return drugs


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build molecular graphs from SIDER drug data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build graphs with default settings
  python scripts/build_molecular_graphs.py data/raw/drug_names.tsv data/raw/drug_atc.tsv
  
  # Force rebuild all graphs
  python scripts/build_molecular_graphs.py data/raw/drug_names.tsv data/raw/drug_atc.tsv --force-rebuild
  
  # Custom output directory
  python scripts/build_molecular_graphs.py data/raw/drug_names.tsv data/raw/drug_atc.tsv \\
      --output-dir data/processed/graphs_custom
        """
    )
    
    parser.add_argument("names_file", help="Path to drug_names.tsv")
    parser.add_argument("atc_file", help="Path to drug_atc.tsv")
    parser.add_argument(
        "--output-dir",
        default="data/processed/graphs_v2",
        help="Directory to save graph pickles (default: data/processed/graphs_v2)"
    )
    parser.add_argument(
        "--cache-file",
        default="data/processed/smiles_cache.csv",
        help="Path to SMILES cache CSV (default: data/processed/smiles_cache.csv)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild all graphs even if cached"
    )
    
    args = parser.parse_args()
    
    build_graphs(
        names_file=args.names_file,
        atc_file=args.atc_file,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()

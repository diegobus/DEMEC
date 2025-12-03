import os, csv, re, sys, time
from dataclasses import dataclass
from typing import Optional, Dict, List
import pubchempy as pcp
import networkx as nx
import pysmiles as psm
import pickle

# Optional RDKit for proper sanitization
try:
    from rdkit import Chem

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

CID_RE = re.compile(r"(\d+)$")

# ---- Config: where to cache -------------------------------------------------
CACHE_DIR = "data/processed"
SMILES_CACHE = os.path.join(CACHE_DIR, "smiles_cache.csv")
GRAPH_DIR = os.path.join(CACHE_DIR, "graphs")  # one file per CID: <cid>.gpickle

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


@dataclass
class Drug:
    cid: Optional[str] = None
    name: Optional[str] = None
    atc: Optional[str] = None
    smiles_raw: Optional[str] = None  # as returned by PubChem
    smiles_sanitized: Optional[str] = None  # stripped stereo (safe for pysmiles)
    graph_path: Optional[str] = None  # path to gpickle if built


# ---- Cache helpers ----------------------------------------------------------
def load_smiles_cache(path: str = SMILES_CACHE) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[row["cid"]] = row
    return cache


def append_smiles_cache(row: Dict[str, str], path: str = SMILES_CACHE) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cid", "name", "atc", "smiles_raw", "smiles_sanitized", "ts"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def cache_graph(cid: str, g: nx.Graph) -> str:
    out = os.path.join(GRAPH_DIR, f"{cid}.gpickle")
    with open(out, "wb") as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
    return out


def graph_cached(cid: str) -> Optional[str]:
    p = os.path.join(GRAPH_DIR, f"{cid}.gpickle")
    return p if os.path.exists(p) else None


# ---- Utilities --------------------------------------------------------------
def parse_cid(raw: str) -> Optional[str]:
    m = CID_RE.search(raw.strip())
    return m.group(1) if m else None


def fetch_smiles_from_pubchem(cid: str) -> Optional[str]:
    try:
        c = pcp.Compound.from_cid(int(cid))
        return (
            getattr(c, "isomeric_smiles", None)
            or getattr(c, "canonical_smiles", None)
            or getattr(c, "smiles", None)
        )
    except Exception:
        return None


def sanitize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Non-isomeric removes / and \ around double bonds that can break pysmiles
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    # last-resort fallback (not chemically rigorous)
    return smiles.replace("/", "").replace("\\", "")


def smiles_to_nx(smiles: str) -> Optional[nx.Graph]:
    try:
        return psm.read_smiles(smiles)
    except Exception:
        return None


# ---- Main pipeline with caching --------------------------------------------
def aggregate_data(names_file: str, atc_file: str) -> List[Drug]:
    # Load cache once
    smiles_cache = load_smiles_cache()

    # Read inputs
    with open(names_file, "r") as f:
        name_rows = [line.strip().split("\t") for line in f if line.strip()]
    with open(atc_file, "r") as f:
        atc_rows = [line.strip().split("\t") for line in f if line.strip()]
    cid_to_atc = {}
    for cid_raw, atc in atc_rows:
        cid = parse_cid(cid_raw)
        if cid and cid not in cid_to_atc:
            cid_to_atc[cid] = atc

    drugs: List[Drug] = []
    total = len(name_rows)
    print(f"Processing {total} compounds with caching…")

    for idx, (cid_raw, name) in enumerate(name_rows, 1):
        cid = parse_cid(cid_raw)
        if not cid:
            continue

        # Progress ping every 10
        if idx % 10 == 0 or idx == 1 or idx == total:
            print(f"[{idx}/{total}] CID {cid} — {name}")

        # Check cache first
        cached = smiles_cache.get(cid)
        if cached:
            smiles_raw = cached["smiles_raw"] or None
            smiles_sanitized = cached["smiles_sanitized"] or None
        else:
            # Miss: fetch and append to cache
            smiles_raw = fetch_smiles_from_pubchem(cid)
            smiles_sanitized = sanitize_smiles(smiles_raw)
            append_smiles_cache(
                {
                    "cid": cid,
                    "name": name,
                    "atc": cid_to_atc.get(cid, ""),
                    "smiles_raw": smiles_raw or "",
                    "smiles_sanitized": smiles_sanitized or "",
                    "ts": str(int(time.time())),
                }
            )
            # Avoid hammering PubChem
            time.sleep(0.2)

        # Build/load graph cache
        gpath = graph_cached(cid)
        if gpath is None and smiles_sanitized:
            g = smiles_to_nx(smiles_sanitized)
            if g is not None:
                gpath = cache_graph(cid, g)

        drugs.append(
            Drug(
                cid=cid,
                name=name,
                atc=cid_to_atc.get(cid),
                smiles_raw=smiles_raw,
                smiles_sanitized=smiles_sanitized,
                graph_path=gpath,
            )
        )

    built = sum(1 for d in drugs if d.graph_path)
    print(
        f"Finished: SMILES cached for {sum(1 for d in drugs if d.smiles_sanitized)}/{len(drugs)} | graphs cached for {built}/{len(drugs)}"
    )
    print(f"• SMILES cache: {SMILES_CACHE}")
    print(f"• Graphs dir  : {GRAPH_DIR}")
    return drugs


# CLI
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/data/smiles_lookup.py drug_names.tsv drug_atc.tsv")
        sys.exit(1)
    aggregate_data(sys.argv[1], sys.argv[2])

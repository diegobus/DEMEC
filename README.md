# DEMEC: Drug Embedding & Multi-Effect Classification

Multi-task Graph Neural Network for predicting drug side effects from molecular structure.

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate demec

# Train model (recommended config)
python -m demec.training.train --config configs/default.yaml
```

## Overview

DEMEC uses Graph Neural Networks to predict drug side effects and therapeutic properties from molecular structure alone. The model learns from 1,430 drugs with 4,251 documented side effects from the SIDER database.

**Key Features:**
- Heterogeneous molecular graphs with typed edges (bond types)
- Multi-task learning (side effects, ATC classes, MACCS fingerprints)
- Focal loss for handling extreme class imbalance
- Attention-based graph pooling

## Architecture

```
Molecular Graph → GNN Backbone → Graph Pooling → Task-Specific Heads
                  (GAT/GCN)      (Attention)     (SE, ATC, MACCS)
```

**Model Components:**
- **Backbone**: GAT or GCN with heterogeneous edge types
- **Pooling**: Mean, attention, or MLP-based aggregation
- **Heads**: Separate prediction heads per task
- **Loss**: Focal loss (SE), BCE (ATC/MACCS), MSE (molecular properties)

## Data

### Datasets
- **SIDER 4.1**: 1,430 drugs, 4,251 side effects, 139,756 interactions
- **PubChem**: SMILES strings and compound data

### Tasks
| Task | Type | Dimension | Loss |
|------|------|-----------|------|
| Side Effects | Multi-label classification | 4,251 (or top-N) | Focal Loss |
| ATC Classes | Multi-label classification | 167 | BCEWithLogits |
| MACCS Fingerprints | Multi-label classification | 166 | BCEWithLogits |
| Molecular Weight | Regression | 1 | MSE |

### Data Split
- **80/10/10** train/validation/test split
- Fixed seed (42) for reproducibility
- Random split over drug CIDs (no stratification)

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- See `environment.yml` for complete list

### Setup

```bash
# Clone repository
git clone https://github.com/diegobus/DEMEC.git
cd DEMEC

# Create environment
conda env create -f environment.yml
conda activate demec

# Install package
pip install -e .
```

## Data Preparation

### 1. Download Raw Data

Place SIDER files in `data/raw/`:
- `meddra_all_se.tsv` - Side effect mappings
- `drug_atc.tsv` - ATC classifications
- `drug_names.tsv` - Drug names

### 2. Run Preprocessing Pipeline

```bash
# Process side effect data
python scripts/aggregate_sider.py

# Build ATC matrix
python scripts/build_atc_matrix.py

# Build MACCS fingerprints
python scripts/build_maccs_matrix.py

# Extract molecular properties
python scripts/extract_molecular_properties.py

# Build molecular graphs
python scripts/build_molecular_graphs.py
```

**Output:**
```
data/processed/
├── cid_se_matrix.csv              # Side effect labels (1430 x 4251)
├── cid_atc_l3_matrix.csv          # ATC classifications (1430 x 167)
├── cid_maccs_matrix.csv           # MACCS fingerprints (1430 x 166)
├── cid_molprops_matrix_simple.csv # Molecular properties
└── graphs_v2/                     # Molecular graphs (.gpickle)
```

## Training

### Using Config Files (Recommended)

```bash
# Multi-task training (default)
python -m demec.training.train --config configs/default.yaml

# Single-task baseline
python -m demec.training.train --config configs/single_task_side_effects.yaml

# GCN baseline
python -m demec.training.train --config configs/gcn_baseline.yaml
```

### Command-Line Options

```bash
python -m demec.training.train \
    --config configs/default.yaml \
    --epochs 300 \
    --batch_size 32 \
    --lr 0.0001 \
    --model gat \
    --pooling attention
```

### Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--model` | `gcn`, `gat` | GNN architecture |
| `--pooling` | `mean`, `attention`, `mlp` | Graph pooling method |
| `--hidden_dim` | int | Hidden layer size (default: 128) |
| `--num_layers` | int | Number of GNN layers (default: 5) |
| `--dropout` | float | Dropout rate (default: 0.2) |
| `--focal_alpha` | float | Focal loss alpha (default: 0.25) |

See `configs/README.md` for detailed configuration guide.

## Configuration

Example `default.yaml`:

```yaml
experiment_name: "multitask_gat"

model:
  architecture: "gat"
  hidden_dim: 128
  num_layers: 5
  dropout: 0.2
  heads: 3
  pooling: "attention"

training:
  epochs: 300
  batch_size: 32
  lr: 0.0001
  seed: 42
  task_weights: "side_effects:1.0,atc:1.0,maccs:1.0"
  focal_alpha: 0.25

data:
  train_tasks: "side_effects,atc,maccs"
  eval_tasks: "side_effects"
  max_side_effects: 100  # Top-100 most common
```

## Evaluation

### Metrics

**Classification Tasks:**
- mAP (mean Average Precision)
- AUROC (Area Under ROC Curve)
- Precision, Recall, F1

**Regression Tasks:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R^2 Score

### Monitoring

TensorBoard logs are saved to `runs/`:

```bash
tensorboard --logdir runs/
```

### Checkpoints

Best models are saved to `checkpoints/` based on validation mAP.

## Results

### Side Effect Prediction (Top-100)

| Model | Pooling | mAP | AUROC |
|-------|---------|-----|-------|
| GAT | Attention | 0.679 | 0.657 |
| GAT | Mean | 0.669 | 0.652 |
| GCN | Attention | 0.665 | 0.648 |
| GCN | Mean | 0.658 | 0.645 |

### Multi-task Learning Impact

| Configuration | Side Effects mAP | Notes |
|---------------|------------------|-------|
| Single-task | 0.669 | Baseline |
| Multi-task (SE + ATC + MACCS) | 0.679 | +1.5% improvement |
| Multi-task (all 4,251 SEs) | 0.427 | Label noise degrades performance |

**Key Findings:**
- Attention pooling outperforms mean pooling
- GAT slightly better than GCN for this task
- Multi-task learning helps with clean labels (top-N filtering)
- Label quality matters more than task difficulty

## Project Structure

```
DEMEC/
├── configs/                  # Configuration files
│   ├── default.yaml         # Recommended config
│   ├── single_task_side_effects.yaml
│   ├── gcn_baseline.yaml
│   └── README.md            # Config guide
├── data/
│   ├── raw/                 # Downloaded datasets
│   └── processed/           # Preprocessed data
├── scripts/                 # Data preprocessing
│   ├── aggregate_sider.py
│   ├── build_atc_matrix.py
│   ├── build_maccs_matrix.py
│   ├── build_molecular_graphs.py
│   └── extract_molecular_properties.py
├── src/demec/              # Main package
│   ├── data/
│   │   └── data_loader.py  # PyG dataset
│   ├── models/
│   │   ├── gnn_backbone.py # GNN architectures
│   │   └── heads.py        # Prediction heads
│   ├── training/
│   │   └── train.py        # Training loop
│   └── utils/
│       ├── eval_metrics.py # Metrics computation
│       ├── logger.py       # Experiment logging
│       └── losses.py       # Focal loss
├── checkpoints/            # Saved models
├── runs/                   # TensorBoard logs
├── environment.yml         # Conda environment
└── README.md              # This file
```

## References

- **SIDER 4.1:** Kuhn et al., *Nucleic Acids Research* (2016). [http://sideeffects.embl.de/](http://sideeffects.embl.de/)  
- **DrugBank:** Wishart et al., *Nucleic Acids Research* (2018). [https://pubmed.ncbi.nlm.nih.gov/29126136/](https://pubmed.ncbi.nlm.nih.gov/29126136/)  
- **Multi-task GNNs for Molecular Prediction:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC11606038/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11606038/)
- **Focal Loss:** Lin et al., *ICCV* (2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **Graph Attention Networks:** Veličković et al., *ICLR* (2018). [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

## License

MIT License

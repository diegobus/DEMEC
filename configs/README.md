# Configuration Files

This directory contains YAML configuration files for training the DEMEC model.

## Available Configurations

### `default.yaml`
**Multi-task learning with top-100 side effects**
- Architecture: GAT with attention pooling
- Tasks: Side effects (top-100), ATC, MACCS
- Best overall performance

```bash
python -m demec.training.train --config configs/default.yaml
```

### `single_task_side_effects.yaml`
**Single-task baseline for side effect prediction**
- Architecture: GAT with attention pooling
- Tasks: Side effects only (top-100)
- Use for ablation studies

```bash
python -m demec.training.train --config configs/single_task_side_effects.yaml
```

### `gcn_baseline.yaml`
**GCN baseline for comparison**
- Architecture: GCN with mean pooling
- Tasks: Multi-task (SE, ATC, MACCS)
- Simpler architecture for comparison

```bash
python -m demec.training.train --config configs/gcn_baseline.yaml
```

## Configuration Structure

```yaml
experiment_name: "my_experiment"

model:
  architecture: "gat"          # "gcn" or "gat"
  hidden_dim: 128              # Hidden layer size
  num_layers: 5                # Number of GNN layers
  dropout: 0.2                 # Dropout rate
  heads: 3                     # Attention heads (GAT only)
  pooling: "attention"         # Pooling method

training:
  epochs: 300                  # Training epochs
  batch_size: 32               # Batch size
  lr: 0.0001                   # Learning rate
  seed: 42                     # Random seed
  task_weights: "..."          # Task loss weights (optional)
  focal_alpha: 0.25            # Focal loss alpha

data:
  train_tasks: "..."           # Tasks to train on
  eval_tasks: "..."            # Tasks to evaluate
  max_side_effects: 100        # Filter to top-N SEs (optional)
```

## Key Parameters

### Architecture Options
- **`architecture`**: `"gcn"` or `"gat"`
  - GCN: Simple graph convolution
  - GAT: Graph attention networks (better performance)

### Pooling Methods
- **`pooling`**: `"mean"`, `"sum"`, `"max"`, `"mlp"`, `"attention"`
  - `mean`: Average node features (simple, fast)
  - `attention`: Learnable attention weights (best performance)

### Task Configuration
- **`train_tasks`**: Comma-separated list
  - Options: `"side_effects"`, `"atc"`, `"maccs"`, `"molprops"`
  - Example: `"side_effects,atc,maccs"`

- **`eval_tasks`**: Tasks to compute metrics for
  - Usually a subset of `train_tasks`
  - Example: `"side_effects"` (primary task)

### Side Effect Filtering
- **`max_side_effects`**: Limit to top-N most common
  - `100`: Top-100 (recommended, 38.8% coverage)
  - `200`: Top-200 (55.8% coverage)
  - `null`: All 4,251 side effects (very sparse)

### Task Weights
- **`task_weights`**: Balance multitask loss
  - Format: `"task1:weight1,task2:weight2"`
  - Example: `"side_effects:1.0,atc:1.0,maccs:1.0"`
  - Higher weight = more influence on training

## Creating Custom Configs

1. Copy `default.yaml`
2. Modify parameters as needed
3. Run with: `python -m demec.training.train --config configs/your_config.yaml`

## Archived Experiments

The `experiments_archive/` directory contains all experimental configurations used during development.

## Notes

- All configs use **seed=42** for reproducibility
- **80/10/10 train/val/test split** (fixed, no stratification)
- Models are saved to `checkpoints/` directory
- TensorBoard logs saved to `runs/` directory

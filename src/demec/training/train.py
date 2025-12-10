import sys
import os
import argparse
import torch
import yaml
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset, make_splits as make_splits_homo
from demec.data.hetero_data_loader import HeteroGraphDataset, make_splits as make_splits_hetero
from demec.utils.eval_metrics import comprehensive_metrics
from demec.utils.logger import ExperimentLogger, format_metrics_string, format_task_losses
from demec.models.gnn_backbone import GNNBackbone
from demec.models.hetero_gnn import HeteroGNNBackbone, HeteroMultiTaskGNN
from demec.models.multitask import MultiTaskGNN
from demec.models.heads import PredictionHead


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(args, dataset, device):
    """
    Factory function to create model based on graph type and architecture.

    Args:
        args: Command line arguments or config namespace
        dataset: Dataset object
        device: torch device

    Returns:
        model: Initialized model
        loss_funcs: Dictionary of loss functions per task
        task_weights: Dictionary of task weights
    """
    heads_dict = {}
    loss_funcs = {}
    task_weights = {}

    # Parse selected tasks
    selected_tasks = [t.strip() for t in args.tasks.split(',')]

    # Parse task weights
    weight_dict = {}
    if hasattr(args, 'task_weights') and args.task_weights:
        for pair in args.task_weights.split(','):
            task, weight = pair.split(':')
            weight_dict[task.strip()] = float(weight)

    # Initialize prediction heads (only for selected tasks)
    for task_name, dim in dataset.task_dims.items():
        if task_name not in selected_tasks:
            print(f"Skipping task: {task_name} (not in selected tasks)")
            continue

        print(f"Initializing head for task: {task_name} (output_dim={dim})")

        # Determine task type (classification vs regression)
        if task_name == "molprops":
            task_type = "regression"
            loss_type = "mse"
            focal_alpha = 0.25  # Not used for regression
        else:
            task_type = "classification"
            # Determine loss type and focal alpha for classification
            if task_name == "side_effects":
                loss_type = "focal"
                focal_alpha = args.focal_alpha if hasattr(args, 'focal_alpha') and args.focal_alpha is not None else 0.25
            else:
                loss_type = "bce"
                focal_alpha = 0.25  # Not used for BCE

        head = PredictionHead(
            input_dim=args.hidden_dim,
            output_dim=dim,
            hidden_dims=[args.hidden_dim],
            dropout=args.dropout,
            task_type=task_type,
            loss_type=loss_type,
            focal_alpha=focal_alpha
        )
        heads_dict[task_name] = head
        loss_funcs[task_name] = head.get_loss_func()

        # Set task weight (default 1.0 if not specified)
        task_weights[task_name] = weight_dict.get(task_name, 1.0)
        print(f"  Task weight: {task_weights[task_name]}")

    # Create backbone based on graph type
    if args.hetero:
        print(f"Initializing {args.model.upper()} heterogeneous backbone...")
        backbone = HeteroGNNBackbone(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            conv_type=args.model,
            heads=args.heads
        )
        model = HeteroMultiTaskGNN(backbone, heads_dict)
    else:
        print(f"Initializing {args.model.upper()} homogeneous backbone...")
        backbone = GNNBackbone(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            conv_type=args.model,
            heads=args.heads,
            output_dim=None
        )
        model = MultiTaskGNN(backbone, heads_dict)

    return model.to(device), loss_funcs, task_weights


def train_epoch(model, loader, optimizer, loss_funcs, task_weights, device, clip_grad_norm=None):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in loss_funcs.keys()}
    all_se_logits = []
    all_se_targets = []

    for batch in loader:
        batch = batch.to(device)
        results = model(batch)

        batch_loss = 0.0
        for task_name, logits in results.items():
            target_attr = f"y_{task_name}"
            mask_attr = f"mask_{task_name}"

            if hasattr(batch, target_attr):
                target = getattr(batch, target_attr)

                if hasattr(batch, mask_attr):
                    mask = getattr(batch, mask_attr).squeeze()
                    if not mask.any():
                        continue
                    target = target[mask]
                    logits = logits[mask]

                loss = loss_funcs[task_name](logits, target)
                weighted_loss = task_weights[task_name] * loss
                batch_loss += weighted_loss

                # Track per-task losses
                task_losses[task_name] += loss.item() * batch.num_graphs

                # Collect side_effects predictions for metrics
                if task_name == 'side_effects':
                    all_se_logits.append(logits.detach())
                    all_se_targets.append(target.detach())

        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping (optional)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += batch_loss.item() * batch.num_graphs

    # Compute metrics on all accumulated predictions
    if all_se_logits:
        all_se_logits = torch.cat(all_se_logits, dim=0)
        all_se_targets = torch.cat(all_se_targets, dim=0)
        metrics = comprehensive_metrics(all_se_logits, all_se_targets, k_values=[50, 100])
    else:
        metrics = {}

    return total_loss, task_losses, metrics


def validate(model, loader, loss_funcs, task_weights, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in loss_funcs.keys()}
    all_se_logits = []
    all_se_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            results = model(batch)

            batch_loss = 0.0
            for task_name, logits in results.items():
                target_attr = f"y_{task_name}"
                mask_attr = f"mask_{task_name}"

                if hasattr(batch, target_attr):
                    target = getattr(batch, target_attr)

                    if hasattr(batch, mask_attr):
                        mask = getattr(batch, mask_attr).squeeze()
                        if not mask.any():
                            continue
                        target = target[mask]
                        logits = logits[mask]

                    loss = loss_funcs[task_name](logits, target)
                    weighted_loss = task_weights[task_name] * loss
                    batch_loss += weighted_loss

                    # Track per-task losses
                    task_losses[task_name] += loss.item() * batch.num_graphs

                    # Collect side_effects predictions for metrics
                    if task_name == 'side_effects':
                        all_se_logits.append(logits.detach())
                        all_se_targets.append(target.detach())

            total_loss += batch_loss.item() * batch.num_graphs

    # Compute metrics on all accumulated predictions
    if all_se_logits:
        all_se_logits = torch.cat(all_se_logits, dim=0)
        all_se_targets = torch.cat(all_se_targets, dim=0)
        metrics = comprehensive_metrics(all_se_logits, all_se_targets, k_values=[50, 100])
    else:
        metrics = {}

    return total_loss, task_losses, metrics


def main():
    parser = argparse.ArgumentParser(description="Unified Training Framework for GNN Models")

    # Config file argument
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Graph type selection
    parser.add_argument("--hetero", action="store_true",
                        help="Use heterogeneous graphs with bond-type-specific edges")

    # Optional CLI overrides (can override config values)
    parser.add_argument("--model", type=str, choices=["gcn", "gat"], help="Model architecture")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--heads", type=int, help="Number of attention heads (GAT only)")

    # Task selection and weighting (for ablation studies)
    parser.add_argument("--tasks", type=str, help="Comma-separated list of tasks to train")
    parser.add_argument("--task_weights", type=str, default=None,
                        help="Task weights as 'task1:weight1,task2:weight2'")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Alpha parameter for Focal Loss (default: 0.25)")
    parser.add_argument("--clip_grad_norm", type=float, default=None,
                        help="Gradient clipping max norm (e.g., 1.0)")

    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, help="Directory for TensorBoard logs")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for logging")
    parser.add_argument("--save_model", action="store_true",
                        help="Save best model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory for model checkpoints")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create namespace with config defaults
    class ConfigNamespace:
        pass

    cfg = ConfigNamespace()

    # Apply config values
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    # Set attributes from config with CLI overrides
    cfg.model = args.model if args.model else model_cfg.get('architecture', 'gcn')
    cfg.epochs = args.epochs if args.epochs else train_cfg.get('epochs', 10)
    cfg.batch_size = args.batch_size if args.batch_size else train_cfg.get('batch_size', 32)
    cfg.lr = args.lr if args.lr else train_cfg.get('lr', 1e-3)
    cfg.seed = args.seed if args.seed else train_cfg.get('seed', 42)
    cfg.hidden_dim = args.hidden_dim if args.hidden_dim else model_cfg.get('hidden_dim', 64)
    cfg.num_layers = args.num_layers if args.num_layers else model_cfg.get('num_layers', 5)
    cfg.dropout = args.dropout if args.dropout else model_cfg.get('dropout', 0.2)
    cfg.heads = args.heads if args.heads else model_cfg.get('heads', 3)
    cfg.hetero = args.hetero
    cfg.focal_alpha = args.focal_alpha
    cfg.clip_grad_norm = args.clip_grad_norm
    cfg.save_model = args.save_model
    cfg.exp_name = args.exp_name
    cfg.task_weights = args.task_weights
    cfg.log_dir = args.log_dir if args.log_dir else train_cfg.get('log_dir', 'runs')
    cfg.checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else train_cfg.get('checkpoint_dir', 'checkpoints')

    # Task configuration from config or CLI
    if args.tasks:
        cfg.tasks = args.tasks
    else:
        cfg.tasks = data_cfg.get('tasks_enabled', 'side_effects,atc,maccs')

    # Auto-detect graph directory and input dimension based on graph type
    if cfg.hetero:
        cfg.graphs_dir = data_cfg.get('graphs_dir', 'data/processed/graphs_v2/')
        cfg.input_dim = model_cfg.get('input_dim', 154)
        cfg.feature_key = data_cfg.get('feature_key', 'x')
    else:
        cfg.graphs_dir = data_cfg.get('graphs_dir', 'data/processed/graphs/')
        cfg.input_dim = model_cfg.get('node_dim', model_cfg.get('input_dim', 1))
        cfg.feature_key = data_cfg.get('feature_key', None)

    print(f"Loaded Configuration: {config}")
    print(f"Final Config: {vars(cfg)}")
    print(f"Graph Type: {'Heterogeneous' if cfg.hetero else 'Homogeneous'}")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Task configuration
    task_config = {
        'side_effects': "data/processed/cid_se_matrix.csv",
        'atc': "data/processed/cid_atc_l3_matrix.csv",
        'maccs': "data/processed/cid_maccs_matrix.csv",
        'molprops': "data/processed/cid_molprops_matrix.csv"
    }

    # Load dataset based on graph type
    print(f"Loading {'heterogeneous' if cfg.hetero else 'homogeneous'} graph dataset...")
    if cfg.hetero:
        dataset = HeteroGraphDataset(
            cfg.graphs_dir,
            task_config=task_config,
            feature_key=cfg.feature_key
        )
        train_ds, val_ds, test_ds = make_splits_hetero(dataset, train=0.8, val=0.1, seed=cfg.seed)
    else:
        dataset = GraphStructureDataset(
            cfg.graphs_dir,
            task_config=task_config,
            node_dim=cfg.input_dim,
            feature_key=cfg.feature_key
        )
        train_ds, val_ds, test_ds = make_splits_homo(dataset, train=0.8, val=0.1, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # Create model
    model, loss_funcs, task_weights = create_model(cfg, dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Setup experiment logger
    logger = ExperimentLogger(cfg, log_dir=cfg.log_dir, checkpoint_dir=cfg.checkpoint_dir)
    logger.log_hyperparameters(task_weights)

    print(f"\nStarting training loop for {cfg.epochs} epochs...")
    print(f"Active tasks: {', '.join(loss_funcs.keys())}")
    print(f"Task weights: {task_weights}")
    if cfg.clip_grad_norm:
        print(f"Gradient clipping: max_norm={cfg.clip_grad_norm}")
    print(f"Metrics: mAP (primary), P@50, P@100, AUROC")
    print("-" * 80)

    for epoch in range(cfg.epochs):
        train_loss, train_task_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_funcs, task_weights, device, cfg.clip_grad_norm
        )
        val_loss, val_task_losses, val_metrics = validate(
            model, val_loader, loss_funcs, task_weights, device
        )

        # Log metrics to TensorBoard
        avg_train_loss, avg_val_loss = logger.log_epoch(
            epoch, train_loss, val_loss, train_task_losses, val_task_losses,
            train_metrics, val_metrics, len(train_ds), len(val_ds), optimizer
        )

        # Print to console
        train_str = format_metrics_string(train_metrics)
        val_str = format_metrics_string(val_metrics)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Train: {train_str} | Val: {val_str}"
        )

        # Print per-task losses every 10 epochs
        if (epoch + 1) % 10 == 0 and len(train_task_losses) > 1:
            task_loss_str = format_task_losses(train_task_losses, len(train_ds))
            print(f"  Task losses: {task_loss_str}")

        # Save checkpoint if best
        current_val_map = val_metrics.get('mAP', 0)
        if logger.save_checkpoint(epoch, model, optimizer, current_val_map):
            print(f"    Saved best model (mAP: {current_val_map:.4f})")

    # Finalize logging
    logger.finalize(val_metrics)


if __name__ == "__main__":
    main()

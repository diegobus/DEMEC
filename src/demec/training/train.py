import sys
import os
import argparse
import torch
import yaml
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphDataset, make_splits
from demec.utils.eval_metrics import comprehensive_metrics
from demec.utils.logger import ExperimentLogger, format_metrics_string, format_task_losses
from demec.models.gnn_backbone import GNNBackbone, MultiTaskGNN
from demec.models.heads import PredictionHead


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(dataset, args, device):
    """
    Initialize model with task-specific prediction heads.
    
    Returns:
        model: Initialized model
        loss_funcs: Dictionary of loss functions per task
        task_weights: Dictionary of task weights
        train_tasks: List of tasks to train on
        eval_tasks: List of tasks to evaluate on
    """
    heads_dict = {}
    loss_funcs = {}
    task_weights = {}

    # Parse train and eval tasks
    train_tasks = [t.strip() for t in args.train_tasks.split(',')]
    eval_tasks = [t.strip() for t in args.eval_tasks.split(',')]
    
    # All tasks that need heads (union of train and eval)
    all_needed_tasks = set(train_tasks + eval_tasks)

    # Parse task weights
    weight_dict = {}
    if hasattr(args, 'task_weights') and args.task_weights:
        for pair in args.task_weights.split(','):
            task, weight = pair.split(':')
            weight_dict[task.strip()] = float(weight)

    # Initialize prediction heads for all needed tasks
    print("\nInitializing prediction heads:")
    for task_name, dim in dataset.task_dims.items():
        if task_name not in all_needed_tasks:
            continue

        train_flag = "1" if task_name in train_tasks else "0"
        eval_flag = "1" if task_name in eval_tasks else "0"
        print(f"  {task_name} (dim={dim}) | Train: {train_flag} | Eval: {eval_flag}")

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

        # Set task weight (default 1.0 if not specified, 0.0 if not training)
        if task_name in train_tasks:
            task_weights[task_name] = weight_dict.get(task_name, 1.0)
        else:
            task_weights[task_name] = 0.0  # No contribution to loss

    # Create backbone
    pooling = getattr(args, 'pooling', 'mean')
    print(f"\nInitializing {args.model.upper()} backbone (pooling: {pooling})...")
    backbone = GNNBackbone(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.model,
        heads=args.heads,
        pooling=pooling
    )
    model = MultiTaskGNN(backbone, heads_dict)

    return model.to(device), loss_funcs, task_weights, train_tasks, eval_tasks


def train_epoch(model, loader, optimizer, loss_funcs, task_weights, eval_tasks, device, clip_grad_norm=None):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in loss_funcs.keys()}
    
    # Collect predictions for each eval task
    eval_predictions = {task: {'logits': [], 'targets': []} for task in eval_tasks}

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

                # Collect predictions for eval tasks
                if task_name in eval_tasks:
                    eval_predictions[task_name]['logits'].append(logits.detach())
                    eval_predictions[task_name]['targets'].append(target.detach())

        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping (optional)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += batch_loss.item() * batch.num_graphs

    # Compute metrics for each eval task
    all_metrics = {}
    for task_name in eval_tasks:
        if eval_predictions[task_name]['logits']:
            logits = torch.cat(eval_predictions[task_name]['logits'], dim=0)
            targets = torch.cat(eval_predictions[task_name]['targets'], dim=0)
            
            # Determine task type for appropriate metrics
            if task_name == 'molprops':
                task_type = 'regression'
            elif task_name == 'maccs':
                task_type = 'fingerprint'
            else:
                task_type = 'classification'
            
            metrics = comprehensive_metrics(logits, targets, task_type=task_type)
            all_metrics[task_name] = metrics
        else:
            all_metrics[task_name] = {}

    return total_loss, task_losses, all_metrics


def validate(model, loader, loss_funcs, task_weights, eval_tasks, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in loss_funcs.keys()}
    
    # Collect predictions for each eval task
    eval_predictions = {task: {'logits': [], 'targets': []} for task in eval_tasks}

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

                    # Collect predictions for eval tasks
                    if task_name in eval_tasks:
                        eval_predictions[task_name]['logits'].append(logits.detach())
                        eval_predictions[task_name]['targets'].append(target.detach())

            total_loss += batch_loss.item() * batch.num_graphs

    # Compute metrics for each eval task
    all_metrics = {}
    for task_name in eval_tasks:
        if eval_predictions[task_name]['logits']:
            logits = torch.cat(eval_predictions[task_name]['logits'], dim=0)
            targets = torch.cat(eval_predictions[task_name]['targets'], dim=0)
            
            # Determine task type for appropriate metrics
            if task_name == 'molprops':
                task_type = 'regression'
            elif task_name == 'maccs':
                task_type = 'fingerprint'
            else:
                task_type = 'classification'
            
            metrics = comprehensive_metrics(logits, targets, task_type=task_type)
            all_metrics[task_name] = metrics
        else:
            all_metrics[task_name] = {}

    return total_loss, task_losses, all_metrics


def train_model(config, device_str=None):
    """
    Wrapper function for hyperparameter optimization with Optuna.
    Executes training with the provided config and returns the best validation mAP.

    Args:
        config: Dictionary with model, training, and data configuration
        device_str: Optional device string ("cpu" or "cuda")

    Returns:
        Negative of best validation mAP (Optuna minimizes by default)
    """
    # This function is called by Optuna's tune script, so we follow the same pattern
    # as main() but return a metric instead of running interactively

    from types import SimpleNamespace
    cfg = SimpleNamespace()

    # Extract config sections
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    # Set all configuration values
    cfg.model = model_cfg.get('architecture', 'gcn')
    cfg.epochs = train_cfg.get('epochs', 10)
    cfg.batch_size = train_cfg.get('batch_size', 32)
    cfg.lr = train_cfg.get('lr', 1e-3)
    cfg.seed = train_cfg.get('seed', 42)
    cfg.hidden_dim = model_cfg.get('hidden_dim', 64)
    cfg.num_layers = model_cfg.get('num_layers', 5)
    cfg.dropout = model_cfg.get('dropout', 0.2)
    cfg.heads = model_cfg.get('heads', 3)
    cfg.max_side_effects = data_cfg.get('max_side_effects', None)
    cfg.focal_alpha = None
    cfg.clip_grad_norm = None
    cfg.save_model = False
    cfg.exp_name = None
    cfg.task_weights = None
    cfg.log_dir = train_cfg.get('log_dir', 'runs')
    cfg.checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints')
    
    # Separate train and eval tasks
    cfg.train_tasks = data_cfg.get('train_tasks', data_cfg.get('tasks_enabled', 'side_effects,atc,maccs'))
    cfg.eval_tasks = data_cfg.get('eval_tasks', cfg.train_tasks)  # Default to same as train_tasks

    # Graph configuration
    cfg.graphs_dir = data_cfg.get('graphs_dir', 'data/processed/graphs_v2/')
    cfg.input_dim = model_cfg.get('input_dim', 154)
    cfg.feature_key = data_cfg.get('feature_key', 'x')

    # Set seed
    torch.manual_seed(cfg.seed)

    # Set device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task configuration
    task_config = {
        'side_effects': "data/processed/cid_se_matrix.csv",
        'atc': "data/processed/cid_atc_l3_matrix.csv",
        'maccs': "data/processed/cid_maccs_matrix.csv",
        'molprops': "data/processed/cid_molprops_matrix_simple.csv"
    }

    # Load dataset
    dataset = GraphDataset(
        cfg.graphs_dir,
        task_config=task_config,
        feature_key=cfg.feature_key,
        max_side_effects=cfg.max_side_effects
    )
    train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Create model
    model, loss_funcs, task_weights = create_model(cfg, dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop - track best validation mAP
    best_val_map = 0.0

    for epoch in range(cfg.epochs):
        train_loss, train_task_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_funcs, task_weights, device, cfg.clip_grad_norm
        )
        val_loss, val_task_losses, val_metrics = validate(
            model, val_loader, loss_funcs, task_weights, device
        )

        # Track best validation mAP
        current_val_map = val_metrics.get('mAP', 0)
        if current_val_map > best_val_map:
            best_val_map = current_val_map

    # Return negative mAP since Optuna minimizes by default
    return -best_val_map


def main():
    parser = argparse.ArgumentParser(description="Unified Training Framework for GNN Models")

    # Config file argument
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

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
    parser.add_argument("--pooling", type=str, choices=["mean", "sum", "max", "mlp", "attention"],
                        help="Graph pooling method")

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
    cfg.pooling = args.pooling if args.pooling else model_cfg.get('pooling', 'mean')
    cfg.max_side_effects = data_cfg.get('max_side_effects', None)
    cfg.focal_alpha = args.focal_alpha
    cfg.clip_grad_norm = args.clip_grad_norm
    cfg.save_model = args.save_model
    # Use experiment_name from config if available, otherwise use CLI arg
    cfg.exp_name = args.exp_name if args.exp_name else config.get('experiment_name', None)
    cfg.task_weights = args.task_weights
    cfg.log_dir = args.log_dir if args.log_dir else train_cfg.get('log_dir', 'runs')
    cfg.checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else train_cfg.get('checkpoint_dir', 'checkpoints')

    # Task configuration from config or CLI
    if args.tasks:
        # If --tasks is provided, use it for both train and eval
        cfg.train_tasks = args.tasks
        cfg.eval_tasks = args.tasks
    else:
        # Use config file or defaults
        cfg.train_tasks = data_cfg.get('train_tasks', data_cfg.get('tasks_enabled', 'side_effects,atc,maccs'))
        cfg.eval_tasks = data_cfg.get('eval_tasks', cfg.train_tasks)

    # Graph configuration
    cfg.graphs_dir = data_cfg.get('graphs_dir', 'data/processed/graphs_v2/')
    cfg.input_dim = model_cfg.get('input_dim', 154)
    cfg.feature_key = data_cfg.get('feature_key', 'x')

    print(f"Loaded Configuration: {config}")
    print(f"Final Config: {vars(cfg)}")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Task configuration
    task_files = {
        'side_effects': "data/processed/cid_se_matrix.csv",
        'atc': "data/processed/cid_atc_l3_matrix.csv",
        'maccs': "data/processed/cid_maccs_matrix.csv",
        'molprops': "data/processed/cid_molprops_matrix_simple.csv"
    }

    # Load dataset
    print(f"Loading graph dataset...")
    dataset = GraphDataset(
        cfg.graphs_dir,
        task_config=task_files,
        feature_key=cfg.feature_key,
        max_side_effects=cfg.max_side_effects
    )
    train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # Create model
    model, loss_funcs, task_weights, train_tasks, eval_tasks = setup_model(dataset, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    
    # Add cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr/100)

    # Setup experiment logger
    logger = ExperimentLogger(cfg, log_dir=cfg.log_dir, checkpoint_dir=cfg.checkpoint_dir)
    logger.log_hyperparameters(task_weights)

    print(f"\nStarting training loop for {cfg.epochs} epochs...")
    print(f"Train tasks: {', '.join(train_tasks)}")
    print(f"Eval tasks: {', '.join(eval_tasks)}")
    print(f"Task weights: {task_weights}")
    if cfg.clip_grad_norm:
        print(f"Gradient clipping: max_norm={cfg.clip_grad_norm}")
    
    # Determine primary metric based on primary eval task
    primary_task = eval_tasks[0]
    if primary_task == 'molprops':
        print(f"Metrics: R² (primary), MSE, MAE")
    elif primary_task == 'maccs':
        print(f"Metrics: Bit Accuracy (primary), Hamming, Tanimoto")
    else:
        print(f"Metrics: mAP (primary), P@50, P@100, AUROC")
    print("-" * 80)

    # Early stopping (disabled for fair experiment comparison)
    best_val_metric = float('-inf')
    patience = 10000  # Effectively disabled
    patience_counter = 0

    for epoch in range(cfg.epochs):
        train_loss, train_task_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_funcs, task_weights, eval_tasks, device, cfg.clip_grad_norm
        )
        val_loss, val_task_losses, val_metrics = validate(
            model, val_loader, loss_funcs, task_weights, eval_tasks, device
        )

        # Log metrics to TensorBoard
        avg_train_loss, avg_val_loss = logger.log_epoch(
            epoch, train_loss, val_loss, train_task_losses, val_task_losses,
            train_metrics, val_metrics, len(train_ds), len(val_ds), optimizer
        )

        # Print to console - use first eval task's metrics
        primary_eval_task = eval_tasks[0]
        train_str = format_metrics_string(train_metrics.get(primary_eval_task, {}))
        val_str = format_metrics_string(val_metrics.get(primary_eval_task, {}))

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Train: {train_str} | Val: {val_str}"
        )

        # Print per-task losses every 10 epochs
        if (epoch + 1) % 10 == 0 and len(train_task_losses) > 1:
            task_loss_str = format_task_losses(train_task_losses, len(train_ds))
            print(f"  Task losses: {task_loss_str}")

        # Save checkpoint if best (use primary eval task's primary metric)
        task_metrics = val_metrics.get(primary_eval_task, {})
        
        # Determine primary metric based on task type
        if 'MSE' in task_metrics:
            # Regression: lower MSE is better, so negate for comparison
            current_val_metric = -task_metrics.get('MSE', float('inf'))
            metric_name = 'MSE'
            metric_value = task_metrics.get('MSE', 0)
        elif 'Hamming' in task_metrics:
            # Fingerprint: lower Hamming is better, use Tanimoto (higher is better)
            current_val_metric = task_metrics.get('Tanimoto', 0)
            metric_name = 'Tanimoto'
            metric_value = task_metrics.get('Tanimoto', 0)
        else:
            # Classification: higher mAP is better
            current_val_metric = task_metrics.get('mAP', 0)
            metric_name = 'mAP'
            metric_value = task_metrics.get('mAP', 0)
        
        if logger.save_checkpoint(epoch, model, optimizer, current_val_metric, metric_name):
            print(f"    Saved best model ({metric_name}: {metric_value:.4f})")
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={patience})")
                break

    # Finalize logging
    logger.finalize(val_metrics)


if __name__ == "__main__":
    main()

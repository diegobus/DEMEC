import sys
import os
import argparse
import torch
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset, make_splits as make_splits_homo
from demec.data.hetero_data_loader import HeteroGraphDataset, make_splits as make_splits_hetero
from demec.utils.eval_metrics import comprehensive_metrics
from demec.utils.logger import ExperimentLogger, format_metrics_string, format_task_losses
from demec.models.gnn_backbone import GNNBackbone
from demec.models.hetero_gnn import HeteroGNNBackbone, HeteroMultiTaskGNN
from demec.models.multitask import MultiTaskGNN
from demec.models.heads import PredictionHead


def create_model(args, dataset, device):
    """
    Factory function to create model based on graph type and architecture.
    
    Args:
        args: Command line arguments
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
    if args.task_weights:
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
                focal_alpha = args.focal_alpha if args.focal_alpha is not None else 0.25
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
    
    # Graph type selection
    parser.add_argument("--hetero", action="store_true", 
                        help="Use heterogeneous graphs with bond-type-specific edges")
    
    # Model selection
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "gat"], 
                        help="Model architecture to use")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Data hyperparameters
    parser.add_argument("--graphs_dir", type=str, default=None, 
                        help="Path to graphs directory (auto-detected if not specified)")
    parser.add_argument("--feature_key", type=str, default=None, 
                        help="Key for node features in graph objects")
    
    # Model hyperparameters
    parser.add_argument("--input_dim", type=int, default=None, 
                        help="Input feature dimension (auto-detected if not specified)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--heads", type=int, default=3, help="Number of attention heads (GAT only)")
    
    # Task selection and weighting (for ablation studies)
    parser.add_argument("--tasks", type=str, default="side_effects,atc,maccs",
                        help="Comma-separated list of tasks to train (e.g., 'side_effects' or 'side_effects,maccs,molprops')")
    parser.add_argument("--task_weights", type=str, default=None,
                        help="Task weights as 'task1:weight1,task2:weight2' (e.g., 'side_effects:1.0,atc:0.3,maccs:0.1')")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Alpha parameter for Focal Loss (default: 0.25). Try 0.75 for rare positives")
    parser.add_argument("--clip_grad_norm", type=float, default=None,
                        help="Gradient clipping max norm (e.g., 1.0). None = no clipping")
    
    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="Directory for TensorBoard logs (default: runs)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for logging (default: auto-generated)")
    parser.add_argument("--save_model", action="store_true",
                        help="Save best model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints (default: checkpoints)")
    
    args = parser.parse_args()
    
    # Auto-detect graph directory and input dimension based on graph type
    if args.graphs_dir is None:
        args.graphs_dir = "data/processed/graphs_v2/" if args.hetero else "data/processed/graphs/"
    
    if args.input_dim is None:
        args.input_dim = 154 if args.hetero else 1
    
    if args.feature_key is None:
        args.feature_key = 'x' if args.hetero else None
    
    print(f"Configuration: {args}")
    print(f"Graph Type: {'Heterogeneous' if args.hetero else 'Homogeneous'}")
    
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
    print(f"Loading {'heterogeneous' if args.hetero else 'homogeneous'} graph dataset...")
    if args.hetero:
        dataset = HeteroGraphDataset(
            args.graphs_dir,
            task_config=task_config,
            feature_key=args.feature_key
        )
        train_ds, val_ds, test_ds = make_splits_hetero(dataset, train=0.8, val=0.1, seed=args.seed)
    else:
        dataset = GraphStructureDataset(
            args.graphs_dir,
            task_config=task_config,
            node_dim=args.input_dim,
            feature_key=args.feature_key
        )
        train_ds, val_ds, test_ds = make_splits_homo(dataset, train=0.8, val=0.1, seed=args.seed)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Create model
    model, loss_funcs, task_weights = create_model(args, dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup experiment logger
    logger = ExperimentLogger(args, log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir)
    logger.log_hyperparameters(task_weights)
    
    print(f"\nStarting training loop for {args.epochs} epochs...")
    print(f"Active tasks: {', '.join(loss_funcs.keys())}")
    print(f"Task weights: {task_weights}")
    if args.clip_grad_norm:
        print(f"Gradient clipping: max_norm={args.clip_grad_norm}")
    print(f"Metrics: mAP (primary), P@50, P@100, AUROC")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_task_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_funcs, task_weights, device, args.clip_grad_norm
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

import sys
import os
import argparse
import torch
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset, make_splits as make_splits_homo
from demec.data.hetero_data_loader import HeteroGraphDataset, make_splits as make_splits_hetero
from demec.utils.eval_metrics import comprehensive_metrics
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
    """
    heads_dict = {}
    loss_funcs = {}
    
    # Initialize prediction heads (same for both graph types)
    for task_name, dim in dataset.task_dims.items():
        print(f"Initializing head for task: {task_name} (output_dim={dim})")
        
        loss_type = "focal" if task_name == "side_effects" else "bce"
        
        head = PredictionHead(
            input_dim=args.hidden_dim,
            output_dim=dim,
            hidden_dims=[args.hidden_dim],
            dropout=args.dropout,
            task_type="classification",
            loss_type=loss_type
        )
        heads_dict[task_name] = head
        loss_funcs[task_name] = head.get_loss_func()
    
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
    
    return model.to(device), loss_funcs


def train_epoch(model, loader, optimizer, loss_funcs, device):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
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
                batch_loss += loss
                
                # Collect side_effects predictions for metrics
                if task_name == 'side_effects':
                    all_se_logits.append(logits.detach())
                    all_se_targets.append(target.detach())
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item() * batch.num_graphs
    
    # Compute metrics on all accumulated predictions
    if all_se_logits:
        all_se_logits = torch.cat(all_se_logits, dim=0)
        all_se_targets = torch.cat(all_se_targets, dim=0)
        metrics = comprehensive_metrics(all_se_logits, all_se_targets, k_values=[50, 100])
    else:
        metrics = {}
    
    return total_loss, metrics


def validate(model, loader, loss_funcs, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
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
                    batch_loss += loss
                    
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
    
    return total_loss, metrics


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
        'maccs': "data/processed/cid_maccs_matrix.csv"
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
    model, loss_funcs = create_model(args, dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training loop for {args.epochs} epochs...")
    print(f"Metrics: mAP (primary), P@50, P@100, AUROC")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_funcs, device)
        val_loss, val_metrics = validate(model, val_loader, loss_funcs, device)
        
        avg_train_loss = train_loss / len(train_ds)
        avg_val_loss = val_loss / len(val_ds)
        
        # Format metrics for display
        train_str = f"mAP:{train_metrics.get('mAP', 0):.3f} P@50:{train_metrics.get('P@50', 0):.3f} P@100:{train_metrics.get('P@100', 0):.3f} AUROC:{train_metrics.get('AUROC', 0):.3f}"
        val_str = f"mAP:{val_metrics.get('mAP', 0):.3f} P@50:{val_metrics.get('P@50', 0):.3f} P@100:{val_metrics.get('P@100', 0):.3f} AUROC:{val_metrics.get('AUROC', 0):.3f}"
        
        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Train: {train_str} | Val: {val_str}"
        )


if __name__ == "__main__":
    main()

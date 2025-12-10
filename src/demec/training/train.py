import sys
import os
import argparse
import torch
import yaml
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset, make_splits
from demec.utils.eval_metrics import recall_at_all
from demec.models.gcn_baseline import GCNBackbone
from demec.models.gat import GATBackbone
from demec.models.multitask import MultiTaskGNN
from demec.models.heads import PredictionHead

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train Graph Models (GCN or GAT)")
    
    # Config file argument
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Optional CLI overrides (can override config values)
    parser.add_argument("--model", type=str, choices=["gcn", "gat"], help="Model architecture")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args if provided
    if args.model: config['model']['architecture'] = args.model
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.lr: config['training']['lr'] = args.lr
    if args.seed: config['training']['seed'] = args.seed
    
    print(f"Loaded Configuration: {config}")
    
    # Extract params for cleaner code
    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['data']
    
    # Set seed
    torch.manual_seed(train_cfg['seed'])
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    graphs_dir = data_cfg['graphs_dir']
    
    # Define tasks configuration from YAML
    task_config = data_cfg.get('tasks', {})
    if not task_config:
        print("Warning: No tasks defined in configuration!")
    
    print("Loading dataset...")
    # Handle optional feature_key
    feature_key = data_cfg.get('feature_key', None)
    
    dataset = GraphStructureDataset(graphs_dir, task_config=task_config, 
                                    node_dim=model_cfg['node_dim'], 
                                    feature_key=feature_key)
    
    train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=train_cfg['seed'])

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=train_cfg['batch_size'])

    # Initialize Backbone
    architecture = model_cfg['architecture']
    hidden_dim = model_cfg['hidden_dim']
    num_layers = model_cfg['num_layers']
    dropout = model_cfg['dropout']
    
    if architecture == "gcn":
        backbone = GCNBackbone(
            input_dim=model_cfg['node_dim'],
            out_dim=None, # Return embeddings
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout
        )
    elif architecture == "gat":
        backbone = GATBackbone(
            input_dim=model_cfg['node_dim'],
            output_dim=None, # Return embeddings
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=model_cfg.get('heads', 1),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    
    # Initialize Heads
    heads_dict = {}
    loss_funcs = {}
    
    for task_name, dim in dataset.task_dims.items():
        print(f"Initializing head for task: {task_name} (output_dim={dim})")
        
        # Use Focal Loss for side_effects to handle rare classes
        # This logic is kept from original script, but could be moved to config in future
        loss_type = "focal" if task_name == "side_effects" else "bce"
        
        head = PredictionHead(
            input_dim=hidden_dim,
            output_dim=dim,
            hidden_dims=[hidden_dim],
            dropout=dropout,
            task_type="classification",
            loss_type=loss_type
        )
        heads_dict[task_name] = head
        loss_funcs[task_name] = head.get_loss_func()
        
    model = MultiTaskGNN(backbone, heads_dict)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])

    print(f"Starting training loop for {train_cfg['epochs']} epochs...")

    for epoch in range(train_cfg['epochs']):
        model.train()
        total_train_loss = 0.0
        total_recall = 0.0 # Specifically for side_effects
        
        for batch in train_loader:
            batch = batch.to(device)
            
            results = model(batch)
            
            batch_loss = 0.0
            for task_name, logits in results.items():
                target_attr = f"y_{task_name}"
                mask_attr = f"mask_{task_name}"
                
                if hasattr(batch, target_attr):
                    target = getattr(batch, target_attr)
                    
                    # Apply mask if present
                    if hasattr(batch, mask_attr):
                        mask = getattr(batch, mask_attr).squeeze()
                        if not mask.any():
                            continue
                        target = target[mask]
                        logits = logits[mask]
                    
                    loss = loss_funcs[task_name](logits, target)
                    batch_loss += loss
                    
                    # Specific metrics for side_effects
                    if task_name == 'side_effects':
                        total_recall += recall_at_all(logits, target)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_train_loss += batch_loss.item() * batch.num_graphs

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_recall = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                results = model(batch)
                
                batch_val_loss = 0.0
                for task_name, logits in results.items():
                    target_attr = f"y_{task_name}"
                    mask_attr = f"mask_{task_name}"
                    
                    if hasattr(batch, target_attr):
                        target = getattr(batch, target_attr)
                        
                        # Apply mask if present
                        if hasattr(batch, mask_attr):
                            mask = getattr(batch, mask_attr).squeeze()
                            if not mask.any():
                                continue
                            target = target[mask]
                            logits = logits[mask]
                            
                        loss = loss_funcs[task_name](logits, target)
                        batch_val_loss += loss
                        
                        if task_name == 'side_effects':
                            val_recall += recall_at_all(logits, target)
                            
                val_loss += batch_val_loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_ds)
        avg_train_recall = total_recall / len(train_ds)
        avg_val_loss = val_loss / len(val_ds)
        avg_val_recall = val_recall / len(val_ds)

        print(
            f"Epoch {epoch+1} | "
            f"train loss: {avg_train_loss:.4f} | train recall: {avg_train_recall:.4f} | "
            f"val loss: {avg_val_loss:.4f} | val recall: {avg_val_recall:.4f}"
        )

if __name__ == "__main__":
    main()
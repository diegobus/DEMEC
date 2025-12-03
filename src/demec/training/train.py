import sys
import os
import argparse
import torch
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset, make_splits
from demec.utils.eval_metrics import recall_at_all
from demec.models.gcn_baseline import GCNBackbone
from demec.models.gat import GATBackbone
from demec.models.multitask import MultiTaskGNN
from demec.models.heads import PredictionHead

def main():
    parser = argparse.ArgumentParser(description="Train Graph Models (GCN or GAT)")
    
    # Model selection
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "gat"], 
                        help="Model architecture to use")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Data hyperparameters
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs/", 
                        help="Relative path to graphs directory")
    parser.add_argument("--feature_key", type=str, default=None, 
                        help="Key for node features in graph objects (e.g., 'emb')")

    # Model hyperparameters
    parser.add_argument("--node_dim", type=int, default=1, help="Dimension of node features")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--heads", type=int, default=3, help="Number of attention heads (GAT only)")
    
    args = parser.parse_args()
    
    print(f"Configuration: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    # Assuming we run from the project root
    graphs_dir = args.graphs_dir
    cid_se_csv = "data/processed/cid_se_matrix.csv"
    
    # Define tasks configuration
    # ----- Add more tasks here ------
    task_config = {
        'side_effects': cid_se_csv
    }
    
    print("Loading dataset...")
    dataset = GraphStructureDataset(graphs_dir, task_config=task_config, node_dim=args.node_dim, feature_key=args.feature_key)
    train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Initialize Backbone
    if args.model == "gcn":
        backbone = GCNBackbone(
            input_dim=args.node_dim,
            out_dim=None, # Return embeddings
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        )
    elif args.model == "gat":
        backbone = GATBackbone(
            input_dim=args.node_dim,
            output_dim=None, # Return embeddings
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout
        )
    
    # Initialize Heads
    heads_dict = {}
    loss_funcs = {}
    
    for task_name, dim in dataset.task_dims.items():
        print(f"Initializing head for task: {task_name} (output_dim={dim})")
        head = PredictionHead(
            input_dim=args.hidden_dim,
            output_dim=dim,
            hidden_dims=[args.hidden_dim],
            dropout=args.dropout,
            task_type="classification" # Assuming classification for now
        )
        heads_dict[task_name] = head
        loss_funcs[task_name] = head.get_loss_func()
        
    model = MultiTaskGNN(backbone, heads_dict)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training loop for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_recall = 0.0 # Specifically for side_effects
        
        for batch in train_loader:
            batch = batch.to(device)
            
            results = model(batch)
            
            batch_loss = 0.0
            for task_name, logits in results.items():
                target_attr = f"y_{task_name}"
                if hasattr(batch, target_attr):
                    target = getattr(batch, target_attr)
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
                    if hasattr(batch, target_attr):
                        target = getattr(batch, target_attr)
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
import sys
import os
import argparse
import torch
from torch_geometric.loader import DataLoader

# Add the project root directory to the Python path to allow imports
# This assumes the script is located at src/training/train.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import from scripts and src
from scripts.data_loader import GraphStructureDataset, make_splits
from scripts.eval_metrics import recall_at_all
from src.models.gcn_baseline import StructuralGCN
from src.models.gat import StructuralGAT

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
    
    # Model hyperparameters
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
    graphs_dir = os.path.join(project_root, "data/processed/graphs/")
    cid_se_csv = os.path.join(project_root, "data/processed/cid_se_matrix.csv")
    
    print("Loading dataset...")
    dataset = GraphStructureDataset(graphs_dir, cid_se_csv)
    train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    n_labels = len(dataset.se_cols)
    node_dim = dataset.node_dim

    # Initialize Model
    if args.model == "gcn":
        model = StructuralGCN(
            out_dim=n_labels, 
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        )
    elif args.model == "gat":
        model = StructuralGAT(
            input_dim=node_dim,
            output_dim=n_labels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout
        )
    
    model = model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training loop for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_recall = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Handle forward pass differences
            if args.model == "gcn":
                logits = model(batch.x, batch.edge_index, batch.batch)
            else:
                logits = model(batch)
                
            loss = loss_fn(logits, batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            total_recall += recall_at_all(logits, batch.y)

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_recall = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                if args.model == "gcn":
                    logits = model(batch.x, batch.edge_index, batch.batch)
                else:
                    logits = model(batch)
                    
                loss = loss_fn(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                val_recall += recall_at_all(logits, batch.y)

        avg_train_loss = total_loss / len(train_ds)
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
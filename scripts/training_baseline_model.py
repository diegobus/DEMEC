import torch
from torch_geometric.loader import DataLoader
from scripts.data_loader import GraphStructureDataset, make_splits
from scripts.models.gcn_baseline import StructuralGCN  
from scripts.eval_metrics import recall_at_all

graphs_dir = "data/processed/graphs/"
cid_se_csv = "data/processed/cid_se_matrix.csv" 

dataset = GraphStructureDataset(graphs_dir, cid_se_csv) 
train_ds, val_ds, test_ds = make_splits(dataset, train=0.8, val=0.1, seed=42)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)
test_loader  = DataLoader(test_ds,  batch_size=32)

n_labels = len(dataset.se_cols)
model = StructuralGCN(out_dim=n_labels, hidden_dim=64, num_layers=5, dropout=0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_fn = torch.nn.BCEWithLogitsLoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Starting Epochs")
for epoch in range(10):  
    model.train()
    total = 0.0
    recall = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(logits, batch.y)                             
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
        recall += recall_at_all(logits, batch.y)
        # if epoch == 9:
        #     print(batch.y)
        #     print(logits)

    model.eval()
    total_eval = 0.0
    recall_eval = 0.0
    for batch in val_loader: 
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(logits, batch.y)
        total_eval += loss.item() * batch.num_graphs
        recall_eval += recall_at_all(logits, batch.y)

    print(f"Epoch {epoch+1} | train loss: {total/len(train_ds):.4f} | train recall: {recall/len(train_ds):.4f} | val loss: {total_eval/len(val_ds):.4f} | val recall: {recall_eval/len(val_ds):.4f}" )
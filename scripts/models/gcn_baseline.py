import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

class StructuralGCN(nn.Module):
    """
    GCN using only graph structure
    """

    def __init__(self, out_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2):
        
        super().__init__()
        self.dropout = dropout

        # Only 1 dimension since using only structural information 
        in_dim = 1

        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)] + 
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True) for _ in range(num_layers - 1)])
        
        self.head = nn.Linear(hidden_dim, out_dim)

        
    

    def forward(self, x, edge_index, batch):
        
        # No embeddings for current structure-only approach
        num_nodes = batch.size(0)
        x = torch.ones((num_nodes, 1), device=batch.device)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_add_pool(x, batch)
        logits = self.head(pooled)
        return logits

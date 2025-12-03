import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

class GCNBackbone(nn.Module):
    """
    GCN backbone for graph embedding.
    Supports configurable input dimension for node features.
    """

    def __init__(self, input_dim: int = 1, out_dim: int = None, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2):
        
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True)] + 
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True) for _ in range(num_layers - 1)])
        
        self.out_dim = out_dim
        if out_dim is not None:
            self.head = nn.Linear(hidden_dim, out_dim)
        else:
            self.head = None

        
    

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_add_pool(x, batch)
        
        if self.out_dim is None:
            return pooled
            
        logits = self.head(pooled)
        return logits

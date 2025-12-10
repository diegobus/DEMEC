import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, GCNConv, global_mean_pool
from torch_geometric.data import HeteroData


class GNNBackbone(nn.Module):
    """
    GNN backbone that handles different bond types as edge types.
    Uses HeteroConv to apply different message passing for each bond type.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=5,
        dropout=0.2,
        conv_type='gat',
        heads=3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.heads = heads
        
        # Bond types we expect
        self.bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            
            for bond_type in self.bond_types:
                edge_type = ('atom', bond_type, 'atom')
                
                if conv_type == 'gat':
                    # Use concat=False to get mean aggregation instead of concat
                    # This keeps output dimension = hidden_dim
                    conv_dict[edge_type] = GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False
                    )
                elif conv_type == 'gcn':
                    conv_dict[edge_type] = GCNConv(
                        hidden_dim,
                        hidden_dim,
                        add_self_loops=False
                    )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data: HeteroData):
        """
        Forward pass for heterogeneous graph.
        
        Args:
            data: HeteroData object with 'atom' nodes and bond-typed edges
            
        Returns:
            Graph-level embedding tensor of shape (batch_size, hidden_dim)
        """
        x_dict = {'atom': self.input_proj(data['atom'].x)}
        
        # Build edge_index_dict from available edge types in the batch
        edge_index_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # Apply heterogeneous convolutions
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Handle case where no edges exist (HeteroConv returns empty dict)
            if 'atom' in x_dict_new:
                x_dict['atom'] = norm(x_dict_new['atom'])
                x_dict['atom'] = torch.relu(x_dict['atom'])
                x_dict['atom'] = self.dropout_layer(x_dict['atom'])
            # If no edges, keep previous features (skip this layer)
        
        # Global pooling
        x = x_dict['atom']
        
        # Handle batching
        if hasattr(data['atom'], 'batch'):
            batch = data['atom'].batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Global mean pooling
        graph_emb = global_mean_pool(x, batch)
        
        return graph_emb


class MultiTaskGNN(nn.Module):
    """
    Multi-task GNN with separate prediction heads.
    """
    
    def __init__(self, backbone, heads_dict):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads_dict)
    
    def forward(self, data: HeteroData):
        """
        Forward pass through backbone and all task heads.
        
        Returns:
            Dictionary mapping task names to predictions
        """
        graph_emb = self.backbone(data)
        
        results = {}
        for task_name, head in self.heads.items():
            results[task_name] = head(graph_emb)
        
        return results

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, scatter


class GNNBackbone(nn.Module):
    """
    Unified GNN backbone supporting both GCN and GAT architectures.
    Returns graph-level embeddings for downstream tasks.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        conv_type='gcn',
        heads=3,
        output_dim=None,
        dropout=0.2,
        negative_slope=0.2
    ):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            conv_type: 'gcn' or 'gat'
            heads: Number of attention heads (GAT only)
            output_dim: Output dimension (None returns embeddings)
            dropout: Dropout rate
            negative_slope: LeakyReLU negative slope (GAT only)
        """
        super().__init__()
        
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        
        if conv_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True))
        
        elif conv_type == 'gat':
            self.convs.append(GATLayer(input_dim, hidden_dim, heads=heads, negative_slope=negative_slope))
            for _ in range(num_layers - 1):
                self.convs.append(GATLayer(heads * hidden_dim, hidden_dim, heads=heads, negative_slope=negative_slope))
            
            # Project multi-head output to hidden_dim
            self.proj_heads = nn.Linear(heads * hidden_dim, hidden_dim)
        
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}. Choose 'gcn' or 'gat'.")
        
        # Optional output head
        if output_dim is not None:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.head = None
    
    def forward(self, data):
        """
        Forward pass through GNN layers and pooling.
        
        Args:
            data: PyG Data object with x, edge_index, batch
            
        Returns:
            Graph-level embeddings or predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Project GAT heads if needed
        if self.conv_type == 'gat':
            x = self.proj_heads(x)
        
        # Return embeddings or apply output head
        if self.output_dim is None:
            return x
        
        return self.head(x)


class GATLayer(MessagePassing):
    """
    Graph Attention Layer with multi-head attention.
    Adapted from CS224W, Colab 4 in Fall 2025.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=2,
        negative_slope=0.2,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin_l = nn.Linear(in_channels, out_channels * heads)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(torch.Tensor(heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(heads, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
    
    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels
        
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)
        
        out = self.propagate(
            edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size
        )
        out = out.view(-1, H * C)
        
        return out
    
    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)
    
    def aggregate(self, inputs, index, dim_size=None):
        return scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum"
        )

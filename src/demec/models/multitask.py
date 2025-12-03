import torch
import torch.nn as nn

class MultiTaskGNN(nn.Module):
    def __init__(self, backbone, heads):
        """
        Args:
            backbone: A GNN model that outputs a graph embedding (shape: [batch_size, embed_dim]).
            heads: A dictionary of {task_name: PredictionHead_instance}.
        """
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        
    def forward(self, data):
        # Get shared embedding from backbone
        # We assume the backbone's forward method returns the embedding when configured correctly
        embedding = self.backbone(data)
        
        results = {}
        for task_name, head in self.heads.items():
            results[task_name] = head(embedding)
            
        return results

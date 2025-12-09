import torch
import torch.nn as nn
from demec.utils.losses import FocalLoss

class PredictionHead(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_dims=[64], 
        dropout=0.2, 
        task_type="classification",
        loss_type="bce",
        focal_alpha=0.25
    ):
        """
        Args:
            input_dim: Size of the input embedding from the backbone.
            output_dim: Number of output classes/values.
            hidden_dims: List of hidden layer sizes for the MLP.
            dropout: Dropout rate.
            task_type: "classification" or "regression".
            loss_type: "bce" or "focal".
            focal_alpha: Alpha parameter for Focal Loss (default: 0.25).
        """
        super().__init__()
        self.task_type = task_type
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_loss_func(self):
        if self.task_type == "classification":
            if self.loss_type == "focal":
                return FocalLoss(alpha=self.focal_alpha, gamma=2.0)
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

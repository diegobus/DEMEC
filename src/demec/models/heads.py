import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_dims=[64], 
        dropout=0.2, 
        task_type="classification"
    ):
        """
        Args:
            input_dim: Size of the input embedding from the backbone.
            output_dim: Number of output classes/values.
            hidden_dims: List of hidden layer sizes for the MLP.
            dropout: Dropout rate.
            task_type: "classification" or "regression".
        """
        super().__init__()
        self.task_type = task_type
        
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
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

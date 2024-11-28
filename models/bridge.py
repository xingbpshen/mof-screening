import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self, in_dim: int, out_dims: list, dropout_rate: float):
        """
        Initializes the Bridge module.

        Args:
            in_dim (int): Dimension of the input embedding.
            out_dims (list): A list where each entry specifies the output dimension for a prediction.
                             e.g., out_dims=[1, 1] means two scalar predictions.
        """
        super(Bridge, self).__init__()

        # Create a separate fully connected layer for each output dimension
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),  # Hidden layer
                nn.Sigmoid(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_dim // 2, out_dim)  # Output layer
            )
            for out_dim in out_dims
        ])

    def forward(self, embed):
        """
        Forward pass to generate multiple predictions from a common embedding.

        Args:
            embed (torch.Tensor): The input embedding of shape (batch_size, in_dim).

        Returns:
            list[torch.Tensor]: A list of predictions, each with shape (batch_size, out_dim).
        """
        predictions = [predictor(embed) for predictor in self.predictors]
        return predictions

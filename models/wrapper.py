import torch
from models.resnet import resnet18
from models.masked_resnet import resnet18 as masked_resnet18
from models.sparse import SparseNet


class Model(torch.nn.Module):
    def __init__(self, args, config):
        super(Model, self).__init__()
        if config.model.type == 'resnet18':
            self.model = resnet18(dropout_rate=config.model.dropout_rate)
        elif config.model.type == 'sparse':
            self.model = SparseNet(dropout_rate=config.model.dropout_rate, in_channels=config.data.channels)
        elif config.model.type == 'masked_resnet18':
            self.model = masked_resnet18(dropout_rate=config.model.dropout_rate)
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        self.mc_dropout = args.mc_dropout

    def forward(self, x):
        wc, sel = self.model(x)
        return wc, sel

    def sample(self, x, n_samples):
        """
        Samples predictions using Monte Carlo Dropout while keeping other stochastic layers (e.g., BatchNorm) disabled.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            n_samples (int): Number of stochastic forward passes.

        Returns:
            wcs (torch.Tensor): Collected `wc` predictions of shape (batch_size, n_samples).
            sels (torch.Tensor): Collected `sel` predictions of shape (batch_size, n_samples).
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Enable only Dropout layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        # Perform `n_samples` stochastic forward passes
        wcs = []
        sels = []
        for _ in range(n_samples):
            with torch.no_grad():  # No gradient computation
                wc, sel = self.model(x)  # Forward pass
                wcs.append(wc)  # Collect wc
                sels.append(sel)  # Collect sel
        # Stack predictions along a new dimension
        wcs = torch.stack(wcs, dim=1)  # Shape: (batch_size, n_samples)
        sels = torch.stack(sels, dim=1)  # Shape: (batch_size, n_samples)

        # Change the model back to evaluation mode
        self.model.eval()

        return wcs, sels

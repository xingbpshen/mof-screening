import torch.optim as optim
import torch
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt


# Code adapted from DDIM by Song et al. (https://github.com/ermongroup/ddim)
def get_optimizer(config, parameters):
    if config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    else:
        raise NotImplementedError(f"Invalid optimizer {config.optim.optimizer}.")


def inverse_normalization(x, mean, std):
    return x * std + mean


# Compute data densities using Kernel Density Estimation (KDE)
def compute_densities(x, bandwidth=0.5):
    # Check if x is np.ndarray, if not, change to np.ndarray
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    x = x.reshape(-1, 1)
    kde.fit(x)
    log_densities = kde.score_samples(x)
    densities = torch.tensor(np.exp(log_densities), dtype=torch.float32)
    return densities


# def compute_weights(x, bandwidth=0.5):
#     densities = compute_densities(x, bandwidth)
#     # Reweight the loss using inverse densities
#     weights = 1.0 / (densities + 1e-6)  # Add epsilon for stability
#     # Scale the weights to have a maximum of 1
#     weights /= torch.max(weights)
#     # Add epsilon for stability
#     weights += 1e-6
#     return weights
def compute_weights(x, bandwidth=0.5, lower_percentile=5, upper_percentile=95):
    densities = compute_densities(x, bandwidth)
    # Reweight the loss using inverse densities
    weights = 1.0 / (densities + 1e-6)  # Add epsilon for stability
    # Convert to numpy array for percentile computation
    weights_np = weights.cpu().numpy()
    # Compute clipping thresholds based on percentiles
    lower_bound = np.percentile(weights_np, lower_percentile)
    upper_bound = np.percentile(weights_np, upper_percentile)
    # Clip weights to the computed bounds
    weights = torch.clamp(weights, min=lower_bound, max=upper_bound)
    # Normalize weights to have a mean of 1
    weights = weights / torch.mean(weights)
    # Detach weights from the computation graph
    weights = weights.detach()

    return weights


def add_noise(x, noise_level, clip=True):
    # Add positive Gaussian noise to the input, clip to [0, 1] if clip=True
    noise = torch.randn_like(x) * noise_level
    x_noisy = x + torch.abs(noise)
    if clip:
        x_noisy = torch.clamp(x_noisy, 0, 1)
    return x_noisy


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - ss_res / ss_tot
    return r2


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def plot_gt_pred(y_true: np.ndarray, y_pred: np.ndarray, variable_name=None, save_path=None) -> None:
    if variable_name is None:
        variable_name = "Value"
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'black', label='Reference line')

    # Change the font size
    plt.rcParams.update({'font.size': 20})
    # Increase the thickness of the lines
    plt.rcParams.update({'lines.linewidth': 2})

    # Add labels and title
    plt.xlabel(f'Ground-truth {variable_name}', fontsize=20)
    plt.ylabel(f'Predicted {variable_name}', fontsize=20)
    # plt.title('Ground-truth vs Predicted Values')
    plt.legend()

    # Adjust layout to prevent cropping of labels
    plt.tight_layout()
    if save_path is not None:
        # Save the plot
        plt.savefig(save_path, format='pdf', transparent=True)


def plot_x(x: torch.tensor, save_path=None):
    """
    Plots a 3-channel 64x64 image tensor and saves the plot as a PDF.

    Args:
        x (torch.tensor): A 3-channel 64x64 image tensor with values in the range [0, 1].
        filename (str): The name of the output PDF file. Defaults to "image_plot.pdf".
    """
    # Ensure input tensor is the correct shape and range
    if x.ndim != 3 or x.shape[0] != 3 or x.shape[1] != 64 or x.shape[2] != 64:
        raise ValueError("Input tensor must have shape [3, 64, 64].")
    if torch.min(x) < 0 or torch.max(x) > 1:
        raise ValueError("Tensor values must be in the range [0, 1].")

    # Convert to numpy for plotting
    x_np = x.clone().detach().cpu().numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        im = axes[i].imshow(x_np[i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f"Channel {i + 1}")
        axes[i].axis("off")

    # Add a colorbar to indicate the range
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Value')

    # Save the figure as a PDF
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', transparent=True)
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = compute_r2(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    return {"r2": r2, "mae": mae, "rmse": rmse}

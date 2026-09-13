"""
Utility functions for time series processing and visualization.
Includes plotting, model visualization, and data display utilities.
"""
import os
from typing import List, Optional, Union, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


def sample_posterior(mu: torch.Tensor, L: torch.Tensor, mask=None):
    if mask is not None:
        mu = (mu * mask)
        diagonal_mask = torch.diag_embed(mask).float()
        L = torch.bmm(L, diagonal_mask)
    mu = mu.unsqueeze(-1)
    z = mu + torch.bmm(L, torch.randn_like(mu))
    return z.squeeze(-1)


def sample_posterior_correlated(mu_psi, L_psi, mu_phi, L_phi, rho=1):
    mu_psi = mu_psi.unsqueeze(-1)
    mu_phi = mu_phi.unsqueeze(-1)
    epsilon = torch.randn_like(mu_psi)
    z_psi = mu_psi + L_psi @ epsilon
    z_phi = mu_phi + L_phi @ epsilon
    return z_psi.squeeze(-1), z_phi.squeeze(-1)


def construct_cholesky(L_elements, latent_dim):
    """
    Constructs a lower triangular matrix L from its elements.

    Args:
        L_elements (torch.Tensor): Elements of the lower triangular matrix.
        latent_dim (int): Dimension of the latent space.

    Returns:
        torch.Tensor: Lower triangular matrix L of shape (batch_size, latent_dim, latent_dim).
    """
    device = L_elements.device
    L = torch.zeros(L_elements.size(0), latent_dim, latent_dim, device=device)
    indices = torch.tril_indices(row=latent_dim, col=latent_dim, offset=0, device=device)
    L[:, indices[0], indices[1]] = L_elements
    diag_indices = torch.arange(latent_dim, device=device)
    # Ensure positivity with sharp softplus
    L[:, diag_indices, diag_indices] = F.softplus(
        L[:, diag_indices, diag_indices], beta=5.0, threshold=20.0
    )
    return L



def display_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy for display purposes."""
    return tensor.detach().cpu().numpy()


def plot_latent_variables(z: torch.Tensor, figsize: Tuple[int, int] = (8, 12)) -> None:
    """
    Plot latent variables for each batch sample.

    Args:
        z: Latent variables tensor of shape (batch_size, sequence_length)
        figsize: Figure size for the plot
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting latent variables")

    batch_size, sequence_length = z.shape
    fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=figsize)

    # Handle single sample case
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        axes[i].plot(z[i, :].detach().cpu().numpy())
        axes[i].set_title(f'Latent Variables - Sample {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Latent Value')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trajectory_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sample_indices: List[int] = [0],
    datalength: Optional[torch.Tensor] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot comparison between original and reconstructed trajectories.

    Args:
        original: Original trajectories (batch_size, time_steps, num_channels)
        reconstructed: Reconstructed trajectories (batch_size, time_steps, num_channels)
        sample_indices: List of sample indices to plot
        datalength: Valid length for each sample
        figsize: Figure size, auto-calculated if None
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting trajectory comparisons")

    batch_size, time_steps, num_channels = original.shape
    num_plots = len(sample_indices)

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (4 * num_channels, 3 * num_plots)

    fig, axes = plt.subplots(num_plots, num_channels, figsize=figsize)

    # Handle single plot cases
    if num_plots == 1 and num_channels == 1:
        axes = [[axes]]
    elif num_plots == 1:
        axes = [axes]
    elif num_channels == 1:
        axes = [[ax] for ax in axes]

    for plot_idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= batch_size:
            print(f"Warning: Sample index {sample_idx} exceeds batch size {batch_size}")
            continue

        # Determine valid length for this sample
        valid_length = time_steps
        if datalength is not None and sample_idx < len(datalength):
            valid_length = int(datalength[sample_idx])

        for channel_idx in range(num_channels):
            ax = axes[plot_idx][channel_idx]

            # Plot original and reconstructed trajectories
            time_range = range(valid_length)
            orig_data = original[sample_idx, :valid_length, channel_idx].detach().cpu().numpy()
            recon_data = reconstructed[sample_idx, :valid_length, channel_idx].detach().cpu().numpy()

            ax.plot(time_range, orig_data, label='Original', color='blue', linewidth=1.5)
            ax.plot(time_range, recon_data, label='Reconstructed',
                   color='red', linestyle='--', linewidth=1.5)

            ax.set_title(f'Channel {channel_idx + 1}, Sample {sample_idx + 1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_model_graph(
    model: torch.nn.Module,
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: str = "model_graph",
    filename: str = "model_viz",
    format: str = "pdf",
    show_attrs: bool = True,
    show_saved: bool = True
) -> str:
    """
    Visualize the computational graph of a PyTorch model.

    Args:
        model: PyTorch model to visualize
        input_data: Example input to the model
        output_path: Directory to save the visualization
        filename: Name of the output file (without extension)
        format: Output format (pdf, png, svg)
        show_attrs: Whether to display node attributes
        show_saved: Whether to highlight saved tensors

    Returns:
        Path to the saved visualization file

    Raises:
        ImportError: If torchviz is not installed
    """
    try:
        from torchviz import make_dot
    except ImportError:
        raise ImportError("torchviz is required for model visualization. Install with: pip install torchviz")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Forward pass with the provided input
    model.eval()
    with torch.no_grad():
        if isinstance(input_data, tuple):
            output = model(*input_data)
        else:
            output = model(input_data)

    # Create visualization based on output type
    if hasattr(output, "reconstruction_loss"):
        # For models with custom output objects (e.g., TSPVAE)
        target_tensor = output.reconstruction_loss
    elif hasattr(output, "loss"):
        # For models with loss attribute
        target_tensor = output.loss
    else:
        # For standard models that return tensors
        target_tensor = output

    # Generate computational graph
    dot = make_dot(target_tensor, params=dict(model.named_parameters()),
                   show_attrs=show_attrs, show_saved=show_saved)

    # Customize appearance
    dot.attr('graph', rankdir='TB')  # Top to bottom layout
    dot.attr('node', fontsize='12')
    dot.attr('edge', fontsize='10')

    # Save visualization
    output_file = os.path.join(output_path, filename)
    dot.render(output_file, format=format, cleanup=True)

    final_path = f"{output_file}.{format}"
    print(f"Model visualization saved to {final_path}")
    return final_path


def visualize_tspvae_forward(
    tspvae_model: torch.nn.Module,
    batch_data: Tuple[torch.Tensor, ...],
    device: torch.device,
    output_path: str = "visualizations"
) -> str:
    """
    Visualize TSPVAE model's forward pass with actual batch data.

    Args:
        tspvae_model: The TSPVAE model instance
        batch_data: Batch from the dataloader
        device: Device to run the model on
        output_path: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    _, Y, _, datalength = batch_data
    Y = Y.float().to(device)

    # Create irrelevant mask as done in training
    irrelevant_mask = (
        torch.arange(Y.size(2))
        .expand(Y.size(0), Y.size(1), -1)
        < datalength.unsqueeze(-1).unsqueeze(-1)
    )

    return visualize_model_graph(
        tspvae_model,
        (Y, irrelevant_mask.permute(0, 2, 1).to(device)),
        output_path=output_path,
        filename="tspvae_full_model"
    )


# Backward compatibility aliases
plot_traj_j = plot_trajectory_comparison
Display = display_tensor

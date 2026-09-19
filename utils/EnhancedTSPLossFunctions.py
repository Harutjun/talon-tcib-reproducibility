import torch
import torch.nn.functional as F

def enhanced_tspvae_loss(output, target, beta=1.0, alpha=1.0, mask=None):
    """
    Compute the total loss for Enhanced TSPVAE model.
    Now works with scalar losses that are already properly averaged over batch.

    Args:
        output (ModelOutput): Output from Enhanced TSPVAE forward pass
        target (torch.Tensor): Target tensor of shape (batch_size, n_channels, sequence_length)
        beta (float): Weight for KL divergence term (default: 1.0)
        alpha (float): Weight for reconstruction loss term (default: 1.0)
        mask (torch.Tensor, optional): Mask for valid elements (not used, handled in model)

    Returns:
        dict: Dictionary containing individual loss components and total loss
    """
    # Get losses from model output (already scalar values averaged over batch)
    reconstruction_loss = output.reconstruction_loss.mean()  # Scalar
    kl_loss = output.KL_Loss.mean()  # Scalar

    # Total loss (ELBO = reconstruction_loss + beta * kl_loss)
    total_loss = alpha * reconstruction_loss + beta * kl_loss

    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kl_loss': kl_loss,
        'reconstruction_loss_per_sample': None,  # Not available with new implementation
        'kl_loss_per_sample': None  # Not available with new implementation
    }


def compute_gp_prior_kl_divergence(mu_batch, banded_posterior, gp_prior, mask=None):
    """Compute per-sample KL(q||p) where
    q(z)=N(μ, Σ_q) with Σ_q = Q^{-1} ⊗ I_C (shared temporal precision across latent channels)
    p(z)=N(0, K_t ⊗ K_c)

    Inputs:
      mu_batch: Tensor [B,C,T] or [B,T] (adds channel dim automatically)
      banded_posterior: BandedPrecisionGaussian over T with precision Q
      gp_prior: KronTimeChannelKernel with K_t (T,T) and K_c (C,C)
      mask: Optional [B,T] boolean/float mask (1 valid) to zero-out invalid time steps in quadratic term

    KL(q||p) per sample:
      KL = 0.5[ log|K| + C log|Q| - d + tr(K_t^{-1} Q^{-1}) tr(K_c^{-1}) + tr(K_c^{-1} M K_t^{-1} M^T) ]
      where d = C*T and M is the (C,T) mean matrix for that sample.

    Returns:
      Tensor [B] of KL values (no batch reduction).
    """
    device = mu_batch.device
    if mu_batch.dim() == 2:  # [B,T] -> add channels dim = 1
        mu_batch = mu_batch.unsqueeze(1)
    B, C, T = mu_batch.shape

    # GP prior components
    K_t = gp_prior.K_t().to(device)            # [T,T]
    K_c = gp_prior.K_c().to(device)            # [C,C]
    jitter = float(gp_prior.jitter.item()) if isinstance(gp_prior.jitter, torch.Tensor) else float(gp_prior.jitter)
    K_t_reg = K_t + jitter * torch.eye(T, device=device, dtype=K_t.dtype)
    K_c_reg = K_c + jitter * torch.eye(C, device=device, dtype=K_c.dtype)

    # Posterior precision (shared)
    Q = banded_posterior._build_precision_matrix().to(device)  # [T,T]
    Q_reg = Q + jitter * torch.eye(T, device=device, dtype=Q.dtype)

    # Cholesky factorizations
    L_Kt = torch.linalg.cholesky(K_t_reg)
    L_Kc = torch.linalg.cholesky(K_c_reg)
    L_Q  = torch.linalg.cholesky(Q_reg)

    # Log determinants
    log_det_Kt = 2 * torch.sum(torch.log(torch.diag(L_Kt)))
    log_det_Kc = 2 * torch.sum(torch.log(torch.diag(L_Kc)))
    log_det_Q  = 2 * torch.sum(torch.log(torch.diag(L_Q)))
    log_det_K  = C * log_det_Kt + T * log_det_Kc  # log|K_t⊗K_c|

    # Inverses via solves
    I_T = torch.eye(T, device=device, dtype=K_t.dtype)
    I_C = torch.eye(C, device=device, dtype=K_c.dtype)
    K_t_inv = torch.cholesky_solve(I_T, L_Kt)
    K_c_inv = torch.cholesky_solve(I_C, L_Kc)
    Q_inv   = torch.cholesky_solve(I_T, L_Q)

    # Trace term: tr(K_t^{-1} Q^{-1}) * tr(K_c^{-1})
    trace_term = torch.sum(K_t_inv * Q_inv) * torch.trace(K_c_inv)

    # Mask handling (broadcast along channels)
    if mask is not None:
        mask_f = mask.float()  # [B,T]
        mu_eff = mu_batch * mask_f.unsqueeze(1)
        valid_counts = mask_f.sum(dim=1)  # [B]
    else:
        mu_eff = mu_batch
        valid_counts = torch.full((B,), T, device=device, dtype=mu_batch.dtype)

    # Quadratic term: tr(K_c_inv M K_t_inv M^T)
    temp = torch.matmul(mu_eff, K_t_inv)        # [B,C,T]
    quad = torch.einsum('bct,bdt,cd->b', temp, mu_eff, K_c_inv)  # [B]

    # Effective dimensionality d = C * (#valid time steps per sample)
    d_eff = C * valid_counts  # [B]

    # KL per sample (broadcast scalar terms)
    # log|K| + C log|Q| - d_eff + trace_term + quad
    base = log_det_K + C * log_det_Q + trace_term  # scalars
    kl = 0.5 * (base - d_eff + quad)
    return kl


def sample_from_gp_prior(gp_prior, batch_size=1, num_samples=1):
    """
    Sample from the GP prior distribution.
    
    Args:
        gp_prior (KronTimeChannelKernel): GP prior distribution
        batch_size (int): Number of batch samples
        num_samples (int): Number of samples per batch element
    
    Returns:
        torch.Tensor: Samples from GP prior
    """
    device = next(gp_prior.parameters()).device
    T = gp_prior.T
    
    if num_samples == 1:
        # Sample standard normal noise
        eps = torch.randn(batch_size, T, device=device)
        
        # Transform through GP covariance
        z_samples = []
        for i in range(batch_size):
            # Solve K @ alpha = eps to get alpha, then z = L @ alpha where K = L @ L^T
            # This is equivalent to sampling from N(0, K)
            K_t = gp_prior.K_t()
            L_t = torch.linalg.cholesky(K_t)
            z_i = L_t @ eps[i]
            z_samples.append(z_i)
        
        return torch.stack(z_samples, dim=0)
    else:
        # Multiple samples per batch element
        eps = torch.randn(batch_size, num_samples, T, device=device)
        
        z_samples = []
        K_t = gp_prior.K_t()
        L_t = torch.linalg.cholesky(K_t)
        for i in range(batch_size):
            z_i_samples = []
            for j in range(num_samples):
                z_ij = L_t @ eps[i, j]
                z_i_samples.append(z_ij)
            z_samples.append(torch.stack(z_i_samples, dim=0))
        
        return torch.stack(z_samples, dim=0)


def validate_enhanced_tspvae_output(output, input_shape, sequence_length, n_channels):
    """
    Validate the output of Enhanced TSPVAE model.
    
    Args:
        output (ModelOutput): Model output
        input_shape (tuple): Expected input shape (batch_size, n_channels, sequence_length)
        sequence_length (int): Expected sequence length
        n_channels (int): Expected number of channels
    
    Returns:
        bool: True if output is valid, False otherwise
    """
    batch_size, _, _ = input_shape
    
    try:
        # Check x_hat shape
        if output.x_hat.shape != (batch_size, n_channels, sequence_length):
            print(f"Invalid x_hat shape: {output.x_hat.shape}, expected: {(batch_size, n_channels, sequence_length)}")
            return False
        
        # Check z shape (should be flattened sequence length)
        expected_z_shape = (batch_size, sequence_length // 16)  # Assuming patch_length=16
        if output.z.shape != expected_z_shape:
            print(f"Invalid z shape: {output.z.shape}, expected: {expected_z_shape}")
            return False
        
        # Check mu shape
        if output.mu.shape != expected_z_shape:
            print(f"Invalid mu shape: {output.mu.shape}, expected: {expected_z_shape}")
            return False
        
        # Check loss shapes
        if output.KL_Loss.shape != (batch_size,):
            print(f"Invalid KL_Loss shape: {output.KL_Loss.shape}, expected: {(batch_size,)}")
            return False
        
        if output.reconstruction_loss.shape != (batch_size,):
            print(f"Invalid reconstruction_loss shape: {output.reconstruction_loss.shape}, expected: {(batch_size,)}")
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(output.KL_Loss).any() or torch.isinf(output.KL_Loss).any():
            print("KL_Loss contains NaN or infinite values")
            return False
        
        if torch.isnan(output.reconstruction_loss).any() or torch.isinf(output.reconstruction_loss).any():
            print("Reconstruction_loss contains NaN or infinite values")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during output validation: {e}")
        return False


def compute_enhanced_tspvae_metrics(output, target, mask=None):
    """
    Compute comprehensive metrics for Enhanced TSPVAE model.
    
    Args:
        output (ModelOutput): Model output
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for relevant elements
    
    Returns:
        dict: Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic reconstruction metrics
    mse = F.mse_loss(output.x_hat, target, reduction='none')
    if mask is not None:
        mse = mse * mask.permute(0, 2, 1)
        metrics['mse_per_sample'] = mse.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
    else:
        metrics['mse_per_sample'] = mse.mean(dim=[1, 2])
    
    metrics['mse_total'] = metrics['mse_per_sample'].mean()
    
    # MAE
    mae = torch.abs(output.x_hat - target)
    if mask is not None:
        mae = mae * mask.permute(0, 2, 1)
        metrics['mae_per_sample'] = mae.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
    else:
        metrics['mae_per_sample'] = mae.mean(dim=[1, 2])
    
    metrics['mae_total'] = metrics['mae_per_sample'].mean()
    
    # KL divergence metrics
    metrics['kl_per_sample'] = output.KL_Loss
    metrics['kl_total'] = output.KL_Loss.mean()
    metrics['kl_std'] = output.KL_Loss.std()
    
    # Latent space metrics
    metrics['latent_mean_norm'] = torch.norm(output.mu, dim=-1).mean()
    metrics['latent_std'] = output.mu.std(dim=-1).mean()
    
    # Signal-to-noise ratio (SNR)
    signal_power = torch.mean(target ** 2, dim=[1, 2])
    noise_power = torch.mean((output.x_hat - target) ** 2, dim=[1, 2])
    metrics['snr_db'] = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    metrics['snr_db_mean'] = metrics['snr_db'].mean()
    
    return metrics

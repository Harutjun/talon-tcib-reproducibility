
import torch

def kl_divergence_isotropic(mu_q, L_q, sigma_p=torch.tensor([1.0]), mask=None):
    """
    Compute the KL divergence between a multivariate Gaussian q and an isotropic Gaussian p.

    KL(q || p) for q = N(mu_q, Sigma_q) and p = N(0, I)

    Parameters:
    - mu_q: Tensor, mean of q, shape (batch_size, d)
    - L_q: Tensor, lower triangular Cholesky factor of Sigma_q, shape (batch_size, d, d)

    Returns:
    - kl: Tensor, KL divergence for each sample in the batch, shape (batch_size,)
    """

    if mask is None:
        mask = torch.ones_like(mu_q).to(mu_q.device)

    batch_size, d = mu_q.shape
    sigma_p = sigma_p.to(mu_q.device)
    diagonal_mask = torch.diag_embed(mask).float()
    L_q = torch.bmm(L_q, diagonal_mask)
    # Compute Sigma_q from its Cholesky factor
    Sigma_q = L_q @ L_q.transpose(-2, -1)

    # Compute the log-determinant using the Cholesky diagonals
    log_det_Sigma_q = 2 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)  + 1e-10 ) *  torch.diagonal(diagonal_mask, dim1=-2, dim2=-1), dim=1)

    # Compute the trace term: Tr(I^-1 Sigma_q) = Tr(Sigma_q) = sum of diagonal elements of Sigma_q
    trace_term = torch.sum((torch.diagonal(Sigma_q, dim1=-2, dim2=-1) / (sigma_p ** 2)), dim=1)

    # Compute the quadratic term: (mu_q)^T I^-1 (mu_q) = (mu_q)^T (mu_q) = sum of squares of mu_q
    quadratic_term = torch.sum(mu_q ** 2 / (sigma_p ** 2), dim=1)
    log_det_Sigma_p = (sigma_p * mask).sum(dim=1)
    # Combine terms to calculate KL divergence
    kl = 0.5 * (log_det_Sigma_p - log_det_Sigma_q - mask.sum(dim=1) + trace_term + quadratic_term)
    kl = torch.clip(kl, -5e3, 1e9)
    if torch.any(torch.isnan(kl)):
        print('Kl is nan for some samples')

    return kl


def kl_divergence_gaussians(mu_q, L_q, mu_p, L_p):
    """
    Compute the KL divergence between two multivariate Gaussians.

    KL(q || p) for q = N(mu_q, Sigma_q) and p = N(mu_p, Sigma_p)

    Parameters:
    - mu_q: Tensor, mean of q, shape (batch_size, d)
    - L_q: Tensor, lower triangular Cholesky factor of Sigma_q, shape (batch_size, d, d)
    - mu_p: Tensor, mean of p, shape (batch_size, d)
    - L_p: Tensor, lower triangular Cholesky factor of Sigma_p, shape (batch_size, d, d)

    Returns:
    - kl: Tensor, KL divergence for each sample in the batch, shape (batch_size,)
    """
    batch_size, d = mu_q.shape

    # Compute Sigma_q and Sigma_p from their Cholesky factors
    Sigma_q = L_q @ L_q.transpose(-2, -1)
    Sigma_p = L_p @ L_p.transpose(-2, -1)

    # Compute the log-determinants using the Cholesky diagonals
    log_det_Sigma_q = 2 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=1)
    log_det_Sigma_p = 2 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)

    # Compute the trace term: Tr(Sigma_p^-1 Sigma_q)
    Sigma_p_inv = torch.linalg.inv(Sigma_p).expand(batch_size, d, d)
    trace_term = torch.einsum("bij,bjk->bk", Sigma_p_inv, Sigma_q).sum(dim=1)

    # Compute the quadratic term: (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q)
    delta_mu = (mu_p - mu_q).unsqueeze(-1)  # Shape (batch_size, d, 1)
    quadratic_term = torch.einsum("bij,bjk,bki->b", delta_mu.transpose(-2, -1), Sigma_p_inv, delta_mu)

    # Combine terms to calculate KL divergence
    kl = 0.5 * (log_det_Sigma_p - log_det_Sigma_q - d + trace_term + quadratic_term)

    return kl


def kl_divergence_cholesky(mu_q, Lq, Lp):
    """
    Compute KL divergence KL(q(z)||p(z)) where
      q(z) = N(mu_q, Sigma_q) with Sigma_q = Lq Lq^T  (Lq provided),
      p(z) = N(0, K) with K = L L^T,
    and L is the lower-triangular Cholesky factor of K.

    Args:
      mu_q: Tensor of shape (T,) or (B, T) -- posterior mean.
      Lq: Tensor of shape (T, T) or (B, T, T) -- lower-triangular Cholesky factor of Sigma_q.
      L: Tensor of shape (T, T) or (B, T, T) -- lower-triangular Cholesky factor of K.

    Returns:
      KL divergence (tensor of shape (B,)) for each batch element.
    """
    # Ensure batched inputs
    if mu_q.dim() == 1:
        mu_q = mu_q.unsqueeze(0)  # shape (1, T)
    if Lq.dim() == 2:
        Lq = Lq.unsqueeze(0)  # shape (1, T, T)
    if Lp.dim() == 2:
        Lp = Lp.unsqueeze(0)  # shape (1, T, T)

    B, T = mu_q.shape

    # 1. Log-determinant of K: log|K| = 2*sum(log(diag(L)))
    diag_L = torch.diagonal(Lp, dim1=-2, dim2=-1)  # shape (B, T)
    logdet_K = 2 * torch.sum(torch.log(diag_L), dim=1)  # shape (B,)

    # 2. Log-determinant of Sigma_q: log|Sigma_q| = 2*sum(log(diag(Lq)))
    diag_Lq = torch.diagonal(Lq, dim1=-2, dim2=-1)  # shape (B, T)
    logdet_Sigma_q = 2 * torch.sum(torch.log(diag_Lq), dim=1)  # shape (B,)

    # 3. Quadratic term: Solve L * alpha = mu_q for alpha, then compute ||alpha||^2.
    mu_q_unsq = mu_q.unsqueeze(-1)  # shape (B, T, 1)
    alpha, _ = torch.triangular_solve(mu_q_unsq, L, upper=False)
    quad_term = torch.sum(alpha ** 2, dim=[1, 2])  # shape (B,)

    # 4. Trace term: Compute Y = L^{-1} * Lq, then trace(K^{-1} Sigma_q) = sum(Y^2)
    Y, _ = torch.triangular_solve(Lq, L, upper=False)
    trace_term = torch.sum(Y ** 2, dim=[1, 2])  # shape (B,)

    # 5. Combine into KL divergence:
    kl = 0.5 * (logdet_K - logdet_Sigma_q - T + trace_term + quad_term)
    return kl



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_mi_loss_weight(mi_loss_weight, beta: float):
    """Resolve an MI weight config value.

    Supports numeric values directly and the string ``"auto"``, which maps to
    the optimal ``beta / (1 + beta)`` setting.
    """
    if isinstance(mi_loss_weight, str) and mi_loss_weight.strip().lower() == 'auto':
        beta_val = float(beta)
        if beta_val < 0.0:
            raise ValueError(f"beta must be non-negative for auto MI weight resolution, got {beta_val}")
        return beta_val / (1.0 + beta_val)
    return float(mi_loss_weight)

def extract_diagonal_variances(Lt, Lc):
    r"""
    Extracts the diagonal variances of the precision matrix factorized as (Lt @ Lt.T) \otimes (Lc @ Lc.T).
    Actually, if the precision matrix is P = P_t \otimes P_c, then the covariance is \Sigma = P_t^{-1} \otimes P_c^{-1}.
    The variance of the (t, c) element is \Sigma_{(t,c), (t,c)} = (P_t^{-1})_{t,t} * (P_c^{-1})_{c,c}.
    
    Since we have the Cholesky factors Lt and Lc for the precisions:
    P_t = Lt @ Lt^T
    P_c = Lc @ Lc^T
    
    We want the diagonal of P_t^{-1} and P_c^{-1}.
    P_t^{-1} = (Lt @ Lt^T)^{-1} = (Lt^T)^{-1} @ Lt^{-1}.
    So (P_t^{-1})_{i,i} is the squared Euclidean norm of the i-th row of Lt^{-1}.
    
    Args:
        Lt: [T, T] or [B, T, T] lower triangular Cholesky factor of the time precision
        Lc: [C, C] or [B, C, C] lower triangular Cholesky factor of the channel precision
    
    Returns:
        var: [B, T * C] or [T * C] containing the diagonal variances.
    """
    # Make sure we have 3D tensors for batching
    has_batch_t = Lt.dim() == 3
    has_batch_c = Lc.dim() == 3
    
    # Compute diagonal of P_t^{-1}
    I_t = torch.eye(Lt.size(-1), device=Lt.device, dtype=Lt.dtype)
    if has_batch_t:
        I_t = I_t.unsqueeze(0).expand(Lt.size(0), -1, -1)
    # L_t^{-1}
    inv_Lt = torch.linalg.solve_triangular(Lt, I_t, upper=False)
    # P_t^{-1} = (Lt @ Lt^T)^{-1} = inv_Lt^T @ inv_Lt
    # Diagonal elements are sum of squares of columns of inv_Lt (or rows of inv_Lt^T)
    var_t = torch.sum(inv_Lt ** 2, dim=-2) # [B, T] or [T]
    
    if Lc is None:
        return var_t
        
    has_batch_c = Lc.dim() == 3
    # Compute diagonal of P_c^{-1}
    I_c = torch.eye(Lc.size(-1), device=Lc.device, dtype=Lc.dtype)
    if has_batch_c:
        I_c = I_c.unsqueeze(0).expand(Lc.size(0), -1, -1)
    inv_Lc = torch.linalg.solve_triangular(Lc, I_c, upper=False)
    var_c = torch.sum(inv_Lc ** 2, dim=-2) # [B, C] or [C]
    
    # \Sigma_{(t,c), (t,c)} = var_t[t] * var_c[c]
    # Outer product to get [B, T, C] or [T, C]
    if has_batch_t and has_batch_c:
        var = var_t.unsqueeze(-1) * var_c.unsqueeze(-2) # [B, T, C]
        var = var.view(var.size(0), -1) # [B, T*C]
    elif has_batch_t and not has_batch_c:
        var = var_t.unsqueeze(-1) * var_c.unsqueeze(0).unsqueeze(0)
        var = var.view(var.size(0), -1)
    elif not has_batch_t and has_batch_c:
        var = var_t.unsqueeze(0).unsqueeze(-1) * var_c.unsqueeze(-2)
        var = var.view(var.size(0), -1)
    else:
        var = var_t.unsqueeze(-1) * var_c.unsqueeze(0)
        var = var.view(-1)
        
    return var

def compute_pairwise_kl(mu_phi, var_phi, mu_psi, var_psi):
    """
    Computes the B x B matrix of pairwise KL divergences D_{i,j} = KL(q_phi^(i) || q_psi^(j))
    under the diagonal Gaussian approximation.
    
    Args:
        mu_phi: [B, K] where K = T * C
        var_phi: [B, K]
        mu_psi: [B, K]
        var_psi: [B, K]
        
    Returns:
        D: [B, B] matrix
    """
    B, K = mu_phi.shape
    var_phi = torch.clamp(var_phi, min=1e-8)
    var_psi = torch.clamp(var_psi, min=1e-8)
    
    # S_psi[j] = sum_k log(var_psi^(j)[k])
    S_psi = torch.sum(torch.log(var_psi), dim=1) # [B]
    
    # S_phi[i] = sum_k log(var_phi^(i)[k])
    S_phi = torch.sum(torch.log(var_phi), dim=1) # [B]
    
    # M_1[i, j] = sum_k (var_phi^(i)[k] + mu_phi^(i)[k]^2) / var_psi^(j)[k]
    A_phi = var_phi + mu_phi ** 2 # [B, K]
    inv_var_psi = 1.0 / var_psi # [B, K]
    M_1 = torch.matmul(A_phi, inv_var_psi.T) # [B, B]
    
    # M_2[i, j] = sum_k mu_phi^(i)[k] * mu_psi^(j)[k] / var_psi^(j)[k]
    mu_psi_over_var = mu_psi * inv_var_psi # [B, K]
    M_2 = torch.matmul(mu_phi, mu_psi_over_var.T) # [B, B]
    
    # V_3[j] = sum_k (mu_psi^(j)[k])^2 / var_psi^(j)[k]
    V_3 = torch.sum((mu_psi ** 2) * inv_var_psi, dim=1) # [B]
    
    # D[i, j] = 0.5 * (S_psi[j] - S_phi[i] + M_1[i, j] - 2*M_2[i, j] + V_3[j] - K)
    D = 0.5 * (S_psi.unsqueeze(0) - S_phi.unsqueeze(1) + M_1 - 2 * M_2 + V_3.unsqueeze(0) - K)
    
    # Add clamp to prevent negative KL due to numerical precision
    return torch.clamp(D, min=0.0)

def compute_pairwise_kl_full(mu_phi, L_phi, mu_psi, L_psi):
    """
    Computes the B x B matrix of pairwise KL divergences D_{i,j} = KL(q_phi^(i) || q_psi^(j))
    using the full covariance matrices.
    
    Args:
        mu_phi: [B, d]
        L_phi: [B, d, d] lower triangular Cholesky factor
        mu_psi: [B, d]
        L_psi: [B, d, d] lower triangular Cholesky factor
        
    Returns:
        D: [B, B] matrix
    """
    B, d = mu_phi.shape
    L_phi = torch.clamp(L_phi, min=-1e5, max=1e5)
    L_psi = torch.clamp(L_psi, min=-1e5, max=1e5)
    
    # Covariances
    Sigma_phi = L_phi @ L_phi.transpose(-2, -1)
    Sigma_psi = L_psi @ L_psi.transpose(-2, -1)
    
    # Inverses
    # L_psi^{-1}
    I_d = torch.eye(d, device=L_psi.device, dtype=L_psi.dtype).unsqueeze(0).expand(B, -1, -1)
    inv_L_psi = torch.linalg.solve_triangular(L_psi, I_d, upper=False)
    # Sigma_psi^{-1} = inv_L_psi^T @ inv_L_psi
    inv_Sigma_psi = inv_L_psi.transpose(-2, -1) @ inv_L_psi
    
    # Log determinants
    log_det_phi = 2 * torch.sum(torch.log(torch.diagonal(L_phi, dim1=-2, dim2=-1) + 1e-12), dim=-1) # [B]
    log_det_psi = 2 * torch.sum(torch.log(torch.diagonal(L_psi, dim1=-2, dim2=-1) + 1e-12), dim=-1) # [B]
    
    # Trace term: Tr((Sigma_psi^(j))^{-1} Sigma_phi^(i)) -> [B, B]
    # inv_Sigma_psi is [B, d, d], Sigma_phi is [B, d, d]
    # We want [i, j] = sum_{c,k} inv_Sigma_psi[j, c, k] * Sigma_phi[i, k, c]
    trace_term = torch.einsum('jck, ikc -> ij', inv_Sigma_psi, Sigma_phi)
    
    # Quadratic term
    # term1: (mu_psi^(j))^T (Sigma_psi^(j))^{-1} (mu_psi^(j)) -> [B]
    quad_1 = torch.einsum('jc, jcd, jd -> j', mu_psi, inv_Sigma_psi, mu_psi)
    
    # term2: -2 * (mu_phi^(i))^T (Sigma_psi^(j))^{-1} (mu_psi^(j)) -> [B, B]
    quad_2 = -2.0 * torch.einsum('ic, jcd, jd -> ij', mu_phi, inv_Sigma_psi, mu_psi)
    
    # term3: (mu_phi^(i))^T (Sigma_psi^(j))^{-1} (mu_phi^(i)) -> [B, B]
    quad_3 = torch.einsum('ic, jcd, id -> ij', mu_phi, inv_Sigma_psi, mu_phi)
    
    quadratic_term = quad_1.unsqueeze(0) + quad_2 + quad_3
    
    # Combine
    D = 0.5 * (log_det_psi.unsqueeze(0) - log_det_phi.unsqueeze(1) - d + trace_term + quadratic_term)
    return torch.clamp(D, min=0.0)

class MIEstimator(nn.Module):
    """
    Mutual Information Estimator.
    Supports:
        - 'barber_agakov' (default): Parameter-free closed-form Barber-Agakov variational lower bound
        - 'infonce': Legacy InfoNCE with pairwise negative KL critic
    """
    def __init__(self, learn_temperature=True, initial_temperature=1.0, estimator_type=None):
        super().__init__()
        import os
        if estimator_type is None:
            estimator_type = os.environ.get('MI_ESTIMATOR_TYPE', 'barber_agakov').strip().lower()
        self.estimator_type = estimator_type
        self.learn_temperature = learn_temperature
        
        # Always register log_tau for full checkpoint compatibility with legacy InfoNCE checkpoints
        if learn_temperature and self.estimator_type == 'infonce':
            self.log_tau = nn.Parameter(torch.tensor([float(initial_temperature)]).log())
        else:
            self.register_buffer('log_tau', torch.tensor([float(initial_temperature)]).log())
            
        if self.estimator_type == 'barber_agakov':
            self.ba_estimator = BarberAgakovMIEstimator(learn_sigma=learn_temperature, initial_sigma_sq=initial_temperature)
        else:
            self.ba_estimator = None

    def get_tau(self):
        return torch.exp(self.log_tau).clamp(min=0.01, max=100.0)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Auto-detect estimator type from checkpoint keys to ensure seamless strict loading
        ba_key = prefix + 'ba_estimator.log_sigma_sq'
        tau_key = prefix + 'log_tau'
        if ba_key not in state_dict and tau_key in state_dict:
            # Checkpoint was trained with InfoNCE
            self.estimator_type = 'infonce'
            self.ba_estimator = None
            if not isinstance(self.log_tau, nn.Parameter) and self.learn_temperature:
                self.log_tau = nn.Parameter(state_dict[tau_key].clone())
        elif ba_key in state_dict:
            # Checkpoint was trained with Barber-Agakov
            self.estimator_type = 'barber_agakov'
            if self.ba_estimator is None:
                self.ba_estimator = BarberAgakovMIEstimator(learn_sigma=self.learn_temperature)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, mu_phi, var_or_L_phi, mu_psi, var_or_L_psi, is_full_cov=False):
        if self.estimator_type == 'barber_agakov':
            return self.ba_estimator(mu_phi, var_or_L_phi, mu_psi, var_or_L_psi, is_full_cov=is_full_cov)
            
        # Legacy InfoNCE branch
        if not is_full_cov and mu_phi.dim() > 2:
            B = mu_phi.shape[0]
            mu_phi = mu_phi.view(B, -1)
            var_or_L_phi = var_or_L_phi.view(B, -1)
            mu_psi = mu_psi.view(B, -1)
            var_or_L_psi = var_or_L_psi.view(B, -1)
            
        B = mu_phi.shape[0]
        if B <= 1:
            device = mu_phi.device
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
            
        if is_full_cov:
            D = compute_pairwise_kl_full(mu_phi, var_or_L_phi, mu_psi, var_or_L_psi)
        else:
            D = compute_pairwise_kl(mu_phi, var_or_L_phi, mu_psi, var_or_L_psi)
        
        tau = self.get_tau()
        scores = -D / tau
        
        labels = torch.arange(B, device=D.device)
        loss_phi_to_psi = F.cross_entropy(scores, labels)
        loss_psi_to_phi = F.cross_entropy(scores.T, labels)
        nce_loss = 0.5 * (loss_phi_to_psi + loss_psi_to_phi)
        mi_estimate = torch.log(torch.tensor(B, dtype=torch.float32, device=D.device)) - nce_loss
        return nce_loss, mi_estimate.detach()



class BarberAgakovMIEstimator(nn.Module):
    """
    Parameter-free Barber-Agakov variational mutual information lower bound.
    
    Under the isotropic Gaussian proposal:
        r(z_phi | z_psi) = N(z_phi; z_psi, sigma^2 I)
    which embodies the core student-teacher alignment assumption.
    
    Given (X, Y), the two latents are conditionally independent, giving:
        E[||z_phi - z_psi||^2] = ||mu_phi - mu_psi||^2 + tr(Sigma_psi) + tr(Sigma_phi)
    exactly in closed form without Monte Carlo sampling or Jensen's gap.
    
    The lower bound on I(Z_psi; Z_phi) is:
        I(Z_psi; Z_phi) >= H(Z_phi) - (d/2)*log(2*pi*sigma^2) - (1/(2*sigma^2)) * E[||mu_phi - mu_psi||^2 + tr(Sigma_psi) + tr(Sigma_phi)]
    
    Since the teacher is frozen, H(Z_phi) is constant w.r.t. student parameters psi.
    Maximizing the lower bound corresponds to minimizing:
        loss_BA = (1 / (2 * sigma^2)) * mean(V) + (d / 2) * log(2 * pi * sigma^2)
    where V = ||mu_phi - mu_psi||^2 + tr(Sigma_psi) + tr(Sigma_phi).
    
    sigma^2 can either be learned via log_sigma_sq (default) or tightened to its
    closed-form optimum sigma^{*2} = mean(V) / d.
    """
    def __init__(self, learn_sigma=True, initial_sigma_sq=1.0, tighten_analytic=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.tighten_analytic = tighten_analytic
        if learn_sigma and not tighten_analytic:
            self.log_sigma_sq = nn.Parameter(torch.tensor([float(initial_sigma_sq)]).log())
        else:
            self.register_buffer('log_sigma_sq', torch.tensor([float(initial_sigma_sq)]).log())

    def get_sigma_sq(self, mean_V=None, d=None):
        if self.tighten_analytic and mean_V is not None and d is not None:
            return (mean_V.detach() / float(d)).clamp(min=1e-6, max=1e6)
        if self.learn_sigma and not self.tighten_analytic:
            return torch.exp(self.log_sigma_sq).clamp(min=1e-4, max=1e4)
        return torch.exp(self.log_sigma_sq)

    def forward(self, mu_phi, var_or_L_phi, mu_psi, var_or_L_psi, is_full_cov=False):
        """
        Computes the Barber-Agakov bound loss and MI estimate.
        
        Args:
            mu_phi: [B, d] or [B, T, C] teacher/student mean
            var_or_L_phi: [B, d] diagonal variances OR [B, d, d] Cholesky factors
            mu_psi: [B, d] or [B, T, C] student/teacher mean
            var_or_L_psi: [B, d] diagonal variances OR [B, d, d] Cholesky factors
            is_full_cov: bool, whether inputs are full covariance Cholesky factors
            
        Returns:
            loss_ba: scalar loss to be minimized (proportional to -I_BA)
            mi_estimate: scalar MI estimate (lower bound surrogate)
        """
        if not is_full_cov and mu_phi.dim() > 2:
            B = mu_phi.shape[0]
            mu_phi = mu_phi.view(B, -1)
            var_or_L_phi = var_or_L_phi.view(B, -1)
            mu_psi = mu_psi.view(B, -1)
            var_or_L_psi = var_or_L_psi.view(B, -1)
            
        B = mu_phi.shape[0]
        d = mu_phi.shape[-1]
        
        # Mean difference norm squared: ||mu_phi - mu_psi||^2
        delta_mu_sq = torch.sum((mu_phi - mu_psi) ** 2, dim=-1)  # [B]
        
        # Traces of covariances
        if is_full_cov and var_or_L_phi.dim() == 3:
            # For Cholesky factor L, Sigma = L @ L.T, so tr(Sigma) = sum(L_{ij}^2)
            tr_phi = torch.sum(var_or_L_phi ** 2, dim=(-2, -1))  # [B]
            tr_psi = torch.sum(var_or_L_psi ** 2, dim=(-2, -1))  # [B]
        else:
            tr_phi = torch.sum(var_or_L_phi, dim=-1)  # [B]
            tr_psi = torch.sum(var_or_L_psi, dim=-1)  # [B]
            
        V = delta_mu_sq + tr_phi + tr_psi  # [B]
        mean_V = torch.mean(V)  # scalar
        
        sigma_sq = self.get_sigma_sq(mean_V, d)
        
        # Loss to minimize
        loss_ba = (1.0 / (2.0 * sigma_sq)) * mean_V + (d / 2.0) * torch.log(2.0 * math.pi * sigma_sq)
        
        # MI estimate: lower bound surrogate (without the constant H(Z_phi))
        mi_estimate = -loss_ba.detach()
        
        return loss_ba, mi_estimate


if __name__ == '__main__':
    # Unit Test for compute_pairwise_kl
    B, K = 4, 10
    mu_phi = torch.randn(B, K)
    var_phi = torch.rand(B, K) + 0.1
    mu_psi = torch.randn(B, K)
    var_psi = torch.rand(B, K) + 0.1
    
    D_fast = compute_pairwise_kl(mu_phi, var_phi, mu_psi, var_psi)
    
    D_slow = torch.zeros(B, B)
    for i in range(B):
        for j in range(B):
            D_slow[i, j] = 0.5 * torch.sum(
                torch.log(var_psi[j]) - torch.log(var_phi[i]) + 
                (var_phi[i] + (mu_phi[i] - mu_psi[j])**2) / var_psi[j] - 1
            )
            
    print("Max diff:", torch.max(torch.abs(D_fast - D_slow)).item())
    assert torch.allclose(D_fast, D_slow, atol=1e-5)
    
    # Test MIEstimator
    estimator = MIEstimator()
    loss, mi = estimator(mu_phi, var_phi, mu_psi, var_psi)
    print("NCE Loss:", loss.item())
    print("MI Estimate:", mi.item())
    print("All tests passed!")

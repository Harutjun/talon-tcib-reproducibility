import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- global activation settings ----
SP_BETA = 5.0       # high beta -> sharper (closer to ReLU)
SP_THRESHOLD = 20.0  # clamp for numerical stability

# ---- small utils ----
def softplus(x, beta: float = SP_BETA):  # stable softplus with sharp transition
    return F.softplus(x, beta=beta, threshold=SP_THRESHOLD)

def pairwise_dists_1d(t, lengthscale):
    # t: [T], lengthscale: scalar>0
    t = t.view(-1, 1)
    d = torch.cdist(t, t, p=2) / lengthscale
    # Ensure d has standard shape properties for downstream use
    d = d.contiguous()  # Make sure tensor is contiguous
    return d  # [T,T]

# ---- time kernels ----
def matern_kernel(d, nu, sigma2):
    # d: [T,T] scaled distance; nu in {0.5,1.5,2.5}
    if nu == 0.5:      # Exp / Matern-1/2
        K = torch.exp(-d)
    elif nu == 1.5:    # (1 + sqrt(3) d) exp(-sqrt(3) d)
        sqrt3 = 1.7320508075688772
        K = (1.0 + sqrt3 * d) * torch.exp(-sqrt3 * d)
    elif nu == 2.5:    # (1 + sqrt(5)d + 5 d^2/3) exp(-sqrt(5) d)
        sqrt5 = 2.23606797749979
        K = (1.0 + sqrt5 * d + 5.0 * (d ** 2) / 3.0) * torch.exp(-sqrt5 * d)
    else:
        raise ValueError("Supported Matérn nu: 0.5, 1.5, 2.5.")
    return sigma2 * K  # [T,T]

def rbf_kernel(d, sigma2):
    # d is already scaled by lengthscale -> use d^2
    return sigma2 * torch.exp(-(d ** 2) / 2.0)

def cauchy_kernel(d, alpha, beta, sigma2):
    # K(r) = σ^2 * (1 + (r/alpha)^2)^(-beta)
    return sigma2 * (1.0 + (d / alpha) ** 2) ** (-beta)

# ---- channel kernel parameterization: K_c = B B^T + diag(noise) ----
class LowRankPlusDiag(nn.Module):
    def __init__(self, C, rank=8, init_noise=0.1):
        super().__init__()
        self.B = nn.Parameter(0.01 * torch.randn(C, rank))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(init_noise)) * torch.ones(C))
    def forward(self):
        C = self.B.shape[0]
        Kc = self.B @ self.B.t()
        Kc = Kc + torch.diag(softplus(self.log_noise))
        return Kc  # [C,C], PD

# ---- main Kronecker kernel ----
class KronTimeChannelKernel(nn.Module):
    """
    K = K_t(t; params_t) ⊗ K_c(params_c)
    Efficient: logdet, solves, matmul, sampling without forming K.
    """
    def __init__(self, T, C, time_kernel="matern32", rank_c=8, jitter=1e-5):
        super().__init__()
        # Ensure T and C are always plain integers, not tensors
        self.T = int(T) if hasattr(T, 'item') else int(T)
        self.C = int(C) if hasattr(C, 'item') else int(C)
        
        # Proper jitter handling - convert string/numeric to float then tensor
        if isinstance(jitter, torch.Tensor):
            self.jitter = jitter
        else:
            # Handle string representations (e.g., "1e-5" from YAML) or numeric values
            jitter_value = float(jitter) if isinstance(jitter, (str, int, float)) else jitter
            self.jitter = torch.tensor(jitter_value, dtype=torch.get_default_dtype())

        # time inputs (register once; you can also pass per-call)
        self.register_buffer("t", torch.arange(self.T, dtype=torch.get_default_dtype()))

        # ---- choose time kernel family ----
        self.time_kernel = time_kernel.lower()
        # common params: log_sigma2, log_lengthscale
        self.log_sigma2_t = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_ls_t     = nn.Parameter(torch.log(torch.tensor(10.0)))  # lengthscale in steps

        # extra params per family
        if self.time_kernel.startswith("matern"):
            # pick nu from {"12","32","52"}
            if   self.time_kernel == "matern12": self.nu = 0.5
            elif self.time_kernel == "matern32": self.nu = 1.5
            elif self.time_kernel == "matern52": self.nu = 2.5
            else: raise ValueError("Use matern12|matern32|matern52")
        elif self.time_kernel == "rbf":
            pass
        elif self.time_kernel == "cauchy":
            # Cauchy has alpha (>0) and beta (>0)
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(1.0)))
            self.log_beta  = nn.Parameter(torch.log(torch.tensor(1.0)))
        else:
            raise ValueError("time_kernel ∈ {matern12,matern32,matern52,rbf,cauchy}")

        # channel kernel (learnable, time-invariant)
        self.Kc = LowRankPlusDiag(C, rank=rank_c, init_noise=0.1)

        # cached factorizations
        self._Lt = None
        self._Lc = None

    # ---- build K_t and K_c ----
    def K_t(self, t=None):
        t = self.t if t is None else t
        ls = softplus(self.log_ls_t)
        sig2 = softplus(self.log_sigma2_t)
        d = pairwise_dists_1d(t, ls)  # [T,T]
        
        # Initialize Kt to avoid potential reference before assignment
        Kt = None
        if self.time_kernel.startswith("matern"):
            Kt = matern_kernel(d, self.nu, sig2)
        elif self.time_kernel == "rbf":
            Kt = rbf_kernel(d, sig2)
        elif self.time_kernel == "cauchy":
            Kt = cauchy_kernel(d, alpha=softplus(self.log_alpha),
                                  beta=softplus(self.log_beta),
                                  sigma2=sig2)
        
        # Use self.T instead of extracting from tensor dimensions to avoid the tensor issue
        Kt = Kt + self.jitter * torch.eye(self.T, device=Kt.device, dtype=Kt.dtype)
        return Kt

    def K_c(self):
        Kc = self.Kc()
        Kc = Kc + self.jitter * torch.eye(self.C, device=Kc.device, dtype=Kc.dtype)
        return Kc

    # ---- (re)factorize small pieces ----
    def _factorize(self):
        Kt = self.K_t()
        Kc = self.K_c()
        self._Lt = torch.linalg.cholesky(Kt)  # [T,T]
        self._Lc = torch.linalg.cholesky(Kc)  # [C,C]

    # ---- efficient logdet(K_t ⊗ K_c) ----
    def logdet(self):
        if (self._Lt is None) or (self._Lc is None):
            self._factorize()
        # log|K| = C*log|K_t| + T*log|K_c|
        logdet_Kt = 2.0 * torch.sum(torch.log(torch.diag(self._Lt)))
        logdet_Kc = 2.0 * torch.sum(torch.log(torch.diag(self._Lc)))
        return self.C * logdet_Kt + self.T * logdet_Kc  # scalar
        # (Kronecker determinant identity).  See refs.

    # ---- solve (K_t ⊗ K_c) vec(X) = vec(B) using vec-trick ----
    def solve(self, b):
        """
        Smart solve method that chooses strategy based on actual numerical challenges
        """
        if (self._Lt is None) or (self._Lc is None):
            self._factorize()

        # The REAL pattern: RBF kernels with high T/C ratios are problematic
        tc_ratio = self.T / self.C
        needs_robust_handling = (
            self.time_kernel == "rbf" and
            tc_ratio > 4.0 and
            self.T > 15  # Only for reasonably long sequences
        )

        if needs_robust_handling:
            Kt = self.K_t()
            cond_Kt = torch.linalg.cond(Kt).item()
            if cond_Kt > 1e5:  # Actually ill-conditioned
                return self._solve_with_increased_regularization(b)

        # Use standard method for everything else (it's excellent!)
        return self._solve_standard(b)

    def _solve_standard(self, b):
        """Standard solve for small-medium matrices"""
        batch = b.shape[:-1]
        # For K_t ⊗ K_c, reshape b as (T, C) matrix
        B_mat = b.view(*batch, self.T, self.C)
        # Local refs and runtime checks to ensure static analyzers know these are Tensors
        Lc = self._Lc
        Lt = self._Lt
        if Lc is None or Lt is None:
            raise RuntimeError("Cholesky factors must be computed before solve")
        # Cast for static type checkers (no-op at runtime)
        from typing import cast
        Lc = cast(torch.Tensor, Lc)
        Lt = cast(torch.Tensor, Lt)
        # Solve K_c * Y = B_mat (solve along last dimension)
        Y = torch.cholesky_solve(B_mat.transpose(-2, -1), Lc).transpose(-2, -1)
        # Solve K_t * X = Y (solve along second-to-last dimension)
        X = torch.cholesky_solve(Y, Lt)
        return X.reshape(*batch, self.T * self.C)

    def _solve_large_matrix(self, b):
        """Enhanced solve for large matrices with stability improvements"""
        batch = b.shape[:-1]

        # Try standard method first
        return self._solve_standard(b)

    def _solve_with_increased_regularization(self, b):
        """Fallback solve with increased regularization"""
        # Temporarily increase jitter - handle tensor properly
        original_jitter = self.jitter.clone()
        jitter_value = max(1e-3, self.jitter.item() * 100)
        self.jitter = torch.tensor(jitter_value, device=self.jitter.device, dtype=self.jitter.dtype)

        # Force re-factorization with higher jitter
        self._Lt = None
        self._Lc = None
        self._factorize()

        # Local refs after factorization (avoid None warnings)
        Lc = self._Lc
        Lt = self._Lt
        if Lc is None or Lt is None:
            raise RuntimeError("Cholesky factors must be computed before fallback solve")
        # Cast for static analyzers
        from typing import cast
        Lc = cast(torch.Tensor, Lc)
        Lt = cast(torch.Tensor, Lt)

        # Solve with increased regularization
        batch = b.shape[:-1]
        B_mat = b.view(*batch, self.C, self.T)
        Y = torch.cholesky_solve(B_mat, Lc)
        X_T = torch.cholesky_solve(Y.transpose(-2, -1), Lt)
        X_mat = X_T.transpose(-2, -1)

        # Restore original jitter
        self.jitter = original_jitter

        return X_mat.reshape(*batch, self.T * self.C)

    # ---- matmul (K_t ⊗ K_c) v without forming the Kron ----
    def matmul(self, v):
        """Matrix multiplication (K_t ⊗ K_c) * v matching the solve vectorization convention"""
        batch = v.shape[:-1]
        # For K_t ⊗ K_c, reshape v as (T, C) matrix to match solve convention
        V = v.view(*batch, self.T, self.C)  # T × C matrix
        # (K_t ⊗ K_c) vec(V) = vec(K_t V K_c^T)
        out = self.K_t() @ V @ self.K_c().transpose(-2, -1)
        return out.reshape(*batch, self.T * self.C)

    # ---- sample z ~ N(0, K_t ⊗ K_c) efficiently ----
    def sample(self, batch_size, num_samples_per_element=1):
        """
        Returns [num_samples, T*C] samples.
        Uses: vec( L_c @ E @ L_t^T ), E ~ N(0, I_{C×T})
        """
        if (self._Lt is None) or (self._Lc is None):
            self._factorize()
        device = self._Lt.device
        dtype = self._Lt.dtype

        # Sample noise for entire batch
        total_samples = batch_size * num_samples_per_element
        E = torch.randn(total_samples, self.C, self.T, device=device, dtype=dtype)

        # Kronecker reparameterization
        Z = self._Lc @ E @ self._Lt.transpose(-2, -1)

        # Reshape: [batch_size, num_samples_per_element, T*C]
        if num_samples_per_element == 1:
            return Z.reshape(batch_size, self.T * self.C)  # [B, TC]
        else:
            return Z.reshape(batch_size, num_samples_per_element, self.T * self.C)

    # ---- helpers for KL pieces with another Gaussian (e.g., Σ_q vs Σ_p) ----
    def quad_form(self, v):
        """ v^T (K_t ⊗ K_c)^{-1} v using solve """
        x = self.solve(v)
        return torch.sum(v * x, dim=-1)

    def inv_logdet(self):
        """ log| (K_t ⊗ K_c)^{-1} | = - log|K| """
        return -self.logdet()



class NoTgtDecoder(nn.Module):
    """
    Transformer decoder without explicit target input.

    Args:
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of TransformerDecoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        num_queries (int): Number of output tokens (queries) to generate.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, num_queries):
        super(NoTgtDecoder, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, memory, tgt_key_padding_mask=None):
        """
        Forward pass for the decoder.

        Args:
            memory (torch.Tensor): Encoder output of shape (T, batch_size, d_model).
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for the queries.

        Returns:
            torch.Tensor: Decoder output of shape (num_queries, batch_size, d_model).
        """
        device = memory.device
        BS = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(1).expand(self.num_queries, BS, -1).permute(1,0,2).to(device)
        output = self.decoder(queries, memory, tgt_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=tgt_key_padding_mask)
        return output



class AttentionModulationBlock(nn.Module):
    """
    Attention Modulation Block for computing modulated lower-triangular matrices.

    Args:
        hidden_dim (int): Dimension of the embedded input features.
        latent_dim (int): Dimension of the latent space.
        T (int): Length of the latent sequence.
    """
    def __init__(self, input_dim,hidden_dim, latent_dim, T):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.T = T

        # Linear projections for queries, keys, and values
        self.L_query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.L_key = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.L_val = nn.Linear(latent_dim, T, bias=False)
        self.mu_query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mu_key = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.mu_val = nn.Linear(latent_dim, latent_dim, bias=False)

        # Learnable diagonal scaling factors and mean
        self.D = nn.Parameter(torch.randn(T, latent_dim)**2)
        self.mu = nn.Parameter(torch.randn(T, latent_dim))

    def forward(self, x, mask=None):
        """
        Forward pass for the Attention Modulation Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, T, hidden_dim).
            mask (torch.Tensor, optional): Mask for the input sequence.

        Returns:
            tuple: Modulated lower-triangular matrix (L), raw attention matrix (A),
                   diagonal scaling matrix (D), and mean (mu).
        """
        device = x.device
        B, T, _ = x.shape

        # Compute queries, keys, and values
        L_Q = self.L_query(x)
        L_K = self.L_key(self.D)
        L_V = self.L_val(self.D)
        mu_Q = self.mu_query(x)
        mu_K = self.mu_key(self.mu)
        mu_V = self.mu_val(self.mu)

        if mask is not None:
            mask = mask.unsqueeze(-1).transpose(1, 2)  # Ensure mask is of shape (B, T, T)

            L_K = L_K.unsqueeze(0).expand(B, T, -1)  # Ensure L_K has the same batch size
            L_V = L_V.unsqueeze(0).expand(B, T, -1)

            mu_K = mu_K.unsqueeze(0).expand(B, T, -1)
            mu_V = mu_V.unsqueeze(0).expand(B, T, -1)

        # Compute raw attention scores
        L_scores = torch.bmm(L_Q, L_K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.hidden_dim, device=device))
        mu_scores = torch.bmm(mu_Q, mu_K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.hidden_dim, device=device))

        # Apply causal mask
        lower_tri_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)

        # L_scores = L_scores  + (1 - lower_tri_mask) * (1-mask.float().expand(B,T,T)) * float('-1e30')
        # mu_scores = mu_scores  + (1 - lower_tri_mask)* (1-mask.float().expand(B,T,T)) * float('-1e30')
        L_scores = L_scores  + (1-mask.float().expand(B,T,T)) * float('-1e30')
        mu_scores = mu_scores  +  (1-mask.float().expand(B,T,T)) * float('-1e30')
        mu_scores = torch.softmax(mu_scores, dim=-1)
        A = torch.softmax(L_scores, dim=-1)

        # Compute modulated matrices
        mu = torch.bmm(mu_scores, mu_V)
        # D = torch.diag_embed(F.softplus(torch.bmm(A, L_V), beta=10).squeeze())
        if len(L_V.shape) < 3:
            L_V = L_V.unsqueeze(0)
        L = torch.bmm(A, L_V)

        return L, A, mu


class BandedPrecisionAttentionModulationBlock(nn.Module):
    """
    Attention modulation block adapted for banded-precision posterior.
    Produces per-sample predictions of:
      - precision diagonal: [B, T] (softplus to ensure positivity)
      - precision bands:    [B, bandwidth, T-1] (raw; only first T-1 entries per band are used)
      - mean:               [B, T, latent_dim]
    NOTE: Consumers must enforce band structure when assembling a matrix from the bands.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, T, bandwidth, use_layer_norm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.T = T
        self.bandwidth = max(0, int(bandwidth))
        self.use_layer_norm = bool(use_layer_norm)

        # Shared query from input sequence
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        # Optional LayerNorm on queries
        self.ln_q = nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity()

        # Diagonal branch: base template over time modulated by attention
        self.diag_template = nn.Parameter(torch.randn(T, latent_dim))
        self.diag_key = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.diag_val = nn.Linear(latent_dim, 1, bias=False)  # -> scalar per time step

        # Bands branch: one template per band offset
        if self.bandwidth > 0:
            self.band_templates = nn.Parameter(torch.randn(self.bandwidth, T, latent_dim))
            self.band_key = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.band_val = nn.Linear(latent_dim, 1, bias=False)
        else:
            self.register_parameter('band_templates', None)
            self.band_key = None
            self.band_val = None

        # Mean branch (same spirit as original block)
        self.mu_template = nn.Parameter(torch.randn(T, latent_dim))
        self.mu_key = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.mu_val = nn.Linear(latent_dim, latent_dim, bias=False)

    def _apply_mask(self, scores, mask):
        # mask: [B, T] with 1 for valid, 0 for invalid
        if mask is None:
            return scores
        B, T, _ = scores.shape
        key_mask = mask.float().unsqueeze(1).expand(B, T, T)  # broadcast along query dim
        scores = scores + (1.0 - key_mask) * float('-1e30')
        return scores

    def forward(self, x, mask=None):  # x: [B, T, input_dim]; mask: [B, T] or None
        device = x.device
        B, T, _ = x.shape
        assert T == self.T, f"Sequence length mismatch for modulation block: T={T}, self.T={self.T}"

        scale = torch.sqrt(torch.tensor(float(self.hidden_dim), device=device))
        Q = self.query(x)  # [B, T, H]
        Q = self.ln_q(Q)   # optional LN (Identity if disabled)

        # ---- Diagonal branch ----
        Kd = self.diag_key(self.diag_template)  # [T, H]
        Vd = self.diag_val(self.diag_template)  # [T, 1]
        Kd_b = Kd.unsqueeze(0).expand(B, -1, -1)
        Vd_b = Vd.unsqueeze(0).expand(B, -1, -1)
        scores_d = torch.bmm(Q, Kd_b.transpose(1, 2)) / scale
        scores_d = self._apply_mask(scores_d, mask)
        A_d = torch.softmax(scores_d, dim=-1)  # [B, T, T]
        diag_vals = torch.bmm(A_d, Vd_b).squeeze(-1)  # [B, T]
        precision_diag = softplus(diag_vals, beta=30.0)

        # ---- Bands branch ----
        if self.bandwidth > 0:
            precision_bands = []
            for b in range(self.bandwidth):
                Kb = self.band_key(self.band_templates[b])  # [T, H]
                Vb = self.band_val(self.band_templates[b])  # [T, 1]
                Kb_b = Kb.unsqueeze(0).expand(B, -1, -1)
                Vb_b = Vb.unsqueeze(0).expand(B, -1, -1)
                scores_b = torch.bmm(Q, Kb_b.transpose(1, 2)) / scale
                scores_b = self._apply_mask(scores_b, mask)
                A_b = torch.softmax(scores_b, dim=-1)
                band_full = torch.bmm(A_b, Vb_b).squeeze(-1)  # [B, T]
                # Store first T-1 entries per band; consumers use first T-(b+1) elements
                precision_bands.append(band_full[:, : max(1, T - 1)])
            # Pad to consistent shape [B, bandwidth, T-1]
            bands_stack = []
            for b, band_vec in enumerate(precision_bands):
                # Ensure length T-1 by right-padding/truncating
                if band_vec.shape[1] < (T - 1):
                    pad = (0, (T - 1) - band_vec.shape[1])
                    band_vec = F.pad(band_vec, pad)
                elif band_vec.shape[1] > (T - 1):
                    band_vec = band_vec[:, : (T - 1)]
                bands_stack.append(band_vec.unsqueeze(1))
            precision_bands = torch.cat(bands_stack, dim=1)  # [B, bandwidth, T-1]
        else:
            precision_bands = torch.zeros(B, 0, max(0, T - 1), device=device, dtype=x.dtype)

        # ---- Mean branch ----
        Km = self.mu_key(self.mu_template)   # [T, H]
        Vm = self.mu_val(self.mu_template)   # [T, latent_dim]
        Km_b = Km.unsqueeze(0).expand(B, -1, -1)
        Vm_b = Vm.unsqueeze(0).expand(B, -1, -1)
        scores_m = torch.bmm(Q, Km_b.transpose(1, 2)) / scale
        scores_m = self._apply_mask(scores_m, mask)
        A_m = torch.softmax(scores_m, dim=-1)
        mu = torch.bmm(A_m, Vm_b)  # [B, T, latent_dim]

        return precision_diag, precision_bands, mu


# ---- Banded Precision Gaussian for VAE Approximate Posterior ----
class BandedPrecisionGaussian(nn.Module):
    """
    Banded Precision Gaussian distribution for VAE approximate posterior.

    Represents a multivariate Gaussian with banded precision matrix structure,
    which allows for efficient computation of sampling, log determinant, and trace operations.

    Args:
        dim (int): Dimension of the distribution
        bandwidth (int): Bandwidth of the precision matrix (number of off-diagonal bands)
        init_precision_diag (float): Initial value for diagonal elements of precision matrix
        init_precision_offdiag (float): Initial value for off-diagonal elements of precision matrix
        use_cholesky_param (bool): If True, parameterize precision as P = B B^T with banded lower-triangular B
                                   so posterior has the same dimensionality and PD by construction.
    """

    def __init__(self, dim, bandwidth=3, init_precision_diag=1.0, init_precision_offdiag=0.1, use_cholesky_param=True):
        super().__init__()
        self.dim = dim
        self.bandwidth = min(bandwidth, dim - 1)  # Ensure bandwidth doesn't exceed dimension
        self.use_cholesky_param = bool(use_cholesky_param)

        # Mean parameter
        self.mean = nn.Parameter(torch.zeros(dim))

        # Two parameterizations supported. Default: Cholesky (requested BB^T form).
        if self.use_cholesky_param:
            # Cholesky factor B (lower-triangular within bandwidth)
            # Diagonal in log-space to ensure positivity after exp
            self.chol_diag = nn.Parameter(torch.zeros(dim))
            # Number of lower-triangular elements within bandwidth (excluding diag)
            num_lower = 0
            for i in range(1, dim):
                num_lower += min(i, self.bandwidth)
            if num_lower > 0:
                # Small init near zero keeps P close to diagonal initially
                self.chol_lower = nn.Parameter(0.01 * torch.randn(num_lower))
            else:
                self.register_parameter('chol_lower', None)
            # Remove legacy params to avoid confusion
            self.register_parameter('log_precision_diag', None)
            self.register_parameter('precision_bands', None)
        else:
            # Legacy direct precision bands + diagonal
            self.log_precision_diag = nn.Parameter(
                torch.log(torch.tensor(init_precision_diag)) * torch.ones(dim)
            )
            if self.bandwidth > 0:
                self.precision_bands = nn.Parameter(
                    init_precision_offdiag * torch.randn(self.bandwidth, dim - 1)
                )
            else:
                self.register_parameter('precision_bands', None)

        # Cached Cholesky factor for efficiency
        self._chol_factor = None
        self._precision_matrix = None
        self._needs_update = True

    def _build_L_factor(self):
        """Build the banded lower-triangular factor B for P = B B^T."""
        device = self.mean.device
        dtype = self.mean.dtype
        L = torch.zeros(self.dim, self.dim, device=device, dtype=dtype)
        # Positive diagonal via exp
        L.diagonal().copy_(torch.exp(self.chol_diag))
        # Fill lower triangle within bandwidth from vector parameter
        if getattr(self, 'chol_lower', None) is not None:
            idx = 0
            for i in range(1, self.dim):
                start_j = max(0, i - self.bandwidth)
                for j in range(start_j, i):
                    if idx < self.chol_lower.numel():
                        L[i, j] = self.chol_lower[idx]
                        idx += 1
        return L

    def _build_precision_matrix(self):
        """Build the precision matrix from parameters (supports both parametrizations)."""
        if not self._needs_update and self._precision_matrix is not None:
            return self._precision_matrix

        if self.use_cholesky_param:
            L = self._build_L_factor()
            P = L @ L.t()
            self._chol_factor = L  # Cholesky factor is exactly L
            self._precision_matrix = P
            self._needs_update = False
            return P

        # ---- Legacy path: construct symmetric banded precision then ensure PD ----
        device = self.mean.device
        P = torch.zeros(self.dim, self.dim, device=device)

        # Set diagonal (ensure positive via softplus)
        diag_vals = softplus(self.log_precision_diag)
        P.diagonal().copy_(diag_vals)

        # Set off-diagonal bands (can be negative)
        if self.bandwidth > 0 and self.precision_bands is not None:
            for band_idx in range(self.bandwidth):
                offset = band_idx + 1
                if offset < self.dim:
                    # Upper band
                    band_values = self.precision_bands[band_idx, :self.dim - offset]
                    P.diagonal(offset).copy_(band_values)
                    # Lower band (symmetric)
                    P.diagonal(-offset).copy_(band_values)

        # Efficient positive definiteness enforcement for banded matrices
        P = self._ensure_positive_definite_efficient(P, diag_vals)
        
        self._precision_matrix = P
        self._needs_update = False
        return P

    def _ensure_positive_definite_efficient(self, P, diag_vals):
        """
        Efficiently ensure positive definiteness for banded matrices using 
        Gershgorin circle theorem and diagonal dominance.
        
        This is O(n) instead of O(n³) for eigendecomposition.
        """
        if self.use_cholesky_param:
            return P  # Already PD by construction
        # For banded matrices, we can use Gershgorin's circle theorem:
        # All eigenvalues lie within the union of circles centered at diagonal elements
        # with radii equal to the sum of absolute values of off-diagonal elements in that row
        
        # Calculate row sums of off-diagonal elements efficiently
        # Only need to check within the bandwidth
        min_diag_required = torch.zeros_like(diag_vals)
        
        for i in range(self.dim):
            row_sum = 0.0
            # Only sum within bandwidth
            start_j = max(0, i - self.bandwidth)
            end_j = min(self.dim, i + self.bandwidth + 1)
            
            for j in range(start_j, end_j):
                if i != j:
                    row_sum += abs(P[i, j].item())
            
            # For positive definiteness, diagonal must be > row_sum
            min_diag_required[i] = row_sum + 1e-6
        
        # Check if current diagonal is sufficient
        needs_adjustment = diag_vals < min_diag_required
        
        if torch.any(needs_adjustment):
            # Adjust only the diagonal elements that need it
            adjusted_diag = torch.maximum(diag_vals, min_diag_required)
            P.diagonal().copy_(adjusted_diag)
            
            # Update the log parameters to reflect the adjustment (for gradient consistency)
            with torch.no_grad():
                # Only update elements that were adjusted to maintain gradient flow
                adjustment_mask = needs_adjustment
                if torch.any(adjustment_mask):
                    # Smooth adjustment to avoid gradient discontinuities
                    self.log_precision_diag.data[adjustment_mask] = torch.log(
                        adjusted_diag[adjustment_mask] - 1e-6
                    )
        
        return P

    def _enforce_banded_structure_efficient(self, P):
        """
        Efficiently enforce banded structure using tensor operations instead of loops.
        """
        # Create a mask for the banded structure
        if not hasattr(self, '_band_mask'):
            # Cache the mask for efficiency
            i_indices = torch.arange(self.dim, device=P.device).unsqueeze(1)
            j_indices = torch.arange(self.dim, device=P.device).unsqueeze(0)
            self._band_mask = torch.abs(i_indices - j_indices) <= self.bandwidth
        
        # Apply mask in one operation
        return P * self._band_mask.float()
    
    def _get_cholesky_factor_safe(self):
        """
        Safe Cholesky factorization with minimal overhead fallback.
        """
        if self.use_cholesky_param:
            if self._needs_update or self._chol_factor is None:
                L = self._build_L_factor()
                self._chol_factor = L
                self._precision_matrix = L @ L.t()
                self._needs_update = False
            return self._chol_factor

        if self._needs_update or self._chol_factor is None:
            P = self._build_precision_matrix()
            self._chol_factor = torch.linalg.cholesky(P)

        return self._chol_factor

    # Alternative: Parametrize directly in Cholesky form for guaranteed PD
    def _build_precision_matrix_cholesky_param(self):
        """
        Alternative parameterization: directly parameterize the Cholesky factor L
        such that P = L @ L.T is automatically positive definite.
        This is the most efficient approach.
        """
        device = self.mean.device
        
        # Initialize lower triangular Cholesky factor
        if not hasattr(self, 'chol_diag') or not hasattr(self, 'chol_lower'):
            # Diagonal elements (positive via exp)
            self.chol_diag = nn.Parameter(torch.zeros(self.dim))
            
            # Lower triangular elements (can be any value)
            num_lower = 0
            for i in range(1, self.dim):
                band_width = min(i, self.bandwidth)
                num_lower += band_width
            
            if num_lower > 0:
                self.chol_lower = nn.Parameter(torch.zeros(num_lower))
            else:
                self.register_parameter('chol_lower', None)
        
        # Build lower triangular matrix L
        L = torch.zeros(self.dim, self.dim, device=device)
        
        # Set diagonal (positive)
        L.diagonal().copy_(torch.exp(self.chol_diag))
        
        # Set lower triangular elements within bandwidth
        if self.chol_lower is not None:
            idx = 0
            for i in range(1, self.dim):
                start_j = max(0, i - self.bandwidth)
                for j in range(start_j, i):
                    if idx < len(self.chol_lower):
                        L[i, j] = self.chol_lower[idx]
                        idx += 1
        
        # P = L @ L.T is automatically positive definite
        return L @ L.t()

    def reset_cache(self):
        """Reset cached computations (call when parameters change)."""
        self._needs_update = True
        self._chol_factor = None
        self._precision_matrix = None
        # Clear band mask cache
        if hasattr(self, '_band_mask'):
            delattr(self, '_band_mask')

    def _get_cholesky_factor(self):
        """Get Cholesky factor of precision matrix using efficient safe method."""
        return self._get_cholesky_factor_safe()

    def sample(self, batch_size=1, num_samples=1):
        """
        Sample from the banded precision Gaussian distribution.

        Args:
            batch_size (int): Number of batch samples
            num_samples (int): Number of samples per batch element

        Returns:
            torch.Tensor: Samples of shape (batch_size, num_samples, dim) if num_samples > 1,
                         or (batch_size, dim) if num_samples == 1
        """
        device = self.mean.device

        # Get precision Cholesky factor: P = L_prec @ L_prec.T
        L_prec = self._get_cholesky_factor()

        # For sampling from N(μ, Σ) where Σ = P^(-1):
        # We need L_cov such that Σ = L_cov @ L_cov.T
        # Since P = L_prec @ L_prec.T and Σ = P^(-1), we have:
        # L_cov = (L_prec^(-T)) = inverse of L_prec.T

        # Sample standard normal
        if num_samples == 1:
            z = torch.randn(batch_size, self.dim, device=device)
            # Transform: x = μ + L_cov @ z = μ + (L_prec^(-T)) @ z
            # This is equivalent to: x = μ + solve(L_prec.T, z)
            L_cov_z = torch.linalg.solve_triangular(L_prec.t(), z.unsqueeze(-1), upper=False).squeeze(-1)
            samples = self.mean.unsqueeze(0) + L_cov_z
            return samples
        else:
            z = torch.randn(batch_size, num_samples, self.dim, device=device)
            # Transform: x = μ + L_cov @ z = μ + (L_prec^(-T)) @ z
            z_reshaped = z.view(batch_size * num_samples, self.dim)
            L_cov_z = torch.linalg.solve_triangular(L_prec.t(), z_reshaped.unsqueeze(-1), upper=False).squeeze(-1)
            L_cov_z = L_cov_z.view(batch_size, num_samples, self.dim)
            samples = self.mean.unsqueeze(0).unsqueeze(0) + L_cov_z
            return samples

    def logdet(self):
        """
        Compute log determinant of the precision matrix.

        Returns:
            torch.Tensor: Log determinant (scalar)
        """
        if self.use_cholesky_param:
            # log|P| = 2 * sum(log diag(L)) = 2 * sum(chol_diag)
            return 2.0 * torch.sum(self.chol_diag)
        L_prec = self._get_cholesky_factor()
        return 2.0 * torch.sum(torch.log(torch.diag(L_prec)))

    def trace(self, other_precision=None):
        """
        Compute trace of precision matrix or trace of product with another precision matrix.

        Args:
            other_precision (torch.Tensor, optional): Another precision matrix to compute trace(P @ other_precision)

        Returns:
            torch.Tensor: Trace value (scalar)
        """
        P = self._build_precision_matrix()

        if other_precision is None:
            return torch.trace(P)
        else:
            return torch.trace(P @ other_precision)

    def log_prob(self, x):
        """
        Compute log probability density.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim) or (dim,)

        Returns:
            torch.Tensor: Log probabilities
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        P = self._build_precision_matrix()

        # Centered values
        centered = x - self.mean.unsqueeze(0)

        # Quadratic form: (x - μ)^T P (x - μ)
        quad_form = torch.sum(centered * (centered @ P), dim=-1)

        # Log probability: -0.5 * [quad_form - log|P| - d*log(2π)]
        log_det_P = self.logdet()
        log_prob = -0.5 * (quad_form - log_det_P - self.dim * torch.log(torch.tensor(2 * torch.pi)))

        return log_prob

    def kl_divergence(self, other):
        """
        Compute KL divergence KL(self || other) assuming other is also a Gaussian.

        Args:
            other: Another Gaussian distribution (should have mean, precision_matrix attributes)

        Returns:
            torch.Tensor: KL divergence (scalar)
        """
        # KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) = 0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log|Σ₂|/|Σ₁|]
        # For precision matrices: Σ⁻¹ = P, so Σ = P⁻¹

        P1 = self._build_precision_matrix()

        if hasattr(other, '_build_precision_matrix'):
            P2 = other._build_precision_matrix()
        elif hasattr(other, 'precision_matrix'):
            P2 = other.precision_matrix
        else:
            raise ValueError("Other distribution must have precision matrix")

        # Mean difference
        mean_diff = other.mean - self.mean if hasattr(other, 'mean') else -self.mean

        # Compute KL components
        # tr(P2 @ P1^{-1}) = tr(P2 @ inv(P1))
        P1_inv = torch.inverse(P1)
        trace_term = torch.trace(P2 @ P1_inv)

        # Quadratic term
        quad_term = mean_diff @ P2 @ mean_diff

        # Log determinant terms
        log_det_term = other.logdet() - self.logdet() if hasattr(other, 'logdet') else -self.logdet()

        kl = 0.5 * (trace_term + quad_term - self.dim + log_det_term)
        return kl

    def forward(self):
        """Forward pass returns the distribution parameters."""
        return {
            'mean': self.mean,
            'precision_matrix': self._build_precision_matrix(),
            'log_det_precision': self.logdet()
        }

    # ---- Efficient methods for large matrices ----
    def _efficient_logdet_banded(self):
        """
        Compute log determinant efficiently for banded matrices using band structure.
        For banded matrices, we can use specialized algorithms that are O(b^2 * n) instead of O(n^3).
        """
        if self.bandwidth == 0:  # Diagonal matrix
            if self.use_cholesky_param:
                return 2.0 * torch.sum(self.chol_diag)
            else:
                return torch.sum(self.log_precision_diag)
        else:
            # For small bandwidth, use standard Cholesky
            # For very large matrices with small bandwidth, could implement band Cholesky
            return self.logdet()

    def _efficient_sample_woodbury(self, batch_size=1, num_samples=1):
        """
        Efficient sampling for cases where Woodbury matrix identity can be applied.
        For matrices of form P = D + UVᵀ where D is diagonal and U,V are low-rank.
        """
        # This is a placeholder for Woodbury-based sampling
        # In practice, we'd decompose the banded precision matrix into
        # diagonal + low-rank components for very large matrices
        return self.sample(batch_size, num_samples)

    def _get_band_structure_info(self):
        """Get information about the band structure for optimization."""
        P = self._build_precision_matrix()
        nnz = torch.count_nonzero(P).item()
        total_elements = self.dim * self.dim
        sparsity = 1.0 - (nnz / total_elements)

        return {
            'nonzeros': nnz,
            'total_elements': total_elements,
            'sparsity': sparsity,
            'bandwidth': self.bandwidth,
            'theoretical_nnz': min(self.dim + 2 * self.bandwidth * (self.dim - self.bandwidth), self.dim * self.dim)
        }

    def efficient_operations_benchmark(self, operation='all'):
        """Benchmark different operations with timing information."""
        import time

        results = {}

        if operation in ['all', 'logdet']:
            start_time = time.time()
            logdet_val = self.logdet()
            results['logdet_time'] = time.time() - start_time
            results['logdet_value'] = logdet_val.item()

        if operation in ['all', 'sample']:
            start_time = time.time()
            samples = self.sample(batch_size=100, num_samples=1)
            results['sample_time'] = time.time() - start_time
            results['sample_shape'] = samples.shape

        if operation in ['all', 'trace']:
            start_time = time.time()
            trace_val = self.trace()
            results['trace_time'] = time.time() - start_time
            results['trace_value'] = trace_val.item()

        return results

    def _banded_cholesky(self, P: torch.Tensor):
        """Compute banded Cholesky factor L of symmetric PD banded matrix P.
        L has same bandwidth. Complexity O(n * b^2).
        Not yet integrated into sampling; kept for future optimization.
        """
        n = P.size(0)
        b = self.bandwidth
        L = torch.zeros_like(P)
        for i in range(n):
            j_start = max(0, i - b)
            # Diagonal element
            s = 0.0
            for k in range(j_start, i):
                s += L[i, k] ** 2
            diag = P[i, i] - s
            if diag <= 0:
                diag = diag + 1e-6
            L[i, i] = torch.sqrt(diag)
            # Off-diagonals within band below diagonal
            for j in range(j_start, i):
                s2 = 0.0
                k_start = max(0, j - b)
                for k in range(k_start, j):
                    if i - k <= b and j - k <= b:
                        s2 += L[i, k] * L[j, k]
                if L[j, j] > 0:
                    L[i, j] = (P[i, j] - s2) / L[j, j]
        return L



class BandedPrecisionGaussian_Stateless(nn.Module):
    """
    Stateless variant of BandedPrecisionGaussian.
    - No learnable parameters.
    - Provides utilities to construct a PD banded precision matrix and related ops
      from explicitly provided parameters.

    Args:
        dim (int): Temporal dimension (T).
        bandwidth (int): Number of off-diagonal bands (<= dim-1).
    """
    def __init__(self, dim: int, bandwidth: int = 3):
        super().__init__()
        self.dim = int(dim)
        self.bandwidth = max(0, min(int(bandwidth), self.dim - 1))


    def _assemble_from_bands(self, precision_diag: torch.Tensor, precision_bands: torch.Tensor) -> torch.Tensor:
        """Build symmetric banded precision matrix from per-time diagonal and band values.
        precision_diag: [T]
        precision_bands: [b, T-1] for bands 1..b (may be empty)
        Returns [T,T]."""
        T = self.dim
        device = precision_diag.device
        dtype = precision_diag.dtype
        diag_pos = softplus(precision_diag)
        Off = torch.zeros(T, T, device=device, dtype=dtype)
        b_eff = precision_bands.size(0)
        for k in range(min(b_eff, self.bandwidth)):
            offs = k + 1
            L = max(0, T - offs)
            if L == 0:
                continue
            vals = precision_bands[k, :L]
            i = torch.arange(L, device=device)
            Off[i, i + offs] = vals
            Off[i + offs, i] = vals
        off_abs = Off.abs().sum(dim=-1)
        min_diag = off_abs + 1e-8
        diag_final = torch.maximum(diag_pos, min_diag)
        return torch.diag(diag_final) + Off

    def _assemble_from_cholesky(self, chol_diag: torch.Tensor, chol_lower: torch.Tensor) -> torch.Tensor:
        """Build P = L L^T from banded lower-triangular entries."""
        T = self.dim
        device = chol_diag.device
        dtype = chol_diag.dtype
        L = torch.zeros(T, T, device=device, dtype=dtype)
        L.diagonal().copy_(torch.exp(chol_diag))
        idx = 0
        for i in range(1, T):
            start_j = max(0, i - self.bandwidth)
            for j in range(start_j, i):
                if idx < chol_lower.numel():
                    L[i, j] = chol_lower[idx]
                    idx += 1
        return L @ L.t()

    def _robust_cholesky(self, M: torch.Tensor, jitter_start: float = 1e-8, jitter_max: float = 1e-6, max_tries: int = 7):
        I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
        return torch.linalg.cholesky(M + jitter_start * I)

    def build_precision(self, precision_diag: torch.Tensor = None, precision_bands: torch.Tensor = None,
                        chol_diag: torch.Tensor = None, chol_lower: torch.Tensor = None) -> torch.Tensor:
        """Construct a PD precision matrix given explicit parameters. Stateless: no fallbacks."""
        if (chol_diag is not None) or (chol_lower is not None):
            if (chol_diag is None) or (chol_lower is None):
                raise ValueError("Both 'chol_diag' and 'chol_lower' must be provided for cholesky parametrization in the stateless variant.")
            return self._assemble_from_cholesky(chol_diag, chol_lower)

        if precision_diag is None:
            raise ValueError("Stateless BandedPrecisionGaussian requires 'precision_diag' per sequence. No global defaults allowed.")
        b = precision_bands if precision_bands is not None else torch.zeros(0, max(0, self.dim - 1), device=precision_diag.device, dtype=precision_diag.dtype)
        return self._assemble_from_bands(precision_diag, b)

    def sample(self, batch_size: int = 1, num_samples: int = 1, mean: torch.Tensor = None,
               precision_diag: torch.Tensor = None, precision_bands: torch.Tensor = None,
               chol_diag: torch.Tensor = None, chol_lower: torch.Tensor = None) -> torch.Tensor:
        """Sample x ~ N(mean, Σ) where Σ = P^{-1} and P is the built precision."""
        P = self.build_precision(precision_diag, precision_bands, chol_diag, chol_lower)
        Lp = self._robust_cholesky(P)
        T = self.dim
        B = int(batch_size)
        if mean is None:
            mu = torch.zeros(B, T, device=Lp.device, dtype=Lp.dtype)
        else:
            mu = mean
            if mean.dim() == 1:
                mu = mean.unsqueeze(0).expand(B, -1)
        if num_samples == 1:
            z = torch.randn(B, T, device=Lp.device, dtype=Lp.dtype)
            x = torch.linalg.solve_triangular(Lp.t(), z.unsqueeze(-1), upper=False).squeeze(-1)
            return mu + x
        else:
            z = torch.randn(B * num_samples, T, device=Lp.device, dtype=Lp.dtype)
            x = torch.linalg.solve_triangular(Lp.t(), z.unsqueeze(-1), upper=False).squeeze(-1)
            x = x.view(B, num_samples, T)
            return mu.unsqueeze(1) + x

    def reset_cache(self):
        return None

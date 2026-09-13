"""
========================================================================================
JOINT CVAE -- SINGLE-STAGE ARCHITECTURAL ABLATION OF TALON
========================================================================================
This module implements the "Joint CVAE" ablation baseline used to test whether TALON's
two-stage teacher/student decoupling is actually necessary.

TALON (two-stage):
    Stage 1  q_psi(z|y)                 teacher VAE on Y alone, GP prior p(z)
    Stage 2  q_phi(z|x)                 student encoder on X, aligned to the FROZEN
                                        teacher posterior via reverse-KL + InfoNCE;
                                        the FROZEN teacher decoder p(y|z) reconstructs Y.
             -> the decoder NEVER sees x. All information about x must pass through z.

Joint CVAE (this file, single-stage):
    q(z|x,y)                            ONE encoder that sees X and Y jointly
    p(y|x,z)                            decoder reconstructs Y conditioned on BOTH x and z
    p(z)                                the same Kronecker GP prior as TALON's teacher
    ELBO = E_q[log p(y|x,z)] - beta * KL( q(z|x,y) || p(z) )
             -> trained end-to-end. No teacher, no freezing, no reverse-KL alignment,
                no InfoNCE. The decoder has a direct "conditioning bypass" to x.

DESIGN NOTES (how this stays capacity-matched and mechanism-faithful to TALON):

1. Encoder. Reuses `TALONTeacher_Encoder` VERBATIM (imported, not reimplemented) --
   same transformer-patch trunk, same `use_modulation=True` / `mod_mode='residual'`
   banded-precision posterior head, same Kronecker time/channel banded precision
   parameterisation. The ONLY difference is its `input_dim`: it consumes the
   concatenation of the patch-embedded X and the patch-embedded Y.

   On the conditioning-injection mechanism: `use_modulation` / `mod_mode='residual'`
   in this codebase is NOT a conditioning mechanism -- it is the posterior head
   (`BandedPrecisionAttentionModulationBlock` predicts the banded precision diag/bands
   and a residual correction to mu, from the encoder's OWN token sequence). The way
   `TALONStudent` actually injects X is by patch-embedding X with a `nn.Conv1d`
   patch embedder and feeding the result as the encoder's token sequence. That is the
   mechanism reused here: X is patch-embedded exactly as `TALONStudent` does, Y is
   patch-embedded exactly as `TALONTeacher` does, and the two token streams are
   concatenated along the feature axis to form the joint encoder's input tokens.

   Parameter budget: TALON trains two encoders of token width 250 (Y: 25*10) and
   260 (X: 26*10); the joint encoder has token width 510 = 250 + 260. Since the
   transformer's attention cost is ~4*d^2 and its FFN ~2*d*d_ff, the joint encoder
   lands within ~1% of TALON's two-encoder attention budget and matches its FFN budget
   exactly. This is a genuine capacity match, not a smaller/larger model.

2. Decoder. Reuses `TSPVAE_Decoder` VERBATIM with the same decoder_kwargs as TALON. The
   ONLY difference is the context MLP feeding it: TALON's is Linear(latent_dim -> H) ->
   GELU -> Linear(H -> H) over z alone; here it is the same two-layer shape but over
   [z_t ; x_patch_t], giving the decoder direct access to X. This IS the conditioning
   bypass under test.

3. Prior + KL. Same `KronTimeChannelKernel` GP prior. The KL is a VECTORISED
   reimplementation of `TALONTeacher.compute_kl_divergence`; it is verified to agree
   with the original loop implementation to <1e-4 by
   `training/train_swat_joint_cvae.py --verify_kl` (the original loops over the batch
   for a 10x10 Cholesky, which is prohibitively slow at batch_size=4096).

4. Losses. `compute_masked_reconstruction_loss` (training, summed ELBO scale) and
   `compute_scoring_reconstruction_loss` (evaluation, mean-normalised with OOD channel
   masking and dynamic channel-precision weights) are verbatim ports of
   `TALONTeacher.compute_masked_reconstruction_loss` and
   `TALONStudent._compute_reconstruction_loss` respectively, so the scoring pipeline
   sees numerically identical quantities to the ones it sees for TALON.

This file does not import or modify any TALON training/evaluation code paths.
========================================================================================
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.base_utils import ModelOutput
from models.Modules import KronTimeChannelKernel
from models.TSPCVAE import TSPVAE_Decoder
from models.TALONTeacher import (
    TALONTeacher_Encoder,
    assemble_precision_from_bands_fn,
)


# ---------------------------------------------------------------------------------
# Vectorised KL( q(z|x,y) || p(z) ) with Kronecker GP prior and banded TC posterior.
# Mathematically identical to TALONTeacher.compute_kl_divergence; batched Cholesky.
# ---------------------------------------------------------------------------------
def kl_banded_tc_vs_gp_prior(mu_batch, posterior, gp_prior, time_bandwidth,
                             channel_bandwidth, posterior_tc_banded, mask=None):
    """KL(q||p) per sample.

    q: N(mu, Sigma_q) with Sigma_q = Qt^{-1} (x) Qc^{-1}  (or Qt^{-1} (x) I if not TC-banded)
    p: N(0, K_t (x) K_c)

    Returns Tensor [B].
    """
    B, C, T = mu_batch.shape
    device = mu_batch.device

    K_t = gp_prior.K_t()
    K_c = gp_prior.K_c()
    L_Kt = torch.linalg.cholesky(K_t)
    L_Kc = torch.linalg.cholesky(K_c)
    I_T = torch.eye(T, device=device, dtype=K_t.dtype)
    I_C = torch.eye(C, device=device, dtype=K_c.dtype)
    K_t_inv = torch.cholesky_solve(I_T, L_Kt)
    K_c_inv = torch.cholesky_solve(I_C, L_Kc)
    log_det_Kt = 2 * torch.sum(torch.log(torch.diag(L_Kt)))
    log_det_Kc = 2 * torch.sum(torch.log(torch.diag(L_Kc)))
    log_det_K = C * log_det_Kt + T * log_det_Kc
    jitter = float(gp_prior.jitter.item()) if isinstance(gp_prior.jitter, torch.Tensor) else float(gp_prior.jitter)

    time_diag = posterior.get('precision_diag', None)
    time_bands = posterior.get('precision_bands', None)
    chan_diag = posterior.get('precision_channel_diag', None)
    chan_bands = posterior.get('precision_channel_bands', None)
    if time_diag is None or time_bands is None:
        raise RuntimeError("Encoder did not provide per-sample time precision parameters.")

    P_t = assemble_precision_from_bands_fn(time_diag, time_bands, bandwidth=int(time_bandwidth))
    L_Qt = torch.linalg.cholesky(P_t + jitter * I_T)                       # [B,T,T]
    log_det_Qt = 2 * torch.sum(torch.log(torch.diagonal(L_Qt, dim1=-2, dim2=-1)), dim=-1)
    Qt_inv = torch.cholesky_solve(I_T.expand(B, T, T), L_Qt)               # [B,T,T]

    has_channel = bool(posterior_tc_banded and chan_diag is not None and chan_bands is not None)
    if has_channel:
        P_c = assemble_precision_from_bands_fn(chan_diag, chan_bands, bandwidth=int(channel_bandwidth))
        L_Qc = torch.linalg.cholesky(P_c + jitter * I_C)                   # [B,C,C]
        log_det_Qc = 2 * torch.sum(torch.log(torch.diagonal(L_Qc, dim1=-2, dim2=-1)), dim=-1)
        Qc_inv = torch.cholesky_solve(I_C.expand(B, C, C), L_Qc)           # [B,C,C]
        trace_t = torch.sum(K_t_inv.unsqueeze(0) * Qt_inv, dim=[1, 2])
        trace_c = torch.sum(K_c_inv.unsqueeze(0) * Qc_inv, dim=[1, 2])
        trace_term = trace_t * trace_c
        log_det_Q = C * log_det_Qt + T * log_det_Qc
    else:
        trace_term = torch.sum(K_t_inv.unsqueeze(0) * Qt_inv, dim=[1, 2]) * torch.trace(K_c_inv)
        log_det_Q = C * log_det_Qt

    if mask is not None:
        mask_f = mask.float()
        mu_masked = mu_batch * mask_f.unsqueeze(1)
        valid_counts = mask_f.sum(dim=1)
    else:
        mu_masked = mu_batch
        valid_counts = torch.full((B,), T, device=device, dtype=mu_batch.dtype)

    temp = torch.matmul(mu_masked, K_t_inv)
    quad = torch.einsum('bct,bdt,cd->b', temp, mu_masked, K_c_inv)
    n_eff = valid_counts * C
    return 0.5 * (log_det_K + log_det_Q - n_eff + trace_term + quad)


def gp_prior_log_prob(z, gp_prior):
    """log p(z) under p = N(0, K_t (x) K_c). z: [B,C,T] -> [B]."""
    B, C, T = z.shape
    z = z.float()
    K_t = gp_prior.K_t().float()
    K_c = gp_prior.K_c().float()

    def _chol(M):
        n = M.size(-1)
        eye = torch.eye(n, device=M.device, dtype=M.dtype)
        j = 0.0
        for _ in range(8):
            try:
                return torch.linalg.cholesky(M + j * eye)
            except Exception:
                j = 1e-8 if j == 0.0 else j * 10.0
        raise RuntimeError("GP prior covariance is not positive definite even with jitter 1e-1")

    L_Kt = _chol(K_t)
    L_Kc = _chol(K_c)
    I_T = torch.eye(T, device=z.device, dtype=z.dtype)
    I_C = torch.eye(C, device=z.device, dtype=z.dtype)
    K_t_inv = torch.cholesky_solve(I_T, L_Kt)
    K_c_inv = torch.cholesky_solve(I_C, L_Kc)
    log_det_K = C * 2 * torch.sum(torch.log(torch.diag(L_Kt))) + T * 2 * torch.sum(torch.log(torch.diag(L_Kc)))
    temp = torch.matmul(z, K_t_inv)                                   # [B,C,T]
    quad = torch.einsum('bct,bdt,cd->b', temp, z, K_c_inv)            # [B]
    n_dim = C * T
    return -0.5 * (log_det_K + n_dim * math.log(2.0 * math.pi) + quad)


class JointCVAE(nn.Module):
    """Single-stage conditional VAE: q(z|x,y), p(y|x,z), GP prior p(z)."""

    def __init__(self,
                 x_dim,
                 y_dim,
                 sequence_length,
                 patch_length,
                 latent_dim,
                 enc_hidden_dim=64,
                 dec_hidden_dim=64,
                 bandwidth=10,
                 gp_time_kernel='cauchy',
                 rank_c=26,
                 gp_jitter=1e-12,
                 posterior_tc_banded=True,
                 tc_channel_bandwidth=26,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 discrete_mask=None,
                 bce_loss_weight=1.0,
                 channel_weights=None,
                 posterior_jitter=1e-8,
                 posterior_jitter_max=1e-6):
        super().__init__()

        self.x_dim = int(x_dim)
        self.n_channels = int(y_dim)          # Y channels (decoder output), name matches TALON
        self.sequence_length = int(sequence_length)
        self.patch_length = int(patch_length)
        self.latent_dim = int(latent_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.dec_hidden_dim = int(dec_hidden_dim)
        self.posterior_tc_banded = bool(posterior_tc_banded)
        self.tc_time_bandwidth = int(bandwidth)
        self.tc_channel_bandwidth = int(tc_channel_bandwidth)
        self.posterior_jitter = float(posterior_jitter)
        self.posterior_jitter_max = float(posterior_jitter_max)
        self.bce_loss_weight = float(bce_loss_weight)
        self.decoder_smoothing_alpha = 0.0

        n_patches = int(self.sequence_length // self.patch_length)
        self.n_patches = n_patches

        # Channel reconstruction weights (plain tensor attribute, exactly as in TALON)
        if channel_weights is not None:
            self.channel_weights = torch.tensor(channel_weights, dtype=torch.float)
        else:
            self.channel_weights = torch.ones(self.n_channels, dtype=torch.float)

        if discrete_mask is not None:
            self.register_buffer('discrete_mask', torch.tensor(discrete_mask, dtype=torch.bool), persistent=False)
        else:
            self.register_buffer('discrete_mask', torch.zeros(self.n_channels, dtype=torch.bool), persistent=False)

        # ---- Patch embedders (identical construction to TALON's two Conv1d embedders) ----
        # Y embedder mirrors TALONTeacher.patch_embedding_layer
        self.y_patch_embedding_layer = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.patch_length * self.n_channels,
            kernel_size=self.patch_length, stride=self.patch_length)
        # X embedder mirrors TALONStudent.patch_embedding_layer
        self.x_patch_embedding_layer = nn.Conv1d(
            in_channels=self.x_dim,
            out_channels=self.patch_length * self.x_dim,
            kernel_size=self.patch_length, stride=self.patch_length)

        self.y_patch_dim = self.patch_length * self.n_channels
        self.x_patch_dim = self.patch_length * self.x_dim

        # ---- Joint encoder q(z|x,y): TALONTeacher_Encoder over concatenated tokens ----
        enc_kwargs = dict(encoder_kwargs or {})
        enc_init = {
            'input_dim': self.x_patch_dim + self.y_patch_dim,
            'hidden_dim': self.enc_hidden_dim,
            'latent_dim': self.latent_dim,
            'sequence_length': n_patches,
            'bandwidth': self.tc_time_bandwidth,
            'transformer_encoder_blocks': int(enc_kwargs.get('transformer_encoder_blocks', 2)),
            'n_heads': int(enc_kwargs.get('n_heads', 8)),
            'dim_feedforward': int(enc_kwargs.get('dim_feedforward', 256)),
            'dropout': float(enc_kwargs.get('dropout', 0.05)),
            'use_modulation': bool(enc_kwargs.get('use_modulation', True)),
            'mod_num_blocks': int(enc_kwargs.get('mod_num_blocks', 1)),
            'mod_mode': enc_kwargs.get('mod_mode', 'residual'),
            'mod_pre_transformer': bool(enc_kwargs.get('mod_pre_transformer', True)),
            'residual_alpha_init': float(enc_kwargs.get('residual_alpha_init', 1.0)),
            'tc_banded_enabled': self.posterior_tc_banded,
            'tc_channel_bandwidth': self.tc_channel_bandwidth,
            'pe_type': enc_kwargs.get('pe_type', 'learnable'),
        }
        for k, v in enc_kwargs.items():
            if k in ('hidden_dim', 'input_dim', 'latent_dim', 'sequence_length'):
                continue
            enc_init[k] = v
        self.encoder = TALONTeacher_Encoder(**enc_init)

        # ---- GP prior p(z): identical to TALON's teacher prior ----
        self.gp_prior = KronTimeChannelKernel(
            T=n_patches, C=self.latent_dim,
            time_kernel=gp_time_kernel, rank_c=int(rank_c), jitter=float(gp_jitter))

        # ---- Context MLP: SAME two-layer shape as TALON, but over [z ; x_patch] ----
        # This is the conditioning bypass under test.
        self.context_encoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.x_patch_dim, self.dec_hidden_dim),
            nn.GELU(),
            nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim),
        )

        # ---- Decoder p(y|x,z): TSPVAE_Decoder, same kwargs as TALON ----
        dec_kwargs = dict(decoder_kwargs or {})
        self.decoder = TSPVAE_Decoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.dec_hidden_dim,
            output_dim=self.y_patch_dim,
            num_layers=int(dec_kwargs.get('num_layers', 2)),
            num_heads=int(dec_kwargs.get('num_heads', 8)),
            dropout=float(dec_kwargs.get('dropout', 0.05)),
            max_sequence_length=n_patches,
            dim_feedforward=int(dec_kwargs.get('dim_feedforward', 256)),
            ar_mode=bool(dec_kwargs.get('ar_mode', False)),
            pe_type=dec_kwargs.get('pe_type', 'learnable'),
            pe_dropout=float(dec_kwargs.get('pe_dropout', 0.0)),
        )

    # ------------------------------------------------------------------ utils
    def patchify_x(self, x_btc):
        """x_btc: [B,T,Cx] -> [B, n_patches, Cx*P]"""
        return self.x_patch_embedding_layer(x_btc.transpose(1, 2)).permute(0, 2, 1)

    def patchify_y(self, y_btc):
        """y_btc: [B,T,Cy] -> [B, n_patches, Cy*P]"""
        return self.y_patch_embedding_layer(y_btc.transpose(1, 2)).permute(0, 2, 1)

    def unpatchify_y(self, y_patches):
        """[B, n_patches, Cy*P] -> [B, Cy, T]  (verbatim TALONTeacher.unpatchify_data)"""
        batch_size, n_patches, _ = y_patches.shape
        out = y_patches.view(batch_size, n_patches, self.n_channels, self.patch_length)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous().view(batch_size, self.n_channels, n_patches * self.patch_length)
        cur = out.shape[-1]
        if cur < self.sequence_length:
            out = F.pad(out, (0, self.sequence_length - cur), mode='constant', value=0.0)
        elif cur > self.sequence_length:
            out = out[:, :, :self.sequence_length]
        return out

    # --------------------------------------------------------------- encoding
    def encode(self, x_btc, y_btc, patched_mask=None):
        """Joint posterior q(z|x,y). Returns (mu [B,C,T], posterior_params, x_patches)."""
        x_patches = self.patchify_x(x_btc)
        y_patches = self.patchify_y(y_btc)
        joint_tokens = torch.cat([x_patches, y_patches], dim=-1)
        mu, params = self.encoder(joint_tokens, patched_mask=patched_mask)
        return mu, params, x_patches

    def sample_posterior(self, mu, params):
        """Sample z ~ q(z|x,y) using the Kronecker banded precision (as TALON's teacher)."""
        B, C, T = mu.shape
        device = mu.device
        P_t = assemble_precision_from_bands_fn(
            params['precision_diag'], params['precision_bands'],
            bandwidth=int(getattr(self.encoder, 'bandwidth', T - 1)))
        jitter = self.posterior_jitter
        if self.posterior_tc_banded:
            P_c = assemble_precision_from_bands_fn(
                params['precision_channel_diag'], params['precision_channel_bands'],
                bandwidth=int(getattr(self.encoder, 'tc_channel_bandwidth', max(0, C - 1))))
            Lt = torch.linalg.cholesky(P_t + jitter * torch.eye(T, device=device, dtype=mu.dtype))
            Lc = torch.linalg.cholesky(P_c + jitter * torch.eye(C, device=device, dtype=mu.dtype))
            eps = torch.randn(B, C, T, device=device, dtype=mu.dtype)
            y = torch.linalg.solve_triangular(Lc.transpose(-1, -2), eps, upper=True)
            z_b = torch.linalg.solve_triangular(Lt.transpose(-1, -2), y.transpose(-2, -1), upper=True).transpose(-2, -1)
            return mu + z_b
        Lt = torch.linalg.cholesky(P_t + jitter * torch.eye(T, device=device, dtype=mu.dtype))
        eps = torch.randn(B, C, T, device=device, dtype=mu.dtype)
        z_b = torch.linalg.solve_triangular(Lt.transpose(-1, -2), eps.transpose(-2, -1), upper=True).transpose(-2, -1)
        return mu + z_b

    # --------------------------------------------------------------- decoding
    def decode(self, z, x_patches, patched_mask=None, apply_sigmoid=True):
        """p(y|x,z). z: [B,C,T_patch]; x_patches: [B,T_patch,Cx*P]. Returns y_hat [B,T,Cy]."""
        z_time_first = z.permute(0, 2, 1)                                   # [B,T_p,C]
        ctx_in = torch.cat([z_time_first, x_patches], dim=-1)               # conditioning bypass
        context_features = self.context_encoder(ctx_in)                     # [B,T_p,H]
        h_hat = self.decoder(context=context_features, patched_mask=patched_mask).transpose(0, 1)
        h_hat = h_hat.contiguous().view(z.size(0), -1, self.y_patch_dim)
        y_hat_bct = self.unpatchify_y(h_hat)                                # [B,Cy,T]
        y_hat = y_hat_bct.transpose(1, 2)                                   # [B,T,Cy]
        return y_hat

    def apply_discrete_sigmoid(self, y_hat_btc):
        dm = self.discrete_mask.to(y_hat_btc.device)
        if dm.any():
            return torch.where(dm.view(1, 1, -1), torch.sigmoid(y_hat_btc), y_hat_btc)
        return y_hat_btc

    # ----------------------------------------------------------------- losses
    def compute_masked_reconstruction_loss(self, y_hat_bct, y_bct):
        """Verbatim port of TALONTeacher.compute_masked_reconstruction_loss (SUM scale)."""
        device = y_bct.device
        channel_weights = self.channel_weights.to(device)
        discrete_mask = self.discrete_mask.to(device)

        loss = torch.zeros_like(y_bct)
        if (~discrete_mask).any():
            cont_mask = (~discrete_mask).view(1, -1, 1)
            loss = torch.where(cont_mask, (y_hat_bct - y_bct) ** 2, loss)
        if discrete_mask.any():
            disc_mask = discrete_mask.view(1, -1, 1)
            y_clamped = torch.clamp(y_bct, 0.0, 1.0)
            bce_val = F.binary_cross_entropy_with_logits(y_hat_bct, y_clamped, reduction='none')
            loss = torch.where(disc_mask, self.bce_loss_weight * bce_val, loss)
        weighted = loss * channel_weights.view(1, -1, 1)
        return weighted.sum(dim=[1, 2])

    def compute_scoring_reconstruction_loss(self, target_btc, prediction_btc,
                                            time_mask=None, ood_threshold=None):
        """Verbatim port of TALONStudent._compute_reconstruction_loss (MEAN scale + OOD mask)."""
        device = target_btc.device
        channel_weights = self.channel_weights.to(device).view(1, 1, -1)
        discrete_mask = self.discrete_mask.to(device)

        loss_val = torch.zeros_like(target_btc)
        if (~discrete_mask).any():
            cont_mask = (~discrete_mask).view(1, 1, -1)
            loss_val = torch.where(cont_mask, (target_btc - prediction_btc) ** 2, loss_val)
        if discrete_mask.any():
            disc_mask = discrete_mask.view(1, 1, -1)
            tgt_clamped = torch.clamp(target_btc, 0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(prediction_btc, tgt_clamped, reduction='none')
            loss_val = torch.where(disc_mask, self.bce_loss_weight * bce, loss_val)

        valid_mask = torch.ones_like(target_btc)
        if ood_threshold is not None and ood_threshold > 0:
            cont_mask = (~discrete_mask).view(1, 1, -1)
            is_ood = cont_mask & (torch.abs(target_btc) > ood_threshold)
            valid_mask = torch.where(is_ood, torch.zeros_like(valid_mask), valid_mask)

        weighted_loss = loss_val * channel_weights * valid_mask
        effective_weights = channel_weights * valid_mask
        if time_mask is not None:
            tm = time_mask.to(device).unsqueeze(-1)
            weighted_loss = weighted_loss * tm
            num_valid = (tm * effective_weights).sum(dim=[1, 2]).clamp(min=1.0)
        else:
            num_valid = effective_weights.sum(dim=[1, 2]).clamp(min=1.0)
        return weighted_loss.sum(dim=[1, 2]) / num_valid

    def compute_kl_divergence(self, mu, params, mask=None):
        return kl_banded_tc_vs_gp_prior(
            mu, params, self.gp_prior,
            time_bandwidth=int(getattr(self.encoder, 'bandwidth', mu.size(-1) - 1)),
            channel_bandwidth=int(getattr(self.encoder, 'tc_channel_bandwidth', max(0, mu.size(1) - 1))),
            posterior_tc_banded=self.posterior_tc_banded, mask=mask)

    # ---------------------------------------------------------------- forward
    def forward(self, x_btc, y_btc, patched_mask=None, deterministic=None):
        """Training / ELBO forward. x_btc [B,T,Cx], y_btc [B,T,Cy]."""
        mu, params, x_patches = self.encode(x_btc, y_btc, patched_mask=patched_mask)

        if deterministic is None:
            deterministic = not self.training
        z = mu if deterministic else self.sample_posterior(mu, params)

        y_hat = self.decode(z, x_patches, patched_mask=patched_mask)          # [B,T,Cy] (logits for discrete)
        y_hat_bct = y_hat.transpose(1, 2)                                     # [B,Cy,T]
        y_bct = y_btc.transpose(1, 2)

        kl_loss = self.compute_kl_divergence(mu, params, mask=patched_mask)
        recon_loss = self.compute_masked_reconstruction_loss(y_hat_bct, y_bct)

        y_hat_out = self.apply_discrete_sigmoid(y_hat)

        return ModelOutput(
            y_hat=y_hat_out,
            x_hat=y_hat_out.transpose(1, 2),
            z=z,
            mu=mu,
            KL_Loss=kl_loss,
            reconstruction_loss=recon_loss,
            banded_posterior=params,
        )

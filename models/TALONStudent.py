"""
Enhanced Time-Series Pair Conditioning Encoder (TALONStudent) Model

This module contains the TALONStudent model, which is an encoder-only conditioning model
that learns an approximate posterior to match the pre-trained TALONTeacher's banded 
posterior distribution.
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.base_utils import ModelOutput
from utils.mutual_information import extract_diagonal_variances, MIEstimator
from models.Modules import *
from models.TSPCVAE import TransformerBlock
# New: import functional helpers for precision assembly and robust cholesky
from models.TALONTeacher import assemble_precision_from_bands_fn, robust_cholesky_fn
torch.set_default_dtype(torch.float)


def robust_cholesky_batched(M, jitter_start=1e-6, jitter_max=1e-2, max_tries=5):
    """
    Computes Cholesky decomposition of a batch of symmetric matrices M [B, N, N].
    If it fails or contains NaNs, retries with exponentially increasing jitter.
    """
    B, N, _ = M.shape
    device = M.device
    dtype = M.dtype
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    
    jitter = jitter_start
    for attempt in range(max_tries):
        try:
            L = torch.linalg.cholesky(M + jitter * I)
            if not torch.isnan(L).any():
                return L
        except RuntimeError:
            pass
        jitter *= 10.0
        if jitter > jitter_max:
            break
            
    # Final fallback: add large jitter and force it
    return torch.linalg.cholesky(M + jitter_max * I)


class TALONStudent_Encoder(nn.Module):
    """
    Enhanced Time-Series Pair Conditioning Encoder.
    This is an encoder-only model that learns an approximate posterior to match 
    the pre-trained TALONTeacher's banded posterior distribution.
    
    Uses the EXACT same architecture as TALONTeacher_Encoder.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        latent_dim (int): Latent space dimension.
        sequence_length (int): Length of the input sequence.
        bandwidth (int): Bandwidth for the banded precision matrix.
        tc_banded_enabled (bool): Enable TC-banded posterior structure.
        tc_channel_bandwidth (int): Channel bandwidth for TC-banded structure.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length,
                 bandwidth=3, transformer_encoder_blocks=2, n_heads=4,
                 dim_feedforward=2048, dropout=0.1, init_precision_diag=1.0, init_precision_offdiag=0.1,
                 use_modulation=False, mod_num_blocks=1, mod_mode='residual', mod_pre_transformer=True,
                 residual_alpha_init=1.0, use_cholesky_param=True,
                 tc_banded_enabled: bool = False, tc_channel_bandwidth: int = 0,
                 enc_input_layer_norm: bool = False, enc_postproj_layer_norm: bool = False, enc_mod_use_layer_norm: bool = False,
                 pe_type: str = 'none', pe_dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.bandwidth = bandwidth
        self.use_cholesky_param = use_cholesky_param
        # TC-banded posterior flags
        self.tc_banded_enabled = bool(tc_banded_enabled)
        self.tc_channel_bandwidth = max(0, int(tc_channel_bandwidth)) if tc_banded_enabled else 0
        # LayerNorm flags
        self.enc_input_layer_norm = bool(enc_input_layer_norm)
        self.enc_postproj_layer_norm = bool(enc_postproj_layer_norm)
        self.enc_mod_use_layer_norm = bool(enc_mod_use_layer_norm)

        # Transformer encoder blocks for feature extraction
        self.trans_enc_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(transformer_encoder_blocks)
        ])

        # Optional input token LayerNorm (pre-transformer)
        self.input_ln = nn.LayerNorm(input_dim) if self.enc_input_layer_norm else nn.Identity()

        # Positional encodings configuration
        self.pe_type = (pe_type or 'none').lower()
        self.pe_dropout = float(pe_dropout) if pe_dropout is not None else 0.0
        
        self.positional_encoding_param = None
        self.register_buffer('positional_encoding', None, persistent=False)
        
        if self.pe_type == 'learnable':
            self.positional_encoding_param = nn.Parameter(torch.zeros(sequence_length, input_dim))
            nn.init.normal_(self.positional_encoding_param, mean=0.0, std=0.02)
        elif self.pe_type == 'sinusoidal':
            pe = self._sinusoidal_positional_encoding(sequence_length, input_dim)
            self.register_buffer('positional_encoding', pe)
            
        self.pos_drop = nn.Dropout(self.pe_dropout) if self.pe_dropout > 0.0 else nn.Identity()

        # Transformer encoder blocks for feature extraction parameters
        self.precision_proj = nn.Linear(input_dim, hidden_dim)
        # Optional post-projection LN
        self.postproj_ln = nn.LayerNorm(hidden_dim) if self.enc_postproj_layer_norm else nn.Identity()

        # Banded Precision Gaussian for the approximate posterior (global fallback)
        # Stateless-only posterior: the legacy stateful posterior implementation has been removed.
        # Always use the stateless helper which provides default per-sample banded precision params.
        # Stateless posterior helper: keep it stateless (no global defaults). Callers must provide per-sequence params.
        self.banded_posterior = BandedPrecisionGaussian_Stateless(dim=sequence_length, bandwidth=bandwidth)

        # Mean prediction network now outputs latent_dim * sequence_length
        self.mean_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sequence_length * latent_dim)
        )

        self.mean_proj_Dense =  nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden_dim),
        ) for _ in range(4)])


        self.latent_dim_out = latent_dim
        self.use_modulation = use_modulation
        self.mod_mode = mod_mode
        self.mod_pre_transformer = mod_pre_transformer
        if self.use_modulation:
            # Use adapted modulation that predicts banded precision parameters and mean
            self.mod_blocks = nn.ModuleList([
                BandedPrecisionAttentionModulationBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    T=sequence_length,
                    bandwidth=self.bandwidth,
                    use_layer_norm=self.enc_mod_use_layer_norm
                ) for _ in range(mod_num_blocks)
            ])
            if self.mod_mode == 'residual':
                self.residual_alpha = nn.Parameter(torch.tensor(residual_alpha_init, dtype=torch.get_default_dtype()))

        # Optional channel-precision head for TC-banded posterior
        if self.tc_banded_enabled and self.tc_channel_bandwidth > 0:
            # Simple MLP head from pooled sequence representation to channel precision params
            chan_out_dims = self.latent_dim + self.tc_channel_bandwidth * max(0, self.latent_dim - 1)
            self.channel_precision_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, chan_out_dims)
            )
        else:
            self.channel_precision_head = None

    def _sinusoidal_positional_encoding(self, sequence_length, hidden_dim, device=None):
        import math
        if device is None:
            device = torch.device('cpu')
        dtype = torch.get_default_dtype()
        position = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device, dtype=dtype) * (-(math.log(10000.0) / hidden_dim)))
        pe = torch.zeros(sequence_length, hidden_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _apply_pe(self, x):
        if self.pe_type == 'learnable' and self.positional_encoding_param is not None:
            x = x + self.positional_encoding_param.unsqueeze(0)
        elif self.pe_type == 'sinusoidal' and self.positional_encoding is not None:
            x = x + self.positional_encoding.unsqueeze(0)
        return self.pos_drop(x)

    def forward(self, x: torch.Tensor, patched_mask=None):
        """
        Forward pass for the enhanced encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
            patched_mask (torch.Tensor, optional): Mask for the input sequence.

        Returns:
            tuple: Mean (mu) [B,C,T] and posterior_params dict.
                   posterior_params contains keys when modulation is enabled:
                     - 'precision_diag': [B,T]
                     - 'precision_bands': [B, bandwidth, T-1]
                   If TC-banded enabled and channel head present, also:
                     - 'precision_channel_diag': [B,C]
                     - 'precision_channel_bands': [B, channel_bandwidth, C-1]
                   If modulation disabled, both are None and global banded_posterior is used downstream.
        """
        batch_size = x.shape[0]

        # Debug: Print input tensor shape
        # print(f"Debug: Input x shape: {x.shape}, expected: (batch_size, n_patches, input_dim)")

        # Run transformer blocks before modulation if configured OR if modulation disabled
        # Run transformer blocks before modulation if configured OR if modulation disabled
        if (not self.use_modulation) or (self.use_modulation and self.mod_pre_transformer):
            x = self.input_ln(x)
            x = self._apply_pe(x)
            for block in self.trans_enc_blocks:
                x = block(x)

        # Global pooling baseline (match TALONTeacher_Encoder): masked mean over patches
        if patched_mask is not None:
            # patched_mask: [B, T_patches] with 1 for valid, 0 for padded
            masked_x = x * patched_mask.unsqueeze(-1)
            denom = patched_mask.sum(dim=1, keepdim=True).clamp(min=1)
            seq_repr = masked_x.sum(dim=1) / denom
        else:
            seq_repr = x.mean(dim=1)

        h = self.precision_proj(seq_repr)
        h = self.postproj_ln(h)
        for layer in self.mean_proj_Dense:
            h = layer(h)

        mu_flat_base = self.mean_proj(h)
        mu_base = mu_flat_base.view(batch_size, self.latent_dim_out, self.sequence_length)  # [B,C,T]

        posterior_params = {'precision_diag': None, 'precision_bands': None}
        if self.use_modulation:
            if not self.mod_pre_transformer:
                x = self.input_ln(x)
                x = self._apply_pe(x)
                for block in self.trans_enc_blocks:
                    x = block(x)
            mu_mod = None
            pdiag = None
            pbands = None
            for mod in self.mod_blocks:
                pd, pb, mu_temp = mod(x, mask=patched_mask if patched_mask is not None else None)
                mu_temp = mu_temp.transpose(1, 2)  # [B,C,T]
                mu_mod = mu_temp if mu_mod is None else mu_temp
                pdiag = pd
                pbands = pb
            if self.mod_mode == 'replace':
                mu_batch = mu_mod
            else:  # residual
                mu_batch = mu_base + self.residual_alpha * mu_mod
                # mu_batch =  self.residual_alpha * mu_mod

            posterior_params = {'precision_diag': pdiag, 'precision_bands': pbands}
        else:
            mu_batch = mu_base
            # Stateless: do NOT read any global default buffers from banded_posterior here.
            # Downstream code must provide per-sequence posterior parameters when required.
            posterior_params = {'precision_diag': None, 'precision_bands': None}

        # Optionally add channel precision predictions for TC-banded posterior
        if self.tc_banded_enabled and self.channel_precision_head is not None:
            # Predict concatenated [C] diag and [bw*(C-1)] bands, then reshape
            chan_vec = self.channel_precision_head(h)  # [B, C + bw*(C-1)]
            C = self.latent_dim_out
            bw_c = self.tc_channel_bandwidth
            chan_diag = chan_vec[:, :C]
            chan_bands_flat = chan_vec[:, C:]
            if bw_c > 0 and C > 1:
                chan_bands = chan_bands_flat.view(batch_size, bw_c, max(0, C - 1))
            else:
                chan_bands = torch.zeros(batch_size, 0, max(0, C - 1), device=chan_vec.device, dtype=chan_vec.dtype)
            posterior_params['precision_channel_diag'] = chan_diag
            posterior_params['precision_channel_bands'] = chan_bands

        return mu_batch, posterior_params


class TALONStudent(nn.Module):
    """
    Enhanced Time-Series Pair Conditioning Encoder (TALONStudent).

    This is a conditioning model that learns an approximate posterior to match the pre-trained
    TALONTeacher's banded posterior distribution. It contains:
    - A conditioning encoder that learns to approximate the posterior
    - A frozen decoder from the pre-trained TALONTeacher
    - The pre-trained banded posterior as the prior

    Args:
        pretrained_tspvae (TALONTeacher): Pre-trained TALONTeacher model.
        conditioning_input_dim (int): Number of conditioning channels in input.
        enc_hidden_dim (int, optional): Hidden dim for the conditioning encoder (inherited if None).
        dec_hidden_dim (int, optional): Hidden dim for the decoder (inherited if None).
        posterior_tc_banded (bool, optional): Enable TC-banded structure for the posterior.
        bandwidth (int, optional): Bandwidth for the banded precision matrix.
        tc_channel_bandwidth (int, optional): Channel bandwidth for TC-banded structure.
        use_cholesky_param (bool, optional): Use Cholesky parameterization for precision.
        init_precision_diag (float, optional): Initial value for diagonal precision.
        init_precision_offdiag (float, optional): Initial value for off-diagonal precision.
        posterior_jitter (float, optional): Jitter for posterior stability.
        posterior_jitter_max (float, optional): Maximum jitter for posterior stability.
        encoder_kwargs (dict, optional): Additional keyword arguments for the encoder.
        patch_embedder (nn.Module, optional): Pretrained patch embedder.
        freeze_pretrained_components (bool): Whether to freeze the pre-trained components.
        kl_direction (str): "reverse" for KL(q(z|x)||q(z|y)) or "forward" for KL(q(z|y)||q(z|x)).
    """
    def __init__(self, pretrained_tspvae,
                 conditioning_input_dim: int = 2,
                 enc_hidden_dim: int = None,
                 dec_hidden_dim: int = None,
                 posterior_tc_banded: bool = None,
                 bandwidth: int = None,
                 tc_channel_bandwidth: int = None,
                 use_cholesky_param: bool = None,
                 init_precision_diag: float = None,
                 init_precision_offdiag: float = None,
                 posterior_jitter: float = 1e-8,
                 posterior_jitter_max: float = 1e-6,
                 encoder_kwargs: dict = None,
                 patch_embedder=None,
                 freeze_pretrained_components: bool = True,
                 kl_direction: str = "reverse",
                 compute_mi: bool = False,
                 mi_loss_weight: float = 0.0,
                 mi_learn_temperature: bool = True,
                 conditioning_patch_length: int = None):
        """
        Reworked initializer: explicit arguments replace the previous `args`-based configuration.
        Values default to sensible constants or are inherited from the provided pretrained_tspvae
        when not set explicitly.
        """
        super().__init__()
        self.kl_direction = str(kl_direction).strip().lower()
        if self.kl_direction not in {"reverse", "forward"}:
            raise ValueError(f"Unsupported kl_direction '{kl_direction}'. Use 'reverse' or 'forward'.")
        # Use explicit pretrained model reference
        self.pretrained_model = pretrained_tspvae
        # Latent/channel/time configuration strictly from pretrained
        self.latent_dim = int(getattr(pretrained_tspvae, 'latent_dim'))
        self.sequence_length = int(getattr(pretrained_tspvae, 'sequence_length'))
        self.patch_length = int(getattr(pretrained_tspvae, 'patch_length'))
        self.conditioning_patch_length = conditioning_patch_length if conditioning_patch_length is not None else self.patch_length
        # Target channels (decoder outputs) from pretrained model
        self.n_channels = int(getattr(pretrained_tspvae, 'n_channels'))
        # Conditioning channels: now passed explicitly
        self.cond_n_channels = int(conditioning_input_dim)

        # Encoder/decoder hidden dims: prefer explicit, else inherit from pretrained when possible
        if enc_hidden_dim is None:
            try:
                enc_hidden_dim = int(getattr(pretrained_tspvae, 'encoder').hidden_dim)
            except Exception:
                enc_hidden_dim = 128
        self.enc_hidden_dim = int(enc_hidden_dim)
        if dec_hidden_dim is None:
            try:
                dec_hidden_dim = int(getattr(pretrained_tspvae, 'decoder').hidden_dim)
            except Exception:
                dec_hidden_dim = 128
        self.dec_hidden_dim = int(dec_hidden_dim)

        # Store embedder configuration
        self.embedder = patch_embedder
        self.use_embedder = patch_embedder is not None

        # Per-channel reconstruction loss weights must match pretrained
        if hasattr(pretrained_tspvae, 'channel_weights') and pretrained_tspvae.channel_weights is not None:
            self.channel_weights = pretrained_tspvae.channel_weights.detach().clone()
        else:
            self.channel_weights = torch.ones(self.n_channels, dtype=torch.float)
        print(f"Channel reconstruction weights (inherited): {self.channel_weights.tolist()}")

        # Keep reference to pretrained components
        self.pretrained_encoder = pretrained_tspvae.encoder
        self.pretrained_decoder = pretrained_tspvae.decoder
        self.pretrained_gp_prior = pretrained_tspvae.gp_prior

        # Posterior structure hyperparameters: prefer explicit arguments, else inherit from pretrained encoder
        if posterior_tc_banded is None:
            posterior_tc_banded = bool(getattr(self.pretrained_encoder, 'tc_banded_enabled', False))
        self.posterior_tc_banded = bool(posterior_tc_banded)

        if bandwidth is None:
            bandwidth = int(getattr(self.pretrained_encoder, 'bandwidth', 3))
        self.tc_time_bandwidth = int(bandwidth)

        if tc_channel_bandwidth is None:
            tc_channel_bandwidth = int(getattr(self.pretrained_encoder, 'tc_channel_bandwidth', 0))
        self.tc_channel_bandwidth = int(tc_channel_bandwidth)

        inherited_bandwidth = int(self.tc_time_bandwidth if self.tc_time_bandwidth is not None else getattr(self.pretrained_encoder, 'bandwidth', 3))

        # Cholesky flag: allow explicit override otherwise inherit from pretrained encoder
        if use_cholesky_param is None:
            use_cholesky_param = bool(getattr(self.pretrained_encoder, 'use_cholesky_param', True))
        inherited_use_cholesky = bool(use_cholesky_param)

        # Prefer init_precision from explicit args; otherwise use safe constants.
        if init_precision_diag is None:
            init_precision_diag = 1.0
        if init_precision_offdiag is None:
            init_precision_offdiag = 0.1

        # Numerical jitter for Cholesky solves in this model
        self.posterior_jitter = float(posterior_jitter)
        self.posterior_jitter_max = float(posterior_jitter_max)

        # Freeze pre-trained components if requested
        if freeze_pretrained_components:
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            for param in self.pretrained_decoder.parameters():
                param.requires_grad = False
            for param in self.pretrained_gp_prior.parameters():
                param.requires_grad = False

            # Set to eval mode
            self.pretrained_encoder.eval()
            self.pretrained_decoder.eval()
            self.pretrained_gp_prior.eval()

            print("Pre-trained components frozen and set to eval mode")
            
        self.compute_mi = bool(compute_mi)
        self.mi_loss_weight = float(mi_loss_weight)
        if self.compute_mi:
            self.mi_estimator = MIEstimator(learn_temperature=mi_learn_temperature)
        else:
            self.mi_estimator = None

        # Determine input dimensions based on embedder usage
        if self.use_embedder:
            if freeze_pretrained_components and hasattr(self.embedder, 'parameters'):
                for param in self.embedder.parameters():
                    param.requires_grad = False
                self.embedder.eval()

            encoder_input_dim = self.embedder.config.d_model * self.cond_n_channels
            n_patches = self.sequence_length // self.patch_length
        else:
            # Conditioning encoder expects (batch, n_patches, cond_channels * conditioning_patch_length)
            encoder_input_dim = self.cond_n_channels * self.conditioning_patch_length
            n_patches = int(self.sequence_length // self.patch_length)

        # Compute encoder-local options from encoder_kwargs (all encoder-specific options must be provided there)
        enc_kwargs = encoder_kwargs or {}
        transformer_encoder_blocks = int(enc_kwargs.get('transformer_encoder_blocks', 2))
        n_heads = int(enc_kwargs.get('n_heads', 4))
        dim_feedforward = int(enc_kwargs.get('dim_feedforward', 512))
        dropout = float(enc_kwargs.get('dropout', 0.1))
        enc_mod_enabled = bool(enc_kwargs.get('use_modulation', True))
        enc_mod_num_blocks = int(enc_kwargs.get('mod_num_blocks', 1))
        enc_mod_mode = enc_kwargs.get('mod_mode', 'residual')
        enc_mod_pre_transformer = bool(enc_kwargs.get('mod_pre_transformer', True))
        enc_mod_alpha_init = float(enc_kwargs.get('residual_alpha_init', 1.0))
        enc_input_layer_norm = bool(enc_kwargs.get('enc_input_layer_norm', False))
        enc_postproj_layer_norm = bool(enc_kwargs.get('enc_postproj_layer_norm', False))
        enc_mod_use_layer_norm = bool(enc_kwargs.get('enc_mod_use_layer_norm', False))

        print(f"Encoder input dimension calculated: {encoder_input_dim} "
              f"(embedder: {self.use_embedder}, cond_channels: {self.cond_n_channels}, "
              f"patch_length: {self.patch_length})")

        # Build encoder init kwargs from encoder_kwargs (explicit encoder params only)
        encoder_init = {
            'input_dim': encoder_input_dim,
            'hidden_dim': self.enc_hidden_dim,
            'latent_dim': self.latent_dim,
            'sequence_length': n_patches,
            'bandwidth': int(enc_kwargs.get('bandwidth', inherited_bandwidth)),
            'transformer_encoder_blocks': transformer_encoder_blocks,
            'n_heads': n_heads,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'init_precision_diag': float(enc_kwargs.get('init_precision_diag', init_precision_diag)),
            'init_precision_offdiag': float(enc_kwargs.get('init_precision_offdiag', init_precision_offdiag)),
            'use_modulation': enc_mod_enabled,
            'mod_num_blocks': enc_mod_num_blocks,
            'mod_mode': enc_mod_mode,
            'mod_pre_transformer': enc_mod_pre_transformer,
            'residual_alpha_init': enc_mod_alpha_init,
            'use_cholesky_param': bool(enc_kwargs.get('use_cholesky_param', inherited_use_cholesky)),
            'tc_banded_enabled': bool(enc_kwargs.get('tc_banded_enabled', self.posterior_tc_banded)),
            'tc_channel_bandwidth': int(enc_kwargs.get('tc_channel_bandwidth', self.tc_channel_bandwidth)),
            'enc_input_layer_norm': enc_input_layer_norm,
            'enc_postproj_layer_norm': enc_postproj_layer_norm,
            'enc_mod_use_layer_norm': enc_mod_use_layer_norm,
        }
        # Merge any remaining encoder_kwargs (caller overrides) shallowly
        if enc_kwargs:
            encoder_init.update({k: v for k, v in enc_kwargs.items() if k not in encoder_init or True})

        # Create the conditioning encoder (this is what we'll train)
        self.conditioning_encoder = TALONStudent_Encoder(**encoder_init)
        self.patch_embedding_layer = nn.Conv1d(in_channels=conditioning_input_dim,
                                         out_channels=self.conditioning_patch_length*conditioning_input_dim,
                                         kernel_size=self.conditioning_patch_length,
                                         stride=self.conditioning_patch_length)
        print("TALONStudent initialized with conditioning encoder (inherited dims & hyperparams)")

    def _prepare_input(self, x, time_mask: torch.Tensor = None):
        """Prepare input data (patchify and embed if needed). Also builds a patch-level mask when time_mask is provided.
        Returns (x_processed, patched_mask).
        """
        # Optimize input shape normalization - avoid unnecessary transposes
        if x.dim() == 3:
            b, d1, d2 = x.shape
            # More efficient shape detection and transpose logic
            needs_transpose = (d1 == self.cond_n_channels and d2 != self.cond_n_channels) or (d1 < d2 and d2 != self.cond_n_channels)
            if needs_transpose:
                x = x.transpose(1, 2)

        patched_mask = None
        if self.use_embedder:
            embedded = self.embedder(x)
            x_processed = embedded.prediction_outputs if hasattr(embedded, 'prediction_outputs') else embedded
            batch_size, n_patches, n_channels, d_model = x_processed.shape

            # Validate channel dimensions match expectations
            if n_channels != self.cond_n_channels:
                raise ValueError(f"Embedder output channels ({n_channels}) don't match expected conditioning channels ({self.cond_n_channels}). "
                               f"Check your embedder configuration or conditioning input data.")

            # More efficient reshape using view when possible
            x_processed = x_processed.view(batch_size, n_patches, self.cond_n_channels * d_model)



            # Optimize patch mask computation
            if time_mask is not None:
                L, P = self.patch_length, n_patches
                # Use view instead of reshape for better performance
                tm = time_mask[:, :P * L].view(batch_size, P, L)
                patched_mask = (tm.float().mean(dim=2, keepdim=False) > 0.5).float()
        else:
            batch_size, seq_len, n_channels_in = x.shape

            # Validate channel dimensions match expectations
            if n_channels_in != self.cond_n_channels:
                raise ValueError(f"Input data channels ({n_channels_in}) don't match expected conditioning channels ({self.cond_n_channels}). "
                               f"Expected conditioning input with {self.cond_n_channels} channels but got {n_channels_in}.")

            # Optimize patchification - combine operations
            n_patches = seq_len // self.conditioning_patch_length
            truncated_len = n_patches * self.conditioning_patch_length
            # Direct view operation instead of multiple reshapes

            # x_processed = x[:, :truncated_len, :].reshape(batch_size, n_patches, self.conditioning_patch_length * n_channels_in)
            x_processed = self.patch_embedding_layer(x.transpose(1, 2)).transpose(1, 2)
#             print(f"DEBUG _prepare_input: x_processed shape = {x_processed.shape}, self.conditioning_patch_length = {self.conditioning_patch_length}")
            if time_mask is not None:
                tm = time_mask[:, :truncated_len].view(batch_size, n_patches, self.conditioning_patch_length)
                patched_mask = (tm.float().mean(dim=2, keepdim=False) > 0.5).float()

        return x_processed, patched_mask

    def _prepare_y_prior(self, y: torch.Tensor, time_mask: torch.Tensor = None):
        """Patchify y_target using the pretrained model's patchify pipeline to produce prior inputs.
        Returns (y_patches, patched_mask) consistent with the pretrained encoder expectations.
        """
        if y.dim() != 3:
            raise ValueError("y_target must be a 3D tensor [B, T, C] or [B, C, T]")
        b, d1, d2 = y.shape
        if d1 == self.n_channels and d2 != self.n_channels:
            y_bct = y  # [B,C,T]
        elif d2 == self.n_channels and d1 != self.n_channels:
            y_bct = y.transpose(1, 2).contiguous()  # [B,C,T]
        else:
            if d1 < d2:
                y_bct = y.transpose(1, 2).contiguous()
            else:
                y_bct = y
        irrelevant_mask = None
        if time_mask is not None:
            # time_mask [B,T] -> [B,T,C]
            irrelevant_mask = time_mask.unsqueeze(-1).expand(-1, -1, self.n_channels).to(y_bct.device)
        if hasattr(self.pretrained_model, 'patchify_data'):
            y_patches, y_mask = self.pretrained_model.patchify_data(y_bct, irrelevant_mask=irrelevant_mask)
        else:
            y_time_first = y_bct.transpose(1, 2)
            y_patches, y_mask = self._prepare_input(y_time_first, time_mask=time_mask)
        return y_patches, y_mask

    def _build_time_mask_from_lengths(self, seq_lengths: torch.Tensor, T: int, device, dtype=torch.float):
        """Build [B,T] mask with 1 for valid timesteps and 0 for padded, using provided lengths."""
        if seq_lengths is None:
            return torch.ones(1, T, device=device, dtype=dtype)  # caller should expand as needed
        if not torch.is_tensor(seq_lengths):
            seq_lengths = torch.as_tensor(seq_lengths, device=device)
        seq_lengths = seq_lengths.long().clamp(min=0, max=T)
        B = seq_lengths.shape[0]
        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        mask = (arange < seq_lengths.unsqueeze(1)).to(dtype)
        return mask

    def forward(self, x_condition: torch.Tensor, y_patches: torch.Tensor, mode="train", patched_mask=None, prior_override=None,y_target=None, time_mask_full=None, ood_threshold=None):
        """
        Forward pass for the conditioning model.

        Args:
            x_condition (torch.Tensor): Conditioning input (partial trajectory).
            y_target (torch.Tensor): Target output (full trajectory).
            y_patches (torch.Tensor): Patchified target input for prior encoder.
            mode (str): Training mode - "train", "test", or "testRandom".
            prior_override (tuple, optional): Optional tuple (mu_prior, prior_params) providing precomputed
                                              target prior parameters per sample; when provided, skips
                                              the frozen encoder call.
            time_mask_full (torch.Tensor, optional): Full time mask [B,T] for reconstruction loss masking.
            patched_mask (torch.Tensor, optional): Patch-level mask [B, n_patches] for conditioning and prior encoders.
            ood_threshold (float, optional): Dynamic OOD channel masking threshold.

        Returns:
            ModelOutput: Contains reconstruction outputs and loss components.
        """
        # Check if inputs are passed raw (like in run_spatial_benchmark.py)
        is_raw = False
        if x_condition.dim() == 3:
            if x_condition.shape[2] == self.cond_n_channels:
                is_raw = True

        if is_raw:
            if y_target is None:
                y_target = y_patches
            if time_mask_full is None:
                time_mask_full = torch.ones(x_condition.shape[0], self.sequence_length, device=x_condition.device)
            x_processed, x_patched_mask = self._prepare_input(x_condition, time_mask=time_mask_full)
            x_condition = x_processed
            patched_mask = x_patched_mask
            if prior_override is None and y_patches is not None:
                y_patches_prep, _ = self._prepare_y_prior(y_patches, time_mask=time_mask_full)
                y_patches = y_patches_prep

        # patched_mask should be a tensor mask or None; leave as None if not provided
        if patched_mask is None:
            patched_mask = None

        # Initialize prior outputs to satisfy static analyzers
        mu_prior = None
        prior_params = None

        # Get conditioning posterior from our encoder (use x's mask)
        mu_condition, posterior_params_condition = self.conditioning_encoder(x_condition, patched_mask=patched_mask)

        # Get prior (target posterior parameters) from pre-trained encoder (frozen) or override
        # Get prior (target posterior parameters) from pre-trained encoder (frozen) or override
        if prior_override is not None:
            mu_prior, prior_params = prior_override
        else:
            with torch.no_grad():
                mu_prior, prior_params = self.pretrained_encoder(y_patches, patched_mask=patched_mask)

        # Compute KL divergence, masking out padded patches
        if posterior_params_condition['precision_diag'] is not None:
            if self.kl_direction == "forward":
                # KL(q(z|y) || q(z|x))
                kl_div = self._compute_banded_kl_divergence(
                    mu_prior, prior_params,
                    mu_condition, posterior_params_condition,
                    patch_mask=patched_mask
                )
            else:
                # KL(q(z|x) || q(z|y))
                kl_div = self._compute_banded_kl_divergence(
                    mu_condition, posterior_params_condition,
                    mu_prior, prior_params,
                    patch_mask=patched_mask
                )
        else:
            raise ValueError("missing posterior params from conditioning model; ensure modulation is valid.")

        # Sample from posteriors based on mode
        if mode == "train" or mode == "testRandom":
            z_condition, z_target = self._sample_correlated_posteriors(
                mu_condition, posterior_params_condition,
                mu_prior, prior_params
            )
        elif mode == "test":
            z_condition, z_target = mu_condition, mu_prior
        else:
            raise ValueError("mode should be either 'train', 'testRandom', or 'test'")

        # Apply context encoder to latent variables before decoding
        # This is a critical step that was missing - the pretrained VAE uses context_encoder
        # to transform latent variables before passing them to the decoder
        z_target_transposed = z_target.transpose(1, 2)  # [B, T, C]
        z_condition_transposed = z_condition.transpose(1, 2)  # [B, T, C]

        # Apply the pretrained context encoder to both latent variables
        with torch.no_grad():
            context_features_target = self.pretrained_model.context_encoder(z_target_transposed)
            y_hat_target = self.pretrained_decoder(context=context_features_target, patched_mask=patched_mask)

        context_features_condition = self.pretrained_model.context_encoder(z_condition_transposed)
        y_hat_condition = self.pretrained_decoder(context=context_features_condition, patched_mask=patched_mask)

        if y_hat_target.dim() == 3:
            y_hat_target = y_hat_target.permute(1, 0, 2).contiguous()
        if y_hat_condition.dim() == 3:
            y_hat_condition = y_hat_condition.permute(1, 0, 2).contiguous()

        if self.use_embedder:
            y_hat_target_unpatch = self._unpatchify_embedder_output(y_hat_target)
            y_hat_condition_unpatch = self._unpatchify_embedder_output(y_hat_condition)
        else:
            y_hat_target_unpatch = self._unpatchify_direct(y_hat_target)
            y_hat_condition_unpatch = self._unpatchify_direct(y_hat_condition)

        if y_target is not None:
            if time_mask_full is None:
                raise RuntimeError(
                    "time_mask_full must be provided when y_target is given for reconstruction loss computation.")            # Ensure time_mask_full shape matches y_target
            # Compute reconstruction losses (mask padded timesteps) if y_target provided
            reconstruction_loss_condition = self._compute_reconstruction_loss(y_target, y_hat_condition_unpatch, time_mask_full, ood_threshold=ood_threshold)
            reconstruction_loss_target = self._compute_reconstruction_loss(y_target, y_hat_target_unpatch, time_mask_full, ood_threshold=ood_threshold)
        else:
            reconstruction_loss_condition = None
            reconstruction_loss_target = None

        # Apply sigmoid to discrete channels for the returned predictions
        discrete_mask = self.pretrained_model.discrete_mask.to(x_condition.device)
        if discrete_mask.any():
            disc_mask = discrete_mask.view(1, 1, -1) # shape [1, 1, C] since y_hat_condition_unpatch is [B, T, C]
            y_hat_condition_unpatch = torch.where(disc_mask, torch.sigmoid(y_hat_condition_unpatch), y_hat_condition_unpatch)
            y_hat_target_unpatch = torch.where(disc_mask, torch.sigmoid(y_hat_target_unpatch), y_hat_target_unpatch)

        mi_loss = None
        mi_estimate = None
        if self.compute_mi and posterior_params_condition is not None and prior_params is not None:
            # We need to assemble L_t and L_c for both
            def get_variances(mu, params):
                params = {k: v.float() if torch.is_tensor(v) else v for k, v in params.items()}
                P_t = assemble_precision_from_bands_fn(params['precision_diag'], params['precision_bands'], self.tc_time_bandwidth)
                L_t = robust_cholesky_fn(P_t, jitter_start=self.posterior_jitter, jitter_max=self.posterior_jitter_max)
                if self.posterior_tc_banded and 'precision_channel_diag' in params:
                    P_c = assemble_precision_from_bands_fn(params['precision_channel_diag'], params['precision_channel_bands'], self.tc_channel_bandwidth)
                    L_c = robust_cholesky_fn(P_c, jitter_start=self.posterior_jitter, jitter_max=self.posterior_jitter_max)
                else:
                    L_c = None
                return extract_diagonal_variances(L_t, L_c)
                
            var_phi = get_variances(mu_condition, posterior_params_condition)
            var_psi = get_variances(mu_prior, prior_params)
            
            mu_phi_flat = mu_condition.view(mu_condition.shape[0], -1)
            mu_psi_flat = mu_prior.view(mu_prior.shape[0], -1)
            
            nce_loss, mi_est = self.mi_estimator(mu_phi_flat, var_phi, mu_psi_flat, var_psi)
            mi_loss = self.mi_loss_weight * nce_loss
            mi_estimate = mi_est

        return ModelOutput(
            y_hat_condition=y_hat_condition_unpatch,
            y_hat_target=y_hat_target_unpatch,
            mu_condition=mu_condition,
            mu_target=mu_prior,
            z_condition=z_condition,
            z_target=z_target,
            KL_div=kl_div,
            reconstruction_loss_condition=reconstruction_loss_condition,
            reconstruction_loss_target=reconstruction_loss_target,
            posterior_params_condition=posterior_params_condition,
            posterior_params_target=prior_params,
            mi_loss=mi_loss,
            mi_estimate=mi_estimate
        )

    def _sanitize_vector(self, v, default=0.0, min_val=None, max_val=None):
        if v is None:
            return None
        v = torch.nan_to_num(v, nan=default, posinf=default, neginf=default)
        if min_val is not None or max_val is not None:
            v = torch.clamp(v, min=min_val if min_val is not None else -float('inf'), max=max_val if max_val is not None else float('inf'))
        return v

    def _compute_banded_kl_divergence(self, mu_q, params_q, mu_p, params_p, patch_mask=None):
        """
        Compute KL(q||p) between two banded-precision Gaussians with Kronecker structure.
        Only valid (unpadded) time steps (patches) contribute to KL if patch_mask is provided.
        """
        B, C, T = mu_q.shape
        device = mu_q.device
        dtype = torch.float32  # Force float32 for numerical stability (especially with AMP Cholesky/Solves)
        mu_q = self._sanitize_vector(mu_q.float(), default=0.0)
        mu_p = self._sanitize_vector(mu_p.float(), default=0.0)

        # Cast precision parameters
        params_q = {k: v.float() if torch.is_tensor(v) else v for k, v in params_q.items()}
        params_p = {k: v.float() if torch.is_tensor(v) else v for k, v in params_p.items()}

        Qt_q = assemble_precision_from_bands_fn(
            params_q['precision_diag'], params_q['precision_bands'], bandwidth=getattr(self.conditioning_encoder, 'bandwidth', T - 1)
        )  # [B,T,T]
        Qt_p = assemble_precision_from_bands_fn(
            params_p['precision_diag'], params_p['precision_bands'], bandwidth=getattr(self.pretrained_encoder, 'bandwidth', T - 1)
        )  # [B,T,T]

        # Optional channel precisions (if both provided); else identity
        has_qc = params_q.get('precision_channel_diag', None) is not None and params_q.get('precision_channel_bands', None) is not None
        has_pc = params_p.get('precision_channel_diag', None) is not None and params_p.get('precision_channel_bands', None) is not None
        if has_qc and has_pc and self.posterior_tc_banded and self.tc_channel_bandwidth > 0:
            Qc_q = assemble_precision_from_bands_fn(
                params_q['precision_channel_diag'], params_q['precision_channel_bands'], bandwidth=getattr(self.conditioning_encoder, 'tc_channel_bandwidth', max(0, C - 1))
            )  # [B,C,C]
            Qc_p = assemble_precision_from_bands_fn(
                params_p['precision_channel_diag'], params_p['precision_channel_bands'], bandwidth=getattr(self.pretrained_encoder, 'tc_channel_bandwidth', max(0, C - 1))
            )  # [B,C,C]
        else:
            Qc_q = None
            Qc_p = None

        jitter_start = float(getattr(self, 'posterior_jitter', 1e-8))
        jitter_max = float(getattr(self, 'posterior_jitter_max', 1e-6))

        # Check if patch_mask is uniform/same for all batches or None
        is_uniform = True
        if patch_mask is not None:
            is_uniform = bool(torch.all(patch_mask == patch_mask[0:1]))

        if is_uniform:
            # get valid indices from first row
            if patch_mask is not None:
                valid_idx = (patch_mask[0] > 0.5).nonzero(as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(T, device=device)
            
            if valid_idx.numel() == 0:
                return torch.zeros(B, device=device, dtype=dtype)

            # Slice and vectorize
            mu_q_v = mu_q[:, :, valid_idx]  # [B, C, T_valid]
            mu_p_v = mu_p[:, :, valid_idx]  # [B, C, T_valid]
            Qtq = Qt_q[:, valid_idx, :][:, :, valid_idx]  # [B, T_valid, T_valid]
            Qtp = Qt_p[:, valid_idx, :][:, :, valid_idx]  # [B, T_valid, T_valid]

            T_v = len(valid_idx)
            I_Tv = torch.eye(T_v, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            
            Ltq = robust_cholesky_batched(Qtq, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
            Ltp = robust_cholesky_batched(Qtp, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
            Stq = torch.cholesky_solve(I_Tv, Ltq)  # [B, T_valid, T_valid]
            
            logdet_Qtq = 2.0 * torch.sum(torch.log(torch.diagonal(Ltq, dim1=-2, dim2=-1)), dim=-1) # [B]
            logdet_Qtp = 2.0 * torch.sum(torch.log(torch.diagonal(Ltp, dim1=-2, dim2=-1)), dim=-1) # [B]

            if Qc_q is not None and Qc_p is not None:
                Qcq = Qc_q  # [B, C, C]
                Qcp = Qc_p  # [B, C, C]
                I_C = torch.eye(C, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
                Lcq = robust_cholesky_batched(Qcq, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
                Lcp = robust_cholesky_batched(Qcp, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
                Scq = torch.cholesky_solve(I_C, Lcq)  # [B, C, C]
                
                logdet_Qcq = 2.0 * torch.sum(torch.log(torch.diagonal(Lcq, dim1=-2, dim2=-1)), dim=-1) # [B]
                logdet_Qcp = 2.0 * torch.sum(torch.log(torch.diagonal(Lcp, dim1=-2, dim2=-1)), dim=-1) # [B]
                
                # Traces of matrices: trace(A @ B) is the sum of diagonal of (A @ B)
                trace_t = torch.diagonal(Qtp @ Stq, dim1=-2, dim2=-1).sum(dim=-1) # [B]
                trace_c = torch.diagonal(Qcp @ Scq, dim1=-2, dim2=-1).sum(dim=-1) # [B]
                trace_term = trace_t * trace_c  # [B]
                
                logdet_Qp = C * logdet_Qtp + T_v * logdet_Qcp # [B]
                logdet_Qq = C * logdet_Qtq + T_v * logdet_Qcq # [B]
                logdet_ratio = -(logdet_Qp - logdet_Qq) # [B]
                
                delta = mu_p_v - mu_q_v # [B, C, T_valid]
                temp_t = delta @ Qtp # [B, C, T_valid]
                temp = Qcp @ temp_t # [B, C, T_valid]
                quad = torch.sum(delta * temp, dim=[-2, -1]) # [B]
                
                n_dim = C * T_v
                kl_vec = 0.5 * (logdet_ratio - n_dim + trace_term + quad)
                return kl_vec
            else:
                raise(RuntimeError("Qc_q or Qc_p are None"))

        # Fallback loop-based implementation for non-uniform masks
        kl_vals = []
        for b in range(B):
            # Masking: get valid indices for this sample
            if patch_mask is not None:
                valid_idx = (patch_mask[b] > 0.5).nonzero(as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(T, device=device)
            if valid_idx.numel() == 0:
                kl_vals.append(torch.tensor(0.0, device=device, dtype=dtype))
                continue
            # Mask mu and precision matrices
            mu_q_b = mu_q[b][:, valid_idx]  # [C, T_valid]
            mu_p_b = mu_p[b][:, valid_idx]  # [C, T_valid]
            Qtq = Qt_q[b][valid_idx][:, valid_idx]  # [T_valid, T_valid]
            Qtp = Qt_p[b][valid_idx][:, valid_idx]  # [T_valid, T_valid]
            I_Tv = torch.eye(len(valid_idx), device=device, dtype=dtype)
            # Cholesky and inverses for time
            Ltq = robust_cholesky_fn(Qtq, jitter_start=jitter_start, jitter_max=jitter_max)
            Ltp = robust_cholesky_fn(Qtp, jitter_start=jitter_start, jitter_max=jitter_max)
            Stq = torch.cholesky_solve(I_Tv, Ltq)  # Σ_tq
            logdet_Qtq = 2.0 * torch.sum(torch.log(torch.diag(Ltq)))
            logdet_Qtp = 2.0 * torch.sum(torch.log(torch.diag(Ltp)))

            if Qc_q is not None and Qc_p is not None:
                Qcq = Qc_q[b]  # Channel precision matrix for q, shape [C, C]
                Qcp = Qc_p[b]  # Channel precision matrix for p, shape [C, C]
                I_C = torch.eye(C, device=device, dtype=dtype)
                Lcq = robust_cholesky_fn(Qcq, jitter_start=jitter_start, jitter_max=jitter_max)  # Cholesky factor of Qcq
                Lcp = robust_cholesky_fn(Qcp, jitter_start=jitter_start, jitter_max=jitter_max)  # Cholesky factor of Qcp
                Scq = torch.cholesky_solve(I_C, Lcq)  # Channel covariance matrix for q, Σ_cq = Qcq^{-1}
                logdet_Qcq = 2.0 * torch.sum(torch.log(torch.diag(Lcq)))  # Log determinant of Qcq
                logdet_Qcp = 2.0 * torch.sum(torch.log(torch.diag(Lcp)))  # Log determinant of Qcp
                # Trace term: tr((Q_t_p ⊗ Q_c_p) (Σ_t_q ⊗ Σ_c_q)) = tr(Q_t_p Σ_t_q) tr(Q_c_p Σ_c_q)
                trace_term = torch.trace(Qtp @ Stq) * torch.trace(Qcp @ Scq)
                # Log|Σ_p|/|Σ_q| = -(log|Q_p| - log|Q_q|)
                logdet_Qp = C * logdet_Qtp + len(valid_idx) * logdet_Qcp
                logdet_Qq = C * logdet_Qtq + len(valid_idx) * logdet_Qcq
                logdet_ratio = -(logdet_Qp - logdet_Qq)
                # Quad term: vec(Δ)^T (Q_t_p ⊗ Q_c_p) vec(Δ)
                delta = (mu_p_b - mu_q_b)  # [C,T_valid]
                # Apply over time then channels: first time
                temp_t = delta @ Qtp  # [C,T_valid]
                # Then channels
                temp = Qcp @ temp_t  # [C,T_valid]
                quad = torch.sum(delta * temp)
                n_dim = C * len(valid_idx)
                kl_b = 0.5 * (logdet_ratio - n_dim + trace_term + quad)
            else:
                raise(RuntimeError("Qc_q or Qc_p are None"))

            kl_vals.append(kl_b)
        kl_vec = torch.stack(kl_vals)
        return kl_vec

    def _sample_correlated_posteriors(self, mu_condition, params_condition, mu_target, params_target):
        """
        Sample correlated latent variables using shared noise with per-batch time/channel precisions.
        Uses Cholesky solves on precision factors, matching TALONTeacher sampling semantics.
        """
        B, C, T = mu_condition.shape
        device = mu_condition.device
        dtype = torch.float32  # Force float32 for numerical stability (especially with AMP Cholesky/Solves)
        mu_condition = self._sanitize_vector(mu_condition.float(), default=0.0)
        mu_target = self._sanitize_vector(mu_target.float(), default=0.0)

        # Cast precision parameters
        params_condition = {k: v.float() if torch.is_tensor(v) else v for k, v in params_condition.items()}
        params_target = {k: v.float() if torch.is_tensor(v) else v for k, v in params_target.items()}

        # Build per-batch time precisions
        Qt_c = assemble_precision_from_bands_fn(
            params_condition['precision_diag'], params_condition['precision_bands'], bandwidth=getattr(self.conditioning_encoder, 'bandwidth', T - 1)
        )  # [B,T,T]
        Qt_t = assemble_precision_from_bands_fn(
            params_target['precision_diag'], params_target['precision_bands'], bandwidth=getattr(self.pretrained_encoder, 'bandwidth', T - 1)
        )  # [B,T,T]

        # Optional channel precisions
        has_qc = params_condition.get('precision_channel_diag', None) is not None and params_condition.get('precision_channel_bands', None) is not None
        has_pc = params_target.get('precision_channel_diag', None) is not None and params_target.get('precision_channel_bands', None) is not None
        use_chan = has_qc and has_pc and self.posterior_tc_banded and self.tc_channel_bandwidth > 0
        if use_chan:
            Qc_c = assemble_precision_from_bands_fn(
                params_condition['precision_channel_diag'], params_condition['precision_channel_bands'], bandwidth=getattr(self.conditioning_encoder, 'tc_channel_bandwidth', max(0, C - 1))
            )  # [B,C,C]
            Qc_t = assemble_precision_from_bands_fn(
                params_target['precision_channel_diag'], params_target['precision_channel_bands'], bandwidth=getattr(self.pretrained_encoder, 'tc_channel_bandwidth', max(0, C - 1))
            )  # [B,C,C]
        else:
            Qc_c = None
            Qc_t = None

        jitter_start = float(getattr(self, 'posterior_jitter', 1e-8))
        
        # Vectorized sampling
        eps = torch.randn(B, C, T, device=device, dtype=dtype)
        
        Ltc = robust_cholesky_batched(Qt_c, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
        Ltt = robust_cholesky_batched(Qt_t, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
        
        if use_chan:
            Lcc = robust_cholesky_batched(Qc_c, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
            Lct = robust_cholesky_batched(Qc_t, jitter_start=max(1e-5, jitter_start), jitter_max=1e-2)
            
            y_c = torch.linalg.solve_triangular(Lcc.transpose(-1, -2), eps, upper=True)
            y_c_T = y_c.transpose(1, 2)
            y_t_T = torch.linalg.solve_triangular(Ltc.transpose(-1, -2), y_c_T, upper=True)
            y_t = y_t_T.transpose(1, 2)
            z_condition = mu_condition + y_t
            
            y_c2 = torch.linalg.solve_triangular(Lct.transpose(-1, -2), eps, upper=True)
            y_c2_T = y_c2.transpose(1, 2)
            y_t2_T = torch.linalg.solve_triangular(Ltt.transpose(-1, -2), y_c2_T, upper=True)
            y_t2 = y_t2_T.transpose(1, 2)
            z_target = mu_target + y_t2
        else:
            eps_T = eps.transpose(1, 2)
            y_t_T = torch.linalg.solve_triangular(Ltc.transpose(-1, -2), eps_T, upper=True)
            y_t = y_t_T.transpose(1, 2)
            z_condition = mu_condition + y_t
            
            y_t2_T = torch.linalg.solve_triangular(Ltt.transpose(-1, -2), eps_T, upper=True)
            y_t2 = y_t2_T.transpose(1, 2)
            z_target = mu_target + y_t2
            
        return z_condition, z_target

    def _unpatchify_embedder_output(self, patched_output):
        """Convert embedder output back to original format."""
        # This depends on your embedder's output format
        # Placeholder implementation
        return patched_output

    def _unpatchify_direct(self, patched_output):
        """Convert directly patchified output back to original sequence format."""
        batch_size, n_patches, patch_features = patched_output.shape
        n_channels = self.n_channels
        patch_length = patch_features // n_channels

        # Direct unpatchification
        h_hat = patched_output.contiguous().view(patched_output.size(0), -1, self.n_channels * self.patch_length)
        x_hat = self.pretrained_model.unpatchify_data(h_hat)
        decoder_smoothing_alpha = self.pretrained_model.decoder_smoothing_alpha
        if decoder_smoothing_alpha > 0.0:
            x_hat = (1 - decoder_smoothing_alpha) / decoder_smoothing_alpha * x_hat + decoder_smoothing_alpha * self.pretrained_model.smoothing(x_hat)
        # # Reshape and reconstruct sequence
            # reshaped = patched_output.reshape(batch_size, n_patches, n_channels, patch_length)
        # sequence = reshaped.permute(0, 2, 1, 3).contiguous()  # [B, n_channels, n_patches, patch_length]
        # sequence = sequence.reshape(batch_size, n_channels, n_patches * patch_length)
        # sequence = sequence.permute(0, 2, 1).contiguous()  # [B, seq_len, n_channels]

        return x_hat.transpose(1,2)

    def _compute_reconstruction_loss(self, target, prediction, time_mask: torch.Tensor = None, ood_threshold: float = None):
        """Compute weighted reconstruction loss with optional time masking (1=valid, 0=padded) and OOD channel masking."""
        if target.dim() == 3:
            b, d1, d2 = target.shape
            if d1 == self.n_channels and d2 != self.n_channels:
                target = target.transpose(1, 2).contiguous()
        if prediction.shape[1] != target.shape[1]:
            target_len = prediction.shape[1]
            target = target[:, :target_len, :]
            if time_mask is not None:
                time_mask = time_mask[:, :target_len]
        
        device = target.device
        channel_weights = self.channel_weights.to(device).view(1, 1, -1)
        discrete_mask = self.pretrained_model.discrete_mask.to(device)
        bce_loss_weight = getattr(self.pretrained_model, 'bce_loss_weight', 1.0)
        
        loss_val = torch.zeros_like(target)
        
        has_continuous = (~discrete_mask).any()
        has_discrete = discrete_mask.any()
        
        if has_continuous:
            cont_mask = (~discrete_mask).view(1, 1, -1)
            loss_val = torch.where(cont_mask, (target - prediction) ** 2, loss_val)
            
        if has_discrete:
            disc_mask = discrete_mask.view(1, 1, -1)
            target_clamped = torch.clamp(target, 0.0, 1.0)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_clamped, reduction='none')
            loss_val = torch.where(disc_mask, bce_loss_weight * bce, loss_val)
            
        # Dynamic OOD channel masking (Method 4)
        valid_mask = torch.ones_like(target)
        if ood_threshold is not None and ood_threshold > 0:
            cont_mask = (~discrete_mask).view(1, 1, -1)
            is_ood = cont_mask & (torch.abs(target) > ood_threshold)
            valid_mask = torch.where(is_ood, torch.zeros_like(valid_mask), valid_mask)

        # Apply weights and OOD valid mask
        weighted_loss = loss_val * channel_weights * valid_mask
        effective_weights = channel_weights * valid_mask
        
        if time_mask is not None:
            # Apply time mask by expanding to [B,T,C]
            time_mask_exp = time_mask.to(device).unsqueeze(-1)
            weighted_loss = weighted_loss * time_mask_exp
            num_valid = (time_mask_exp * effective_weights).sum(dim=[1, 2]).clamp(min=1.0)
            loss_per_sample = weighted_loss.sum(dim=[1, 2]) / num_valid
        else:
            num_valid = effective_weights.sum(dim=[1, 2]).clamp(min=1.0)
            loss_per_sample = weighted_loss.sum(dim=[1, 2]) / num_valid
        
        return loss_per_sample

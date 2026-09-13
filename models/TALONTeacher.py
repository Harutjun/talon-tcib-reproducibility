import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

from utils.base_utils import ModelOutput
from models.Modules import *
from models.TSPCVAE import TransformerBlock, TSPVAE_Decoder
torch.set_default_dtype(torch.float)


class TALONTeacher_Encoder(nn.Module):
    """
    Enhanced encoder module for the TSPVAE model using Banded Precision Gaussian posterior.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        latent_dim (int): Latent space dimension.
        sequence_length (int): Length of the input sequence.
        bandwidth (int): Bandwidth for the banded precision matrix.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length,
                 bandwidth=3, transformer_encoder_blocks=2, n_heads=4,
                 dim_feedforward=512, dropout=0.1, init_precision_diag=1.0, init_precision_offdiag=0.1,
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
        # Optional LayerNorm flags
        self.enc_input_layer_norm = bool(enc_input_layer_norm)
        self.enc_postproj_layer_norm = bool(enc_postproj_layer_norm)
        self.enc_mod_use_layer_norm = bool(enc_mod_use_layer_norm)

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

        # Transformer encoder blocks for feature extraction
        self.trans_enc_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(transformer_encoder_blocks)
        ])

        # Project to banded precision parameters
        self.precision_proj = nn.Linear(input_dim, hidden_dim)
        # Optional LayerNorm after projection
        self.postproj_ln = nn.LayerNorm(hidden_dim) if self.enc_postproj_layer_norm else nn.Identity()

        # Banded Precision Gaussian for the approximate posterior (global fallback)
        # Stateless-only posterior: remove legacy stateful posterior implementation.
        # Always use the stateless helper which provides default per-sample banded precision params.
        self.banded_posterior = BandedPrecisionGaussian_Stateless(
            dim=sequence_length,
            bandwidth=bandwidth,
        )

        # Mean prediction network now outputs latent_dim * sequence_length
        self.mean_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sequence_length * latent_dim)
        )
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

        # Run transformer blocks before modulation if configured OR if modulation disabled
        # Run transformer blocks before modulation if configured OR if modulation disabled
        if (not self.use_modulation) or (self.use_modulation and self.mod_pre_transformer):
            x = self.input_ln(x)
            x = self._apply_pe(x)
            for block in self.trans_enc_blocks:
                x = block(x)

        # Global pooling always for baseline path
        if patched_mask is not None:
            masked_x = x * patched_mask.unsqueeze(-1)
            seq_repr = masked_x.sum(dim=1) / patched_mask.sum(dim=1, keepdim=True)
        else:
            seq_repr = x.mean(dim=1)
        h = self.precision_proj(seq_repr)
        h = self.postproj_ln(h)
        mu_flat_base = self.mean_proj(h)
        mu_base = mu_flat_base.view(batch_size, self.latent_dim_out, self.sequence_length)  # [B,C,T]

        posterior_params = {'precision_diag': None, 'precision_bands': None}
        if self.use_modulation:
            # If modulation after transformer: we already updated x; else run transformer blocks now
            # If modulation after transformer: we already updated x; else run transformer blocks now
            if not self.mod_pre_transformer:
                x = self.input_ln(x)
                x = self._apply_pe(x)
                for block in self.trans_enc_blocks:
                    x = block(x)
            # Sequentially refine using modulation blocks (take last block outputs)
            mu_mod = None
            pdiag = None
            pbands = None
            for mod in self.mod_blocks:
                pd, pb, mu_temp = mod(x, mask=patched_mask if patched_mask is not None else None)
                mu_temp = mu_temp.transpose(1, 2)  # [B,C,T]
                mu_mod = mu_temp if mu_mod is None else mu_temp  # last block output
                pdiag = pd
                pbands = pb
            if self.mod_mode == 'replace':
                mu_batch = mu_mod
            else:  # residual
                mu_batch = mu_base + self.residual_alpha * mu_mod
            posterior_params = {'precision_diag': pdiag, 'precision_bands': pbands}
        else:
            mu_batch = mu_base

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


class ContextMHSA(nn.Module):
    """Multi-Head Self-Attention + FFN block for latent context encoding.
    Operates over temporal dimension for each latent channel vector.
    Input: z_time_first [B,T,C_in]; projects to hidden_dim.
    """
    def __init__(self, in_channels, hidden_dim, num_heads=4, dropout=0.1, use_layer_norm=True, ffn_multiplier=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # Adjust heads if not divisible
        if hidden_dim % num_heads != 0:
            # Find largest divisor <= num_heads
            divisors = [h for h in range(num_heads, 0, -1) if hidden_dim % h == 0]
            new_h = divisors[0]
            print(f"[ContextMHSA] Adjusting heads {num_heads}->{new_h} to divide hidden_dim {hidden_dim}")
            self.num_heads = new_h
        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        self.ln1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        ffn_hidden = ffn_multiplier * hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x [B,T,C_in]
        x_proj = self.in_proj(x)  # [B,T,H]
        h = self.ln1(x_proj)
        qkv = self.qkv_proj(h)  # [B,T,3H]
        q, k, v = qkv.chunk(3, dim=-1)
        B, T, _ = q.shape

        def reshape_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,Hd,T,head_dim]

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)
        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B,Hd,T,T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, vh)  # [B,Hd,T,head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        attn_out = self.out_proj(attn_out)
        x_res = x_proj + self.dropout(attn_out)  # residual 1
        h2 = self.ln2(x_res)
        ff_out = self.ffn(h2)
        return x_res + ff_out  # residual 2


class MultiLayerMHSA(nn.Module):
    """Stack of ContextMHSA blocks mapping [B,T,C_in] -> [B,T,H] using num_layers."""
    def __init__(self, in_channels, hidden_dim, num_heads=4, num_layers=1, dropout=0.1, use_layer_norm=True, ffn_multiplier=4):
        super().__init__()
        layers = []
        # First layer maps in_channels -> hidden_dim
        layers.append(ContextMHSA(in_channels, hidden_dim, num_heads=num_heads, dropout=dropout, use_layer_norm=use_layer_norm, ffn_multiplier=ffn_multiplier))
        # Subsequent layers keep hidden_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(ContextMHSA(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout, use_layer_norm=use_layer_norm, ffn_multiplier=ffn_multiplier))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for blk in self.layers:
            h = blk(h)
        return h


class ChannelContextAttention(nn.Module):
    """Self-attention across latent channels per time step.
    Input: z_time_first [B,T,C] with C = latent_dim (tokens across channels).
    Process each time step with a TransformerEncoder over C tokens, embedding dim = hidden_dim.
    Output: [B,T,hidden_dim] via mean pooling across channels.
    """
    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.1, use_layer_norm=True):
        super().__init__()
        # Adjust heads if hidden_dim not divisible
        if hidden_dim % num_heads != 0:
            divisors = [h for h in range(num_heads, 0, -1) if hidden_dim % h == 0]
            num_heads = divisors[0]
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.in_proj = nn.Linear(1, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, x):  # x: [B,T,C]
        B, T, C = x.shape
        xt = x.reshape(B * T, C, 1)  # tokens = C, feature dim = 1
        h = self.in_proj(xt)  # [B*T, C, H]
        h = self.encoder(h)   # [B*T, C, H]
        h = self.ln(h)
        # Pool across channels
        h_pool = h.mean(dim=1)  # [B*T, H]
        return h_pool.view(B, T, -1)


class TALONTeacher(nn.Module):
    """
    Enhanced TSPVAE model using GP Prior and Banded Precision Gaussian posterior.

    Reworked initializer: explicit arguments replace the previous `args`-based configuration.
    Sub-modules (encoder/decoder/context) accept their own kwargs via encoder_kwargs, decoder_kwargs, context_kwargs.
    """
    def __init__(
        self,
        latent_dim,
        input_dim,
        sequence_length,
        patch_length=50,
        patch_embedder=None,
        freeze_embedder=True,
        enc_hidden_dim=None,
        dec_hidden_dim=None,
        gp_time_kernel='rbf',
        rank_c=1,
        gp_jitter=1e-6,
        bandwidth=3,
        init_precision_diag=1.0,
        init_precision_offdiag=0.1,
        use_cholesky_param=True,
        posterior_tc_banded=False,
        tc_channel_bandwidth=0,
        encoder_kwargs: dict = None,
        decoder_kwargs: dict = None,
        context_kwargs: dict = None,
        channel_weights=None,
        decoder_smoothing_alpha: float = 0.0,
        posterior_jitter: float = 1e-8,
        posterior_jitter_max: float = 1e-6,
        discrete_mask=None,
        bce_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Core shapes and dims (explicit)
        self.latent_dim = int(latent_dim)
        self.enc_hidden_dim = int(enc_hidden_dim) if enc_hidden_dim is not None else 128
        self.dec_hidden_dim = int(dec_hidden_dim) if dec_hidden_dim is not None else 128
        self.sequence_length = int(sequence_length)
        self.patch_length = int(patch_length)
        self.n_channels = int(input_dim)

        # Embedder
        self.embedder = patch_embedder
        self.use_embedder = patch_embedder is not None

        # Channel reconstruction weights
        if channel_weights is not None:
            if len(channel_weights) != self.n_channels:
                raise ValueError(f"Channel weights length ({len(channel_weights)}) must match number of channels ({self.n_channels})")
            self.channel_weights = torch.tensor(channel_weights, dtype=torch.float)
        else:
            self.channel_weights = torch.ones(self.n_channels, dtype=torch.float)
        print(f"Channel reconstruction weights: {self.channel_weights.tolist()}")

        # Register discrete_mask as a non-persistent buffer so it doesn't get saved/loaded in state dict
        if discrete_mask is not None:
            self.register_buffer('discrete_mask', torch.tensor(discrete_mask, dtype=torch.bool), persistent=False)
        else:
            self.register_buffer('discrete_mask', torch.zeros(self.n_channels, dtype=torch.bool), persistent=False)
            
        self.bce_loss_weight = float(bce_loss_weight)

        # Compute encoder/decoder input dims depending on embedder usage
        if self.use_embedder:
            # freeze embedder if requested
            for param in self.embedder.parameters():
                param.requires_grad = not freeze_embedder
            if freeze_embedder:
                self.embedder.eval()

            encoder_input_dim = self.embedder.config.d_model * self.n_channels
            decoder_output_dim = self.n_channels * self.embedder.config.d_model
            n_patches = int(self.sequence_length // self.patch_length)
        else:
            encoder_input_dim = self.n_channels * self.patch_length
            decoder_output_dim = self.n_channels * self.patch_length
            n_patches = int(self.sequence_length // self.patch_length)

        # Posterior tc-banded flags and bandwidths
        self.posterior_tc_banded = bool(posterior_tc_banded)
        self.tc_time_bandwidth = int(bandwidth)
        self.tc_channel_bandwidth = int(tc_channel_bandwidth)

        # Merge encoder kwargs with sensible defaults (caller can override)
        enc_kwargs = encoder_kwargs or {}
        enc_init = {
            'input_dim': encoder_input_dim,
            'hidden_dim': int(enc_kwargs.get('hidden_dim', self.enc_hidden_dim)),
            'latent_dim': int(self.latent_dim),
            'sequence_length': n_patches,
            'bandwidth': int(enc_kwargs.get('bandwidth', self.tc_time_bandwidth)),
            'transformer_encoder_blocks': int(enc_kwargs.get('transformer_encoder_blocks', 2)),
            'n_heads': int(enc_kwargs.get('n_heads', 4)),
            'dim_feedforward': int(enc_kwargs.get('dim_feedforward', 512)),
            'dropout': float(enc_kwargs.get('dropout', 0.1)),
            'init_precision_diag': float(enc_kwargs.get('init_precision_diag', init_precision_diag)),
            'init_precision_offdiag': float(enc_kwargs.get('init_precision_offdiag', init_precision_offdiag)),
            'use_modulation': bool(enc_kwargs.get('use_modulation', True)),
            'mod_num_blocks': int(enc_kwargs.get('mod_num_blocks', 1)),
            'mod_mode': enc_kwargs.get('mod_mode', 'residual'),
            'mod_pre_transformer': bool(enc_kwargs.get('mod_pre_transformer', True)),
            'residual_alpha_init': float(enc_kwargs.get('residual_alpha_init', 1.0)),
            'use_cholesky_param': bool(enc_kwargs.get('use_cholesky_param', use_cholesky_param)),
            'tc_banded_enabled': bool(enc_kwargs.get('tc_banded_enabled', self.posterior_tc_banded)),
            'tc_channel_bandwidth': int(enc_kwargs.get('tc_channel_bandwidth', self.tc_channel_bandwidth)),
            'enc_input_layer_norm': bool(enc_kwargs.get('enc_input_layer_norm', False)),
            'enc_postproj_layer_norm': bool(enc_kwargs.get('enc_postproj_layer_norm', False)),
            'enc_mod_use_layer_norm': bool(enc_kwargs.get('enc_mod_use_layer_norm', False)),
        }
        # allow caller to override or add custom keys
        enc_init.update({k: v for k, v in enc_kwargs.items()})

        # Create encoder
        self.encoder = TALONTeacher_Encoder(**enc_init)

        # One-time flags about per-batch posterior parameters
        self._per_batch_time_enabled = bool(enc_init.get('use_modulation', True))
        self._per_batch_channel_enabled = bool(self.posterior_tc_banded and getattr(self.encoder, 'channel_precision_head', None) is not None and self.tc_channel_bandwidth > 0)

        if not self._per_batch_time_enabled:
            raise RuntimeError(
                "Configuration error: per-batch time precision posterior is DISABLED. "
                "This model requires the encoder to provide per-sample posterior precision parameters (precision_diag and precision_bands). "
                "Enable encoder modulation (use_modulation=True) or provide a per-sample posterior head in the encoder config."
            )

        if self.posterior_tc_banded and not self._per_batch_channel_enabled:
            raise RuntimeError(
                "Configuration error: TC-banded posterior requested but channel precision head is not configured. "
                "If posterior_tc_banded is True, set tc_channel_bandwidth>0 and provide a channel_precision_head in the encoder config."
            )

        # GP Prior with Kronecker structure
        self.gp_prior = KronTimeChannelKernel(
            T=int(n_patches),
            C=int(self.latent_dim),
            time_kernel=gp_time_kernel,
            rank_c=int(rank_c),
            jitter=float(gp_jitter)
        )

        # Build decoder kwargs and initialize decoder
        dec_kwargs = decoder_kwargs or {}
        dec_init = {
            'latent_dim': int(self.latent_dim),
            'hidden_dim': int(dec_kwargs.get('hidden_dim', self.dec_hidden_dim)),
            'output_dim': int(decoder_output_dim),
            'num_layers': int(dec_kwargs.get('num_layers', dec_kwargs.get('DecoderNumLayers', 2))),
            'num_heads': int(dec_kwargs.get('num_heads', dec_kwargs.get('DecoderNumHeads', 4))),
            'dropout': float(dec_kwargs.get('dropout', 0.1)),
            'max_sequence_length': int(n_patches),
            'dim_feedforward': int(dec_kwargs.get('dim_feedforward', dec_kwargs.get('DecoderFFDim', 512))),
            'ar_mode': bool(dec_kwargs.get('ar_mode', False)),
            'pe_type': dec_kwargs.get('pe_type', dec_kwargs.get('DecoderPEType', 'sinusoidal')),
            'pe_dropout': float(dec_kwargs.get('pe_dropout', dec_kwargs.get('DecoderPEDropout', 0.0))),
        }
        dec_init.update({k: v for k, v in dec_kwargs.items()})

        self.decoder = TSPVAE_Decoder(**dec_init)
        self.ar_mode = bool(dec_init.get('ar_mode', False))
        self.patch_embedding_layer = nn.Conv1d(in_channels=input_dim,
                                         out_channels=self.patch_length*input_dim,
                                         kernel_size=self.patch_length,
                                         stride=self.patch_length)
        # Context encoder setup
        ctx_kwargs = context_kwargs or {}
        self.context_type = ctx_kwargs.get('type', 'mlp').lower()
        att_heads = int(ctx_kwargs.get('attention_heads', ctx_kwargs.get('context_attention_heads', 4)))
        ctx_dropout = float(ctx_kwargs.get('dropout', ctx_kwargs.get('context_dropout', 0.1)))
        ctx_ln = bool(ctx_kwargs.get('use_layer_norm', ctx_kwargs.get('context_use_layer_norm', True)))
        ffn_mult = int(ctx_kwargs.get('hidden_multiplier', ctx_kwargs.get('context_hidden_multiplier', 4)))
        ctx_layers = int(ctx_kwargs.get('num_layers', ctx_kwargs.get('context_num_layers', 1)))

        if self.context_type == 'mhsa':
            self.context_encoder = MultiLayerMHSA(
                in_channels=self.latent_dim,
                hidden_dim=self.dec_hidden_dim,
                num_heads=att_heads,
                num_layers=ctx_layers,
                dropout=ctx_dropout,
                use_layer_norm=ctx_ln,
                ffn_multiplier=ffn_mult
            )
        elif self.context_type == 'attention':
            self.context_encoder = ChannelContextAttention(
                hidden_dim=self.dec_hidden_dim,
                num_heads=att_heads,
                num_layers=ctx_layers,
                dropout=ctx_dropout,
                use_layer_norm=ctx_ln
            )
            self._chan_attn_in_proj = nn.Linear(self.latent_dim, 1)
        else:
            self.context_encoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.dec_hidden_dim),
                nn.GELU(),
                nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim)
            )

        # Smoothing layer and decoder smoothing alpha
        self.smoothing = nn.Sequential(
            nn.Conv1d(self.n_channels, self.n_channels, kernel_size=3, padding=1, groups=self.n_channels)
        )
        self.decoder_smoothing_alpha = float(decoder_smoothing_alpha)

        # Numerical jitter settings for posterior sampling
        self.posterior_jitter = float(posterior_jitter)
        self.posterior_jitter_max = float(posterior_jitter_max)

        # Cache
        self._cached_prior = {}

    def patchify_data(self, x, irrelevant_mask=None):
        """
        Patchify input data when no embedder is available.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, sequence_length)
            irrelevant_mask (torch.Tensor, optional): Mask tensor

        Returns:
            tuple: (patchified_data [B, n_patches, C*P], patched_mask [B, n_patches] or None)
        """
        batch_size, n_channels, seq_len = x.shape
        if seq_len % self.patch_length != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by patch_length {self.patch_length}. "
                f"Set model.sequence_length to a multiple of patch_length in the config or preprocess/pad the data accordingly."
            )
        n_patches = seq_len // self.patch_length

        # Reshape to patches: [BS, C, T] -> [BS, C, n_patches, patch_length] -> [BS, n_patches, C*patch_length]
        # x_patches = x[:, :, :n_patches * self.patch_length]  # Trim to exact multiple
        # x_patches = x_patches.view(batch_size, n_channels, n_patches, self.patch_length)
        # x_patches = x_patches.permute(0, 2, 1, 3)  # [BS, n_patches, C, patch_length]
        # x_patches = x_patches.contiguous().view(batch_size, n_patches, n_channels * self.patch_length)
        x_patches = self.patch_embedding_layer(x).permute(0, 2, 1)  # [BS, n_patches, enc_hidden_dim]

        # Handle mask if provided
        patched_mask = None
        if irrelevant_mask is not None:
            # irrelevant_mask is [BS, T, C] -> reshape to patches
            mask_patches = irrelevant_mask[:, :n_patches * self.patch_length, :]
            mask_patches = mask_patches.view(batch_size, n_patches, self.patch_length, n_channels)
            # Average over patch and channel dimensions to get patch-level mask
            patched_mask = mask_patches.float().mean(dim=[2, 3])  # [BS, n_patches]
            patched_mask = (patched_mask > 0.5).float()

        return x_patches, patched_mask

    def unpatchify_data(self, x_patches):
        """
        Convert patchified data back to original format.
        Ensures output matches original sequence length.

        Args:
            x_patches (torch.Tensor): Patchified tensor of shape [BS, n_patches, C*patch_length]

        Returns:
            torch.Tensor: Unpatchified tensor of shape [BS, C, original_sequence_length]
        """
        batch_size, n_patches, _ = x_patches.shape

        # Reshape back: [BS, n_patches, C*patch_length] -> [BS, n_patches, C, patch_length] -> [BS, C, T]
        x_patches = x_patches.view(batch_size, n_patches, self.n_channels, self.patch_length)
        x_patches = x_patches.permute(0, 2, 1, 3)  # [BS, C, n_patches, patch_length]
        x_reconstructed = x_patches.contiguous().view(batch_size, self.n_channels, n_patches * self.patch_length)

        # If the reconstructed length is shorter than original, pad with zeros or interpolate
        current_length = x_reconstructed.shape[-1]
        if current_length < self.sequence_length:
            padding = self.sequence_length - current_length
            x_reconstructed = torch.nn.functional.pad(x_reconstructed, (0, padding), mode='constant', value=0.0)
        elif current_length > self.sequence_length:
            x_reconstructed = x_reconstructed[:, :, :self.sequence_length]

        return x_reconstructed

    def forward(self, x: torch.Tensor, irrelevant_mask=None):
        """
        Forward pass for the Enhanced TSPVAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, sequence_length).
            irrelevant_mask (torch.Tensor, optional): Mask for irrelevant tokens.

        Returns:
            ModelOutput: Output containing reconstructed x, latent variables, and losses.
        """
        device = x.device
        batch_size = x.shape[0]

        # Handle data patchification
        if self.use_embedder:
            # Use pre-trained embedder
            # PatchTST expects input shape [batch_size, sequence_length, channels]
            # but we have [batch_size, channels, sequence_length], so we need to transpose
            x_for_embedder = x.permute(0, 2, 1)  # [batch_size, sequence_length, channels]

            # Adjust irrelevant_mask to match the transposed input
            if irrelevant_mask is not None:
                # irrelevant_mask is [batch_size, sequence_length, channels]
                embedder_mask = irrelevant_mask
            else:
                embedder_mask = None

            # Get encoder outputs from the underlying PatchTST model (not the pretraining head)
            model_out = self.embedder.model(
                past_values=x_for_embedder,
                irrelevant_mask=embedder_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            hidden_state = model_out.last_hidden_state  # [B, C, P or P+1, D]
            # If CLS token is used, drop it for patchified features
            if getattr(self.embedder.config, 'use_cls_token', False):
                hidden_state = hidden_state[:, :, 1:, :]
            # Flatten to [B, P, C*D]
            x_patchified = hidden_state.permute(0, 2, 1, 3).contiguous().view(
                x.size(0), hidden_state.size(2), -1
            ).to(device)

            # Build patch-level mask [B, P] by aggregating across channels and patch_length
            if embedder_mask is not None and hasattr(model_out, 'patched_mask') and model_out.patched_mask is not None:
                # model_out.patched_mask: [B, C, P, L]
                patched_mask = (model_out.patched_mask.float().mean(dim=(1, 3)) > 0.5).to(device)
            else:
                patched_mask = None
        else:
            # Use direct patchification
            x_patchified, patched_mask = self.patchify_data(x, irrelevant_mask)
            x_patchified = x_patchified.to(device)
            if patched_mask is not None:
                patched_mask = patched_mask.to(device)

        # Encode to get approximate posterior
        mu_batch, posterior_params = self.encoder(x_patchified, patched_mask=patched_mask)  # mu_batch [B,C,T]
        B, C, T = mu_batch.shape

        # Deterministic reconstructions at eval: use z = mu (no sampling)
        if not self.training:
            z = mu_batch
        else:
            # Require per-batch posterior precision parameters. No global fallback allowed.
            time_diag = posterior_params.get('precision_diag', None)
            time_bands = posterior_params.get('precision_bands', None)
            chan_diag = posterior_params.get('precision_channel_diag', None)
            chan_bands = posterior_params.get('precision_channel_bands', None)

            if time_diag is None or time_bands is None:
                raise RuntimeError(
                    "Encoder did not provide per-sample time precision parameters (precision_diag / precision_bands). "
                    "This model requires per-batch posterior precisions for sampling during training. "
                    "Enable encoder modulation or ensure the modulation blocks return valid precision tensors."
                )

            use_tc_banded = self.posterior_tc_banded
            if use_tc_banded and (chan_diag is None or chan_bands is None):
                raise RuntimeError(
                    "TC-banded posterior is enabled but encoder did not provide per-sample channel precision parameters. "
                    "Provide precision_channel_diag and precision_channel_bands from the encoder."
                )

            # Assemble per-batch precision matrices and sample using Kron structure when applicable
            P_t_batch = assemble_precision_from_bands_fn(
                time_diag,
                time_bands,
                bandwidth=int(getattr(self.encoder, 'bandwidth', T - 1))
            )  # [B,T,T]

            # Ensure variable exists (silence static analysis) and build channel precision if requested
            P_c_batch = None
            if use_tc_banded:
                P_c_batch = assemble_precision_from_bands_fn(
                    chan_diag,
                    chan_bands,
                    bandwidth=int(getattr(self.encoder, 'tc_channel_bandwidth', max(0, C - 1)))
                )  # [B,C,C]

            z = torch.empty(B, C, T, device=device, dtype=mu_batch.dtype)
            jitter_start = float(getattr(self, 'posterior_jitter', 1e-8))
            jitter_max = float(getattr(self, 'posterior_jitter_max', 1e-6))

            if use_tc_banded:
                # Kron sampling using per-batch time and channel precision
                # Try batched Cholesky for speed;
                Lt = torch.linalg.cholesky(P_t_batch + jitter_start * torch.eye(T, device=device, dtype=mu_batch.dtype))  # [B,T,T]
                Lc = torch.linalg.cholesky(P_c_batch + jitter_start * torch.eye(C, device=device, dtype=mu_batch.dtype))  # [B,C,C]
                eps = torch.randn(B, C, T, device=device, dtype=mu_batch.dtype)
                y = torch.linalg.solve_triangular(Lc.transpose(-1, -2), eps, upper=True)  # [B,C,T]
                z_b = torch.linalg.solve_triangular(Lt.transpose(-1, -2), y.transpose(-2, -1), upper=True).transpose(-2, -1)  # [B,C,T]
                z = mu_batch + z_b
            else:
                # Per-batch time precision and identity channel precision
                for b in range(B):
                    Qb = P_t_batch[b]
                    L_Qb = robust_cholesky_fn(Qb, jitter_start=jitter_start, jitter_max=jitter_max)
                    eps = torch.randn(C, T, device=device, dtype=mu_batch.dtype)
                    y = torch.linalg.solve_triangular(L_Qb.transpose(-1, -2), eps.T, upper=True).T  # [C,T]
                    z[b] = mu_batch[b] + y

        # Compute latent context features for decoder
        # z shape [B,C,T] -> transpose to [B,T,C]
        z_time_first = z.permute(0, 2, 1)  # [B,T,C]
        if self.context_type == 'mlp':
            context_features = self.context_encoder(z_time_first)
        elif self.context_type == 'attention':
            # Feed z_time_first directly; ChannelContextAttention will treat channels as tokens
            context_features = self.context_encoder(z_time_first)  # [B,T,H]
        elif self.context_type == 'mhsa':
            context_features = self.context_encoder(z_time_first)
        else:
            context_features = z_time_first  # fallback
        # Decoder expects context shape [B,T,hidden_dim]
        h_hat = self.decoder(
            context=context_features,
            patched_mask=patched_mask
        ).transpose(0, 1)

        # Reconstruct original signal
        if self.use_embedder:
            # Use embedder's head for reconstruction
            h_hat = h_hat.contiguous().view(
                -1,
                self.sequence_length // self.embedder.config.patch_length,
                self.n_channels,
                self.embedder.config.d_model
            ).permute(0, 2, 1, 3)  # [B, C, P, D]
            # If CLS token is used, prepend a dummy token so the head can drop it consistently
            if getattr(self.embedder.config, 'use_cls_token', False):
                Bc, Cc, Pp, Dd = h_hat.shape
                cls_pad = torch.zeros(Bc, Cc, 1, Dd, device=h_hat.device, dtype=h_hat.dtype)
                h_hat = torch.cat([cls_pad, h_hat], dim=2)  # [B, C, P+1, D]
            head_out = self.embedder.head(h_hat)  # [B, C, P, L]
            x_hat = head_out.reshape(-1, self.n_channels, self.sequence_length)
        else:
            # Direct unpatchification
            h_hat = h_hat.contiguous().view(x.size(0), -1, self.n_channels * self.patch_length)
            x_hat = self.unpatchify_data(h_hat)
            # Apply residual temporal smoothing only when no pre-trained embedder is used
            if self.decoder_smoothing_alpha > 0.0:
                x_hat = ((1 - self.decoder_smoothing_alpha) / self.decoder_smoothing_alpha) * x_hat + self.decoder_smoothing_alpha * self.smoothing(x_hat)

        # Compute losses with proper masking
        # Require posterior_params to be a per-batch dict (no global posterior allowed)
        kl_loss = self.compute_kl_divergence(mu_batch, posterior_params, patched_mask)
        reconstruction_loss = self.compute_masked_reconstruction_loss(x_hat, x, irrelevant_mask)

        # Apply sigmoid activation to the reconstructed outputs of discrete channels
        if self.discrete_mask.any():
            disc_mask = self.discrete_mask.view(1, -1, 1) # [1, n_channels, 1]
            x_hat_probs = torch.sigmoid(x_hat)
            x_hat = torch.where(disc_mask, x_hat_probs, x_hat)

        return ModelOutput(
            x_hat=x_hat,
            z=z,
            mu=mu_batch,
            L=None,
            KL_Loss=kl_loss,
            reconstruction_loss=reconstruction_loss,
            # Provide the per-batch posterior_params (global/module posterior is disallowed)
            banded_posterior=posterior_params
        )

    def compute_masked_reconstruction_loss(self, x_hat, x, irrelevant_mask=None):
        """
        Compute reconstruction loss excluding padded regions with per-channel weighting.
        Uses SUM over channels and sequence length, then MEAN over batch (for proper ELBO).
        Computes MSE for continuous features and BCE with logits for discrete features.

        Args:
            x_hat (torch.Tensor): Reconstructed tensor [batch_size, n_channels, sequence_length]
            x (torch.Tensor): Target tensor [batch_size, n_channels, sequence_length]
            irrelevant_mask (torch.Tensor, optional): Mask tensor [batch_size, sequence_length, n_channels]

        Returns:
            torch.Tensor: Reconstruction loss per sample [batch_size]
        """
        device = x.device
        channel_weights = self.channel_weights.to(device)
        discrete_mask = self.discrete_mask.to(device)

        # Compute element-wise hybrid loss (MSE for continuous, BCE for discrete)
        loss = torch.zeros_like(x)
        
        has_continuous = (~discrete_mask).any()
        has_discrete = discrete_mask.any()
        
        if has_continuous:
            cont_mask = (~discrete_mask).view(1, -1, 1)
            loss = torch.where(cont_mask, (x_hat - x) ** 2, loss)
            
        if has_discrete:
            disc_mask = discrete_mask.view(1, -1, 1)
            x_clamped = torch.clamp(x, 0.0, 1.0)
            bce_val = F.binary_cross_entropy_with_logits(x_hat, x_clamped, reduction='none')
            loss = torch.where(disc_mask, self.bce_loss_weight * bce_val, loss)

        # Apply per-channel weights: broadcast weights to match tensor shape
        # channel_weights: [n_channels] -> [1, n_channels, 1] to broadcast properly
        weighted_loss = loss * channel_weights.view(1, -1, 1)  # [batch_size, n_channels, sequence_length]

        if irrelevant_mask is not None:
            # irrelevant_mask is [batch_size, sequence_length, n_channels]
            # Need to permute to match weighted_loss shape [batch_size, n_channels, sequence_length]
            mask = irrelevant_mask.permute(0, 2, 1)  # [batch_size, n_channels, sequence_length]

            # Apply mask and sum only over non-masked elements
            masked_weighted_loss = weighted_loss * mask
            # Sum over channels and sequence length for each sample
            reconstruction_loss_per_sample = masked_weighted_loss.sum(dim=[1, 2])  # [batch_size]
        else:
            # No mask: sum over channels and sequence length for each sample
            reconstruction_loss_per_sample = weighted_loss.sum(dim=[1, 2])  # [batch_size]

        # Return loss per sample
        return reconstruction_loss_per_sample

    def compute_kl_divergence(self, mu_batch, posterior, mask=None):
        """KL(q||p) with q: Σ_q = Q^{-1} ⊗ I_C or Q_t^{-1} ⊗ Q_c^{-1}, mean μ (B,C,T); p: N(0, K_t ⊗ K_c).
        Strict requirement: posterior MUST be a per-batch dict containing precision_diag and precision_bands.
        Global/module posterior usage is NOT allowed in this implementation.
        """
        if not isinstance(posterior, dict):
            raise RuntimeError(
                "compute_kl_divergence requires a per-batch posterior dict with keys 'precision_diag' and 'precision_bands'. "
                "Global encoder.banded_posterior module is disallowed."
            )

        B, C, T = mu_batch.shape
        device = mu_batch.device
        # Prior components and inverses
        K_t = self.gp_prior.K_t(); K_c = self.gp_prior.K_c()
        L_Kt = torch.linalg.cholesky(K_t)
        L_Kc = torch.linalg.cholesky(K_c)
        I_T = torch.eye(T, device=device, dtype=K_t.dtype)
        I_C = torch.eye(C, device=device, dtype=K_c.dtype)
        K_t_inv = torch.cholesky_solve(I_T, L_Kt)
        K_c_inv = torch.cholesky_solve(I_C, L_Kc)
        log_det_Kt = 2 * torch.sum(torch.log(torch.diag(L_Kt)))
        log_det_Kc = 2 * torch.sum(torch.log(torch.diag(L_Kc)))
        log_det_K = C * log_det_Kt + T * log_det_Kc
        jitter = float(self.gp_prior.jitter.item()) if isinstance(self.gp_prior.jitter, torch.Tensor) else float(self.gp_prior.jitter)

        # Validate per-batch posterior fields
        time_diag = posterior.get('precision_diag', None)
        time_bands = posterior.get('precision_bands', None)
        chan_diag = posterior.get('precision_channel_diag', None)
        chan_bands = posterior.get('precision_channel_bands', None)

        if time_diag is None or time_bands is None:
            raise RuntimeError(
                "Per-batch posterior missing required time precision components (precision_diag / precision_bands). "
                "This implementation requires per-sample posterior precision parameters from the encoder."
            )

        has_channel = (self.posterior_tc_banded and chan_diag is not None and chan_bands is not None)

        # Time precision
        P_t_batch = assemble_precision_from_bands_fn(
            time_diag, time_bands, bandwidth=int(getattr(self.encoder, 'bandwidth', T - 1))
        )
        log_det_Qt_b = []
        Qt_inv_b = []
        for b in range(B):
            Qt = P_t_batch[b]
            L_Qt = robust_cholesky_fn(Qt, jitter_start=jitter, jitter_max=1e-1)
            log_det_Qt_b.append(2 * torch.sum(torch.log(torch.diag(L_Qt))))
            Qt_inv_b.append(torch.cholesky_solve(I_T, L_Qt))
        log_det_Qt = torch.stack(log_det_Qt_b, dim=0)
        Qt_inv_stack = torch.stack(Qt_inv_b, dim=0)  # [B,T,T]

        if has_channel:
            # Channel precision
            P_c_batch = assemble_precision_from_bands_fn(
                chan_diag, chan_bands, bandwidth=int(getattr(self.encoder, 'tc_channel_bandwidth', max(0, C - 1)))
            )
            log_det_Qc_b = []
            Qc_inv_b = []
            for b in range(B):
                Qc = P_c_batch[b]
                L_Qc = robust_cholesky_fn(Qc, jitter_start=jitter, jitter_max=1e-1)
                log_det_Qc_b.append(2 * torch.sum(torch.log(torch.diag(L_Qc))))
                Qc_inv_b.append(torch.cholesky_solve(I_C, L_Qc))
            log_det_Qc = torch.stack(log_det_Qc_b, dim=0)
            Qc_inv_stack = torch.stack(Qc_inv_b, dim=0)  # [B,C,C]
            # Trace term: tr(K_t^{-1} Qt^{-1}) tr(K_c^{-1} Qc^{-1})
            trace_t = torch.sum(K_t_inv * Qt_inv_stack, dim=[1, 2])  # [B]
            trace_c = torch.sum(K_c_inv * Qc_inv_stack, dim=[1, 2])  # [B]
            trace_term = trace_t * trace_c  # [B]
            # Logdet of Q = |Qt|^C |Qc|^T -> log|Q| = C log|Qt| + T log|Qc|
            log_det_Q = C * log_det_Qt + T * log_det_Qc
        else:
            # Original case: Q = Qt ⊗ I -> log|Q| = C log|Qt|, trace = tr(K_t^{-1} Qt^{-1}) tr(K_c^{-1})
            trace_term = torch.sum(K_t_inv * Qt_inv_stack, dim=[1, 2]) * torch.trace(K_c_inv)
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
        KL_batch = 0.5 * (log_det_K + log_det_Q - n_eff + trace_term + quad)
        return KL_batch

# Module-level functional helpers to avoid in-place autograd issues

def assemble_precision_from_bands_fn(precision_diag, precision_bands, bandwidth):
    """Functional builder for symmetric banded precision matrices.
    precision_diag: [B,T] or [B,C]; precision_bands: [B,b,T-1] or [B,b,C-1]; returns [B,T,T] or [B,C,C].
    Enforces PD via Gershgorin-style diagonal dominance without in-place-after-use.
    """
    B = precision_diag.size(0)
    T = precision_diag.size(1)
    device = precision_diag.device
    dtype = precision_diag.dtype
    diag_pos = softplus(precision_diag)  # [B,T]
    Off = torch.zeros(B, T, T, device=device, dtype=dtype)
    b_eff = precision_bands.size(1)
    for k in range(min(b_eff, bandwidth)):
        offs = k + 1
        L = max(0, T - offs)
        if L == 0:
            continue
        vals = precision_bands[:, k, :L]  # [B, L]
        i = torch.arange(L, device=device)
        Off[:, i, i + offs] = vals
        Off[:, i + offs, i] = vals
    off_abs = Off.abs().sum(dim=-1)  # [B,T]
    min_diag = off_abs + 1e-6
    diag_final = torch.maximum(diag_pos, min_diag)
    return torch.diag_embed(diag_final) + Off


def robust_cholesky_fn(M, jitter_start=1e-6, jitter_max=1e-1, max_tries=7):
    """Attempt Cholesky with exponentially increasing jitter until success or cap."""
    device = M.device
    dtype = M.dtype
    jitter = float(jitter_start)
    I = torch.eye(M.size(-1), device=device, dtype=dtype)
    return torch.linalg.cholesky(M + jitter_start * I)


class _TALONTeacher_Shadow(nn.Module):
    """Deprecated shadow class placeholder (no-op)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    pass

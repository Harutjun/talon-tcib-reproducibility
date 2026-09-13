import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.base_utils import ModelOutput
from utils.TSPLossFunctions import *

torch.set_default_dtype(torch.float)
from models.Modules import *
from utils.TSPUtils import sample_posterior, construct_cholesky


class TSPVAE_Encoder(nn.Module):
    """
    Encoder module for the TSPVAE model.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        latent_dim (int): Latent space dimension.
        sequence_length (int): Length of the input sequence.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, transformer_encoder_blocks=2, n_heads=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.mu_L_Elements = AttentionModulationBlock(input_dim, hidden_dim, latent_dim, sequence_length)
        self.trans_enc_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(transformer_encoder_blocks)
        ])

    def forward(self, x: torch.Tensor, patched_mask=None):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            patched_mask (torch.Tensor, optional): Mask for the input sequence.

        Returns:
            tuple: Mean (mu) and lower triangular matrix (L) of the latent distribution.
        """
        device = x.device
        for block in self.trans_enc_blocks:
            x = block(x)

        L, A, mu = self.mu_L_Elements(x, patched_mask)
        if patched_mask is not None:
            mu = mu.squeeze() * patched_mask
        L_elements = L[:, torch.tril_indices(L.shape[1], L.shape[2], 0, device=device)[0],
        torch.tril_indices(L.shape[1], L.shape[2], 0, device=device)[1]]
        L = construct_cholesky(L_elements, self.sequence_length)
        return mu, L


class TransformerBlock(nn.Module):
    """
    Simplified decoder block with self-attention and feedforward layers.

    Args:
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerBlock, self).__init__()
        # Auto-adjust heads if d_model not divisible
        if d_model % nhead != 0:
            divisors = [h for h in range(nhead, 0, -1) if d_model % h == 0]
            nhead = divisors[0]
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the causal decoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): mask for self-attention with 1 for valid, 0 for invalid or boolean with True for invalid.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        key_padding_mask = None
        if mask is not None:
            # Normalize to boolean mask where True indicates positions to ignore
            if mask.dtype == torch.bool:
                key_padding_mask = mask
            else:
                key_padding_mask = (mask <= 0.5)

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)

        x = x + self.dropout_layer(attn_output)
        x = self.norm1(x)

        # Feedforward network
        ff_output = self.feedforward(x)
        x = x + self.dropout_layer(ff_output)
        x = self.norm2(x)

        return x


class TSPVAE_Decoder(nn.Module):
    """
    Decoder module for the TSPVAE model.

    Args:
        latent_dim (int): Latent space dimension.
        hidden_dim (int): Hidden feature dimension.
        output_dim (int): Output feature dimension.
        num_layers (int): Number of TransformerDecoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        max_sequence_length (int): Maximum sequence length.
        ar_mode (bool): If True, apply causal mask to decoder targets (AR behavior).
        pe_type (str): Positional encoding type: 'sinusoidal' | 'learnable' | 'none'.
        pe_dropout (float): Dropout applied after adding positional encodings.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, max_sequence_length,
                 dim_feedforward=512, ar_mode: bool = False, pe_type: str = 'sinusoidal', pe_dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.ar_mode = bool(ar_mode)
        self.pe_type = (pe_type or 'sinusoidal').lower()
        self.pe_dropout = float(pe_dropout) if pe_dropout is not None else 0.0

        # Positional encodings configuration
        # Remove conflicting attribute assignment; buffers/params are registered below as needed
        self.positional_encoding_param = None
        self.register_buffer('fourier_freqs', None, persistent=False)
        if self.pe_type == 'sinusoidal':
            pe = self.sinusoidal_positional_encoding(max_sequence_length, hidden_dim)
            self.register_buffer('positional_encoding', pe)
        elif self.pe_type == 'learnable':
            self.positional_encoding_param = nn.Parameter(torch.zeros(max_sequence_length, hidden_dim))
            nn.init.normal_(self.positional_encoding_param, mean=0.0, std=0.02)
        elif self.pe_type == 'fourier':
            # Random Fourier features (fixed) following Tancik et al.
            d_half = max(1, hidden_dim // 2)
            freqs = torch.randn(d_half)  # standard normal frequencies
            self.register_buffer('fourier_freqs', freqs)
        elif self.pe_type == 'none':
            pass
        else:
            pe = self.sinusoidal_positional_encoding(max_sequence_length, hidden_dim)
            self.register_buffer('positional_encoding', pe)
            self.pe_type = 'sinusoidal'

        self.pos_drop = nn.Dropout(self.pe_dropout) if self.pe_dropout > 0.0 else nn.Identity()
        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Project latent samples to hidden dimension
        self.latent_scalar_proj = nn.Linear(1, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Learned queries for output tokens (length = max_sequence_length)
        self.query_embed = nn.Embedding(max_sequence_length, hidden_dim)

        # Cross-attentive Transformer decoder (queries attend to latent memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def _fourier_pe(self, T: int, dtype: torch.dtype, device: torch.device):
        # Build [T, hidden_dim] using sin/cos of random frequencies
        d_half = max(1, self.hidden_dim // 2)
        # positions [T,1]
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
        freqs = self.fourier_freqs
        if freqs is None:
            # Fallback in case buffer not set (shouldn't happen normally)
            freqs = torch.randn(d_half, device=device, dtype=dtype)
        else:
            freqs = freqs.to(device=device, dtype=dtype)
        angles = t @ freqs.unsqueeze(0)  # [T, d_half]
        sin_comp = torch.sin(angles)
        cos_comp = torch.cos(angles)
        if self.hidden_dim % 2 == 0:
            pe = torch.cat([sin_comp, cos_comp], dim=-1)
        else:
            # If odd dim, pad last column with zeros
            pe = torch.cat([sin_comp, cos_comp, torch.zeros(T, 1, device=device, dtype=dtype)], dim=-1)
        return pe[:, :self.hidden_dim]

    def _get_pe(self, T: int, dtype: torch.dtype, device: torch.device):
        if self.pe_type == 'sinusoidal' and self.positional_encoding is not None:
            return self.positional_encoding[:T].to(device=device, dtype=dtype)
        if self.pe_type == 'learnable' and self.positional_encoding_param is not None:
            return self.positional_encoding_param[:T].to(device=device, dtype=dtype)
        if self.pe_type == 'fourier':
            return self._fourier_pe(T, dtype=dtype, device=device)
        return None

    def forward(self, context, patched_mask=None):
        """
        Forward pass for the decoder with cross-attention over latent memory.

        Args:
            context (torch.Tensor): Latent sequence z of shape (batch_size, T, latent_dim | 1) or pre-embedded memory (batch_size, T, hidden_dim).
            patched_mask (torch.Tensor, optional): Mask over time steps (1 for valid, 0 for padded) with shape (batch_size, T).

        Returns:
            torch.Tensor: Output tokens of shape (T, batch_size, output_dim).
        """
        device = context.device
        # Infer batch and sequence length from context only
        bs, T = context.size(0), context.size(1)

        # Build memory from latent context
        if context.dim() == 3:
            if context.size(-1) == self.hidden_dim:
                memory = context
            elif context.size(-1) == 1:
                memory = self.latent_scalar_proj(context)
            else:
                memory = self.latent_proj(context)
        else:
            # [B, T] -> [B, T, 1] then project
            if context.dim() == 2:
                context = context.unsqueeze(-1)
            memory = self.latent_scalar_proj(context)

        # Add positional encoding to memory (time-aware keys/values)
        pe = self._get_pe(T, dtype=memory.dtype, device=memory.device)
        if pe is not None:
            memory = self.pos_drop(memory + pe.unsqueeze(0))

        # Build queries for decoder outputs: take first T query embeddings
        queries = self.query_embed.weight[:T].unsqueeze(0).expand(bs, T, -1)
        if pe is not None:
            queries = self.pos_drop(queries + pe.unsqueeze(0).to(dtype=queries.dtype))

        # Prepare masks
        memory_kpm = None
        if patched_mask is not None:
            memory_kpm = (patched_mask == 0)

        tgt_mask = None
        if self.ar_mode:
            # Causal mask: allow attending to self and past positions only
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            # PyTorch expects float mask with -inf for masked positions in TransformerDecoder
            tgt_mask = torch.zeros(T, T, device=device, dtype=memory.dtype)
            tgt_mask[causal] = float('-inf')

        # Run cross-attentive transformer decoder
        dec_output = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_kpm,
        )  # [B, T, H]

        out = self.out_proj(dec_output).permute(1, 0, 2)  # [T, B, output_dim]
        return out

    def sinusoidal_positional_encoding(self, sequence_length, hidden_dim, device=None):
        """
        Create sinusoidal positional encoding.

        Args:
            sequence_length (int): Maximum sequence length.
            hidden_dim (int): Dimension of the embeddings.
            device (str | torch.device | None): Device to create the tensor on.
        Returns:
            torch.Tensor: Positional encodings of shape (sequence_length, hidden_dim).
        """
        if device is None:
            device = torch.device('cpu')
        dtype = torch.get_default_dtype()

        position = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
            * (-(math.log(10000.0) / hidden_dim))
        )
        pe = torch.zeros(sequence_length, hidden_dim, device=device, dtype=dtype)
        # Even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices: slice div_term to match column count when hidden_dim is odd
        cos_cols = pe[:, 1::2].shape[1]
        if cos_cols > 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :cos_cols]
        return pe


class TSPVAE(nn.Module):
    """
    TSPVAE model combining the encoder and decoder.

    Args:
        args (Namespace): Configuration arguments.
        patch_embedder (nn.Module, optional): Pretrained patch embedder.
    """

    def __init__(self, args, patch_embedder=None, freeze_embedder=True):
        super().__init__()
        self.latent_dim = args.latent_dim
        self.enc_hidden_dim = args.TSP_encoder_hidden_dim
        self.dec_hidden_dim = args.TSP_decoder_hidden_dim
        self.sequence_length = args.sequence_length
        self.patch_length = args.patch_length
        self.n_channels = args.input_dim  # Use input_dim from args

        # Store embedder configuration
        self.embedder = patch_embedder
        self.use_embedder = patch_embedder is not None

        if self.use_embedder:
            # Use pre-trained embedder
            for param in self.embedder.parameters():
                param.requires_grad = not freeze_embedder
                self.embedder.eval()

            # Dimensions based on embedder
            encoder_input_dim = self.embedder.config.d_model * self.n_channels
            decoder_output_dim = self.n_channels * self.embedder.config.d_model
            n_patches = self.sequence_length // args.patch_length
        else:
            # No embedder: use direct patchification
            # Input will be patchified from [BS, T, C] -> [BS, T//P, C*P]
            encoder_input_dim = self.n_channels * self.patch_length
            decoder_output_dim = self.n_channels * self.patch_length
            n_patches = self.sequence_length // self.patch_length

        self.encoder = TSPVAE_Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=args.TSP_encoder_hidden_dim,
            latent_dim=args.latent_dim,
            sequence_length=n_patches
        )

        self.decoder = TSPVAE_Decoder(
            latent_dim=args.latent_dim,
            hidden_dim=args.TSP_decoder_hidden_dim,
            output_dim=decoder_output_dim,
            num_layers=args.DecoderNumLayers,
            num_heads=args.DecoderNumHeads,
            dropout=0.1,
            max_sequence_length=n_patches,
            dim_feedforward=args.DecoderFFDim
        )

        self.smoothing = nn.Sequential(
            nn.Conv1d(self.n_channels, self.n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.n_channels, self.n_channels, kernel_size=3, padding=1)
        )

    def patchify_data(self, x, irrelevant_mask=None):
        """
        Patchify input data when no embedder is available.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, sequence_length)
            irrelevant_mask (torch.Tensor, optional): Mask tensor

        Returns:
            tuple: (patchified_data, patched_mask)
        """
        batch_size, n_channels, seq_len = x.shape
        n_patches = seq_len // self.patch_length

        # Reshape to patches: [BS, C, T] -> [BS, C, n_patches, patch_length] -> [BS, n_patches, C*patch_length]
        x_patches = x[:, :, :n_patches * self.patch_length]  # Trim to exact multiple
        x_patches = x_patches.view(batch_size, n_channels, n_patches, self.patch_length)
        x_patches = x_patches.permute(0, 2, 1, 3)  # [BS, n_patches, C, patch_length]
        x_patches = x_patches.contiguous().view(batch_size, n_patches, n_channels * self.patch_length)

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

        Args:
            x_patches (torch.Tensor): Patchified tensor of shape [BS, n_patches, C*patch_length]

        Returns:
            torch.Tensor: Unpatchified tensor of shape [BS, C, T]
        """
        batch_size, n_patches, _ = x_patches.shape

        # Reshape back: [BS, n_patches, C*patch_length] -> [BS, n_patches, C, patch_length] -> [BS, C, T]
        x_patches = x_patches.view(batch_size, n_patches, self.n_channels, self.patch_length)
        x_patches = x_patches.permute(0, 2, 1, 3)  # [BS, C, n_patches, patch_length]
        x_original = x_patches.contiguous().view(batch_size, self.n_channels, n_patches * self.patch_length)

        return x_original

    def forward(self, x: torch.Tensor, irrelevant_mask=None):
        """
        Forward pass for the TSPVAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, sequence_length).
            irrelevant_mask (torch.Tensor, optional): Mask for irrelevant tokens.

        Returns:
            ModelOutput: Output containing reconstructed x, latent variables, and losses.
        """
        device = x.device

        if self.use_embedder:
            # Use pre-trained embedder
            # PatchTST expects input shape [batch_size, sequence_length, channels]
            # but we have [batch_size, channels, sequence_length], so we need to transpose
            x_for_embedder = x.permute(0, 2, 1)  # [batch_size, sequence_length, channels]

            # Adjust irrelevant_mask to match the transposed input
            if irrelevant_mask is not None:
                # irrelevant_mask is [batch_size, sequence_length, channels]
                # PatchTST expects [batch_size, sequence_length, channels], so it's already correct
                embedder_mask = irrelevant_mask
            else:
                embedder_mask = None

            embedder_out = self.embedder(x_for_embedder, irrelevant_mask=embedder_mask)
            hidden_state = embedder_out.hidden_states[-1]
            x_patchified = hidden_state.permute(0, 2, 1, 3).contiguous().view(
                x.size(0), hidden_state.size(2), -1).to(device)

            if irrelevant_mask is not None:
                patched_mask = embedder_out.patched_mask[:, 0, :, :].all(dim=-1).to(device)
            else:
                patched_mask = None
        else:
            # Use direct patchification
            x_patchified, patched_mask = self.patchify_data(x, irrelevant_mask)
            x_patchified = x_patchified.to(device)
            if patched_mask is not None:
                patched_mask = patched_mask.to(device)

        # Encode
        mu, L = self.encoder(x_patchified, patched_mask=patched_mask)
        # Deterministic during eval, sample during training
        if self.training:
            z = sample_posterior(mu, L, patched_mask)
        else:
            z = mu

        # Decode only from latent context (no leakage from input)
        h_hat = self.decoder(
            context=z.unsqueeze(-1),
            patched_mask=patched_mask
        ).transpose(0, 1)

        if self.use_embedder:
            # Use embedder's head for reconstruction
            h_hat = h_hat.view(-1,
                               self.sequence_length // self.embedder.config.patch_length,
                               self.n_channels,
                               self.embedder.config.d_model).permute(0, 2, 1, 3)
            x_hat = self.embedder.head(h_hat).transpose(1, 2).reshape(-1, self.n_channels, self.sequence_length)
        else:
            # Direct unpatchification
            h_hat = h_hat.view(x.size(0), -1, self.n_channels * self.patch_length)
            x_hat = self.unpatchify_data(h_hat)
            # Apply temporal smoothing when no pre-trained embedder is used
            x_hat = self.smoothing(x_hat)

        # Compute losses
        KL_Loss = kl_divergence_isotropic(mu, L, sigma_p=torch.tensor(1, device=device), mask=patched_mask)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none')

        if irrelevant_mask is not None:
            reconstruction_loss = (reconstruction_loss * irrelevant_mask.permute(0, 2, 1)).sum(dim=[1]).mean(dim=[1])
        else:
            reconstruction_loss = reconstruction_loss.mean(dim=[1, 2])

        return ModelOutput(
            x_hat=x_hat,
            z=z,
            mu=mu,
            L=L,
            KL_Loss=KL_Loss,
            reconstruction_loss=reconstruction_loss
        )


if __name__ == "__main__":
    import argparse
    import numpy as np

    # Create a mock args configuration
    args = argparse.Namespace()
    args.latent_dim = 1
    args.TSP_encoder_hidden_dim = 32
    args.TSP_decoder_hidden_dim = 32
    args.sequence_length = 128
    args.patch_length = 16
    args.input_dim = 4
    args.DecoderNumLayers = 2
    args.DecoderNumHeads = 4
    args.DecoderFFDim = 64


    # Create a mock patch embedder class
    class MockPatchEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = argparse.Namespace()
            self.config.patch_length = args.patch_length
            self.config.d_model = 16
            self.head = nn.Linear(self.config.d_model, args.patch_length)

        def forward(self, x, irrelevant_mask=None):
            batch_size, n_channels, seq_len = x.shape
            n_patches = seq_len // self.config.patch_length

            # Create mock hidden states with correct dimensions
            hidden_states = [torch.randn(batch_size, n_channels, n_patches, self.config.d_model)]

            # Create mock patched mask with correct dimensions
            patched_mask = torch.ones(batch_size, n_channels, n_patches, 1, dtype=torch.float)
            if irrelevant_mask is not None:
                # Ensure irrelevant_mask has the right dimensions before processing
                if irrelevant_mask.size(1) != seq_len:
                    irrelevant_mask = irrelevant_mask.permute(0, 2, 1)  # permute to match expected dimensions

                # Reshape irrelevant_mask to match patch structure
                patched_mask = irrelevant_mask.reshape(batch_size, n_channels, n_patches, self.config.patch_length)
                patched_mask = patched_mask.float().mean(dim=-1, keepdim=True)
                patched_mask = (patched_mask > 0.5).float()

            return ModelOutput(hidden_states=hidden_states, patched_mask=patched_mask)


    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Testing TSPVAE model...")

    # Test 1: With embedder
    print("\n=== Test 1: With Pre-trained Embedder ===")
    patch_embedder = MockPatchEmbedder()
    model_with_embedder = TSPVAE(args, patch_embedder=patch_embedder)

    # Generate mock input data - match the expected format (batch_size, n_channels, seq_len)
    batch_size = 2
    n_channels = 4  # Should match model.n_channels
    mock_data = torch.randn(batch_size, n_channels, args.sequence_length)

    # Create a mask as float tensor - match the input dimensions
    mock_mask = torch.ones(batch_size, args.sequence_length, n_channels, dtype=torch.float)
    mock_mask[:, -10:, :] = 0  # Set the last 10 time steps to be irrelevant

    print("Running forward pass with embedder...")
    try:
        output = model_with_embedder(mock_data, irrelevant_mask=mock_mask)
        print(f"✓ Success! Output shapes:")
        print(f"  - x_hat: {output.x_hat.shape}, expected: {(batch_size, n_channels, args.sequence_length)}")
        print(f"  - z: {output.z.shape}, expected: {(batch_size, args.sequence_length // args.patch_length)}")
        print(f"  - mu: {output.mu.shape}, expected: {(batch_size, args.sequence_length // args.patch_length)}")
        print(f"  - L: {output.L.shape}, expected: {(batch_size, args.sequence_length // args.patch_length, args.sequence_length // args.patch_length)}")
        print(f"  - KL Loss: {output.KL_Loss.shape}, expected: {(batch_size,)}")
        print(f"  - Reconstruction Loss: {output.reconstruction_loss.shape}, expected: {(batch_size,)}")

        # Check if lower triangular property of L is maintained
        L_tril = torch.tril(output.L)
        is_lower_triangular = torch.allclose(output.L, L_tril)
        print(f"  - Is L lower triangular? {is_lower_triangular}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Without embedder (direct patchification)
    print("\n=== Test 2: Without Embedder (Direct Patchification) ===")
    model_no_embedder = TSPVAE(args, patch_embedder=None)

    print("Running forward pass without embedder...")
    try:
        output = model_no_embedder(mock_data, irrelevant_mask=mock_mask)
        print(f"✓ Success! Output shapes:")
        print(f"  - x_hat: {output.x_hat.shape}")
        print(f"  - z: {output.z.shape}")
        print(f"  - mu: {output.mu.shape}")
        print(f"  - L: {output.L.shape}")
        print(f"  - KL Loss: {output.KL_Loss.shape}")
        print(f"  - Reconstruction Loss: {output.reconstruction_loss.shape}")

        # Verify patchification worked correctly
        expected_patches = args.sequence_length // args.patch_length
        assert output.z.shape[1] == expected_patches, f"Expected {expected_patches} patches, got {output.z.shape[1]}"
        assert output.x_hat.shape == mock_data.shape, f"Output shape {output.x_hat.shape} doesn't match input {mock_data.shape}"
        print(f"✓ Patchification verification passed!")

        # Check if lower triangular property of L is maintained
        L_tril = torch.tril(output.L)
        is_lower_triangular = torch.allclose(output.L, L_tril)
        print(f"  - Is L lower triangular? {is_lower_triangular}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\nTSPVAE tests completed!")

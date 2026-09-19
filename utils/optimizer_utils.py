"""
Optimizer utilities for selective weight decay.

Best practice: exclude normalization layers (LayerNorm, BatchNorm) and bias
parameters from weight decay.  Regularising these parameters shrinks model
capacity without providing meaningful regularisation benefit.

Reference implementations: GPT-2, BERT, ViT, nanoGPT.
"""

import torch.nn as nn


def get_param_groups_with_weight_decay(model_or_params, weight_decay: float):
    """Split parameters into *decay* and *no-decay* groups for AdamW.

    Parameters that should **not** receive weight decay:
      - All ``bias`` parameters
      - ``LayerNorm`` weight and bias
      - ``BatchNorm`` weight and bias
      - ``Embedding`` weight

    Everything else (Linear / Conv weights, etc.) receives weight decay.

    Args:
        model_or_params: Either an ``nn.Module`` (uses ``named_parameters()``)
            or an iterable of ``(name, Parameter)`` tuples.
        weight_decay: The weight-decay value for the *decay* group.

    Returns:
        A list of two dicts suitable for ``torch.optim.AdamW(param_groups, lr=...)``.
    """
    # Module types whose *all* parameters should be excluded from decay.
    _NO_DECAY_MODULES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                         nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                         nn.Embedding)

    # Standalone nn.Parameter names (not belonging to a no-decay module) that
    # should also be excluded from weight decay. Includes learnable positional
    # encodings, start-of-sequence tokens, and learned scalar gates — all of
    # which are embedding-like or scale parameters that should not be shrunk.
    _NO_DECAY_PARAM_NAMES = {
        'positional_encoding_param',   # learnable PE in EnhancedTSPVAE/CVE
        'positional_encoding',         # generic positional encoding parameter name
        'pos_embed',                   # generic pos embed parameter name
        'sos_token',                   # start-of-sequence token in TSPCVAE decoder
        'residual_alpha',              # learned residual mixing scalar
        'codebook',                    # VQ/codebook parameter name
        'codebook_param',              # variant codebook parameter name
    }

    if isinstance(model_or_params, nn.Module):
        # Build a set of parameter ids that belong to no-decay module types.
        no_decay_param_ids = set()
        for module in model_or_params.modules():
            if isinstance(module, _NO_DECAY_MODULES):
                for param in module.parameters():
                    no_decay_param_ids.add(id(param))

        decay_params = []
        no_decay_params = []

        for name, param in model_or_params.named_parameters():
            if not param.requires_grad:
                continue
            # Extract the leaf attribute name (last segment of dotted path)
            leaf_name = name.split('.')[-1]
            # Exclude by module membership, by bias suffix, or by known embedding-like param names
            if (id(param) in no_decay_param_ids
                    or name.endswith('.bias')
                    or leaf_name in _NO_DECAY_PARAM_NAMES):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    else:
        # Fallback: iterable of (name, param) tuples – use name heuristics only.
        decay_params = []
        no_decay_params = []
        for name, param in model_or_params:
            if not param.requires_grad:
                continue
            if name.endswith('.bias') or 'layernorm' in name.lower() or 'ln' in name.lower() or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"Optimizer param groups: "
        f"{len(decay_params)} tensors ({n_decay:,} params) WITH weight_decay={weight_decay} | "
        f"{len(no_decay_params)} tensors ({n_no_decay:,} params) WITHOUT weight_decay"
    )

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def safe_load_optimizer_state(optimizer, state_dict):
    """Load optimizer state dict, handling param-group count mismatches.

    Old checkpoints may have been saved with a single param group (flat
    ``model.parameters()``), while the current optimizer uses two groups
    (decay / no-decay).  PyTorch's ``load_state_dict`` raises
    ``ValueError`` when the group counts differ.

    This helper catches that error and falls back to skipping the
    optimizer state entirely.  The only cost is losing momentum /
    adaptive-rate buffers, which rebuild within a few training steps.

    Args:
        optimizer: The current ``torch.optim.Optimizer`` instance.
        state_dict: The ``optimizer_state_dict`` loaded from a checkpoint.

    Returns:
        ``True`` if the state was loaded successfully, ``False`` if it
        was skipped due to a mismatch.
    """
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except (ValueError, RuntimeError) as e:
        saved_groups = len(state_dict.get('param_groups', []))
        current_groups = len(optimizer.param_groups)
        print(
            f"⚠️  Skipping optimizer state restore: checkpoint has "
            f"{saved_groups} param group(s) but current optimizer has "
            f"{current_groups}. Momentum buffers will rebuild in a few "
            f"steps. (Detail: {e})"
        )
        return False

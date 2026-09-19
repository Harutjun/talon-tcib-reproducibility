"""Train the TALON student (Stage 2): TALONStudent aligned to the frozen teacher's
posterior, for SWaT or WADI.

Usage: python training/train_student.py --dataset {SWaT,WADI} --vae_checkpoint
<path to a train_teacher.py checkpoint> [hyperparameter overrides]
See the corresponding results/*/config.json for the exact hyperparameters used to
produce the released numbers.
"""

import argparse
import os
import sys
import json
import math
import hashlib
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.spatial_benchmark_config import DATASET_CONFIGS, SHARED_MODEL_CONFIG, TRAINING_CONFIG
from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from models.TALONStudent import TALONStudent
from utils.mutual_information import resolve_mi_loss_weight
from utils.optimizer_utils import get_param_groups_with_weight_decay, safe_load_optimizer_state

class PriorCachedDataset(Dataset):
    """
    A Dataset wrapper that appends precomputed teacher priors to each sample batch.
    Defined at the module level to ensure serializability/picklability for multi-process DataLoader on Windows.
    """
    def __init__(self, dataset, priors):
        self.dataset = dataset
        self.priors = priors

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        prior = self.priors[idx]
        return batch + (prior,)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def precompute_priors(pretrained_tspvae, dataset, device, batch_size=512):
    print("Precomputing teacher priors for the entire training dataset...")
    pretrained_tspvae.eval()
    
    # Run loader with num_workers=0 to avoid multiprocessing overhead during precomputation
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_mu = []
    all_params = {}
    
    with torch.no_grad():
        for batch in loader:
            y = batch[1].transpose(1, 2).float().to(device)  # [B, T, C_y]
            
            T_full = pretrained_tspvae.sequence_length
            time_mask_full = torch.ones(y.size(0), T_full, device=device, dtype=torch.float)
            
            if hasattr(pretrained_tspvae, 'patchify_data'):
                y_patches, y_mask = pretrained_tspvae.patchify_data(y.transpose(1, 2), irrelevant_mask=None)
            else:
                y_patches, y_mask = y, None
                
            mu_prior, prior_params = pretrained_tspvae.encoder(y_patches, patched_mask=y_mask)
            
            all_mu.append(mu_prior.cpu())
            for k, v in prior_params.items():
                if torch.is_tensor(v):
                    if k not in all_params:
                        all_params[k] = []
                    all_params[k].append(v.cpu())
                else:
                    if k not in all_params:
                        all_params[k] = []
                    all_params[k].append(v)
                    
    final_mu = torch.cat(all_mu, dim=0)
    final_params = {}
    for k, v_list in all_params.items():
        if torch.is_tensor(v_list[0]):
            final_params[k] = torch.cat(v_list, dim=0)
        else:
            final_params[k] = v_list
            
    priors = []
    for i in range(len(dataset)):
        p_i = {k: v[i] if torch.is_tensor(v) else v for k, v in final_params.items()}
        priors.append((final_mu[i], p_i))
        
    print(f"Successfully precomputed priors for {len(dataset)} samples.")
    return priors

def forward_batch(model, x, y, device, prior_override=None):
    # LocalTSAD dataset loaders return shapes of [B, C, T]
    # TALONStudent conditioning encoder expects time-first format [B, T, C]
    x = x.transpose(1, 2).float().to(device, non_blocking=True)
    y = y.transpose(1, 2).float().to(device, non_blocking=True)
    
    batch_size = x.size(0)
    T = model.sequence_length
    time_mask_full = torch.ones(batch_size, T, device=device, dtype=torch.float)
    
    # Format and device-migrate prior override
    if prior_override is not None:
        mu_prior, prior_params = prior_override
        mu_prior = mu_prior.to(device)
        prior_params = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prior_params.items()}
        prior_override = (mu_prior, prior_params)
    
    # We call forward pass using raw coordinate tensors
    output = model(
        x_condition=x,
        y_patches=None if prior_override is not None else y,
        mode="train",
        y_target=y,
        time_mask_full=time_mask_full,
        prior_override=prior_override
    )
    return output

def log_reconstructions(model, loader, device, writer, epoch, num_samples=3, num_channels=None):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from torchvision.transforms.functional import to_tensor

    dataset = loader.dataset
    underlying = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    # If the underlying dataset is a Subset or wrapped class, go deeper
    while hasattr(underlying, 'dataset'):
        underlying = underlying.dataset
        
    feature_names = underlying.metadata.get('feature_names', None) if hasattr(underlying, 'metadata') else None
    y_indices = getattr(underlying, 'y_indices', None)
    y_discrete_mask = getattr(underlying, 'y_discrete_mask', None)

    model.eval()
    plotted = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            prior_override = batch[5] if len(batch) > 5 else None
            
            output = forward_batch(model, x, y, device, prior_override=prior_override)
            y_hat = output.y_hat_condition
            
            batch_size, channels, seq_len = y.shape
            channels_to_plot = channels if num_channels is None else min(channels, num_channels)
            for i in range(min(batch_size, num_samples - plotted)):
                for c in range(channels_to_plot):
                    target = y[i, c].cpu().numpy()
                    recon = y_hat[i, c].cpu().numpy() if y_hat.shape[1] == channels else y_hat[i, :, c].cpu().numpy()
                    
                    col_name = f"Channel_{c}"
                    is_disc = False
                    if feature_names is not None and y_indices is not None and c < len(y_indices):
                        col_name = feature_names[y_indices[c]]
                    if y_discrete_mask is not None and c < len(y_discrete_mask):
                        is_disc = bool(y_discrete_mask[c])
                        
                    desc = " [Discrete]" if is_disc else " [Continuous]"
                    
                    t = np.arange(len(target))
                    plt.figure(figsize=(6, 3))
                    plt.plot(t, target, label='Target', color='black')
                    plt.plot(t, recon, label='Reconstruction', color='red', alpha=0.7)
                    plt.title(f'Sample {plotted} - {col_name}{desc}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.tight_layout()
                    
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = plt.imread(buf)
                    plt.close()
                    
                    image_tensor = to_tensor(image)
                    tb_tag = f'ValReconstructions/Sample{plotted}_{col_name}'
                    if is_disc:
                        tb_tag += '_Discrete'
                    writer.add_image(tb_tag, image_tensor, global_step=epoch)
                plotted += 1
                
            if plotted >= num_samples:
                break

def _get_mi_metrics(output):
    mi_estimate = getattr(output, 'mi_estimate', None)
    mi_loss = getattr(output, 'mi_loss', None)
    mi_enabled = mi_estimate is not None or mi_loss is not None
    return mi_enabled, mi_estimate, mi_loss

def compute_loss(output, kl_weight, recon_weight, num_target_elements=None):
    kl_loss = output.KL_div.mean()

    # Scale factor converts per-element mean MSE back to full sequence + channel sum
    scale_factor = float(num_target_elements) if num_target_elements is not None else 1.0

    recon_loss_raw = output.reconstruction_loss_condition
    if recon_loss_raw is None:
        recon_loss = torch.zeros((), device=kl_loss.device, dtype=kl_loss.dtype)
    else:
        recon_loss = (recon_loss_raw * scale_factor).mean()

    recon_loss_target_raw = output.reconstruction_loss_target
    if recon_loss_target_raw is None:
        recon_loss_target = torch.zeros((), device=kl_loss.device, dtype=kl_loss.dtype)
    else:
        recon_loss_target = (recon_loss_target_raw * scale_factor).mean()

    total_loss = kl_weight * kl_loss + recon_weight * recon_loss

    mi_enabled, mi_estimate, mi_loss = _get_mi_metrics(output)
    if mi_loss is not None:
        total_loss = total_loss + mi_loss

    return {
        'total_loss': total_loss,
        'kl_loss': kl_loss,
        'reconstruction_loss': recon_loss,
        'reconstruction_loss_target': recon_loss_target,
        'mi_loss': mi_loss,
        'mi_estimate': mi_estimate,
        'mi_enabled': mi_enabled
    }

def run_epoch(model, loader, device, optimizer=None, scaler=None, kl_weight=1.0, recon_weight=1.0, max_batches=None, grad_clip_norm=1.0):
    is_train = optimizer is not None
    model.pretrained_model.eval()
    if any(p.requires_grad for p in model.pretrained_model.parameters()):
        raise RuntimeError('The Stage 2 teacher must remain frozen.')
    model.conditioning_encoder.train() if is_train else model.conditioning_encoder.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_recon_target = 0.0
    total_kl = 0.0
    total_mi_estimate = 0.0
    total_mi_loss = 0.0
    n_batches = 0
    mi_batches = 0
    mi_enabled = bool(getattr(model, 'compute_mi', False))

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = batch[0], batch[1]
        prior_override = None
        if len(batch) > 5:
            prior_override = batch[5]
        elif len(batch) >= 4 and isinstance(batch[3], (list, tuple)):
            prior_override = batch[3]

        # Compute total target elements (channels * sequence_length) for un-normalized sum scaling
        num_target_elements = y.size(1) * y.size(2) if y.dim() == 3 else y.numel() // y.size(0)

        if is_train:
            optimizer.zero_grad()

        # Automatic Mixed Precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = forward_batch(model, x, y, device, prior_override=prior_override)
                loss_dict = compute_loss(output, kl_weight, recon_weight, num_target_elements=num_target_elements)
                loss = loss_dict['total_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite Stage 2 loss; stopping before an optimizer update.')
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
        else:
            output = forward_batch(model, x, y, device, prior_override=prior_override)
            loss_dict = compute_loss(output, kl_weight, recon_weight, num_target_elements=num_target_elements)
            loss = loss_dict['total_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite Stage 2 loss; stopping before an optimizer update.')
            
            if is_train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm
                    )
                optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict['reconstruction_loss'].item()
        total_recon_target += loss_dict['reconstruction_loss_target'].item()
        total_kl += loss_dict['kl_loss'].item()
        if loss_dict['mi_loss'] is not None:
            total_mi_loss += loss_dict['mi_loss'].item()
        if loss_dict['mi_estimate'] is not None:
            total_mi_estimate += loss_dict['mi_estimate'].item()
            mi_batches += 1
        n_batches += 1

    denom = max(1, n_batches)
    mi_denom = max(1, mi_batches)
    return {
        'loss': total_loss / denom,
        'recon': total_recon / denom,
        'recon_target': total_recon_target / denom,
        'kl': total_kl / denom,
        'mi': total_mi_estimate / mi_denom,
        'mi_loss_term': total_mi_loss / mi_denom,
        'mi_enabled': mi_enabled
    }

def save_checkpoint(model, optimizer, epoch, best_val, path):
    torch.save(
        {
            'epoch': epoch,
            'best_val_loss': best_val,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'teacher_provenance': getattr(model, 'teacher_provenance', None),
        },
        path,
    )

def main():
    parser = argparse.ArgumentParser(description="Train TALONStudent on WADI")
    parser.add_argument('--dataset', type=str, default='WADI')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to the pre-trained TALONTeacher checkpoint')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'])
    parser.add_argument('--stride', type=int, default=None, help='Override dataset sliding window stride')
    parser.add_argument('--sampling_rate_seconds', type=int, default=None,
                        help='Override dataset sampling rate / downsample rate in seconds')
    parser.add_argument('--learning_rate', type=float, default=TRAINING_CONFIG['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default=TRAINING_CONFIG['weight_decay'])
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='results/wadi_cve')
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Max rows to read from each CSV (for fast iteration)')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    
    # Loss weight parameters matching stage 2 objectives
    parser.add_argument('--kl_weight', type=float, default=TRAINING_CONFIG.get('beta', 1.0))
    parser.add_argument('--recon_weight', type=float, default=10.0)
    parser.add_argument('--mi_loss_weight', type=str, default=TRAINING_CONFIG.get('mi_loss_weight', 'auto'))
    parser.add_argument('--kl_direction', type=str, default='reverse', choices=['reverse', 'forward'])
    parser.add_argument('--mi_estimator', type=str, default='barber_agakov', choices=['barber_agakov', 'infonce'],
                        help='Mutual information estimator: barber_agakov (default) or legacy infonce')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'yogi'],
                        help='Optimizer choice: adamw (default) or yogi')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Beta parameters for AdamW / Yogi optimizer (e.g., --betas 0.9 0.999 or --betas 0.7 0.5)')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Numerical epsilon for optimizer (increase to 1e-4 or 1e-3 to stabilize noisy late-stage gradients)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Maximum gradient norm threshold for gradient clipping')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='When resuming from a checkpoint, reset optimizer momentum state to apply new optimizer options or learning rate cleanly')
    # Checkpoint / logging frequency. Defaults reproduce the original behaviour exactly
    # (latest checkpoint every epoch, reconstruction figures every 50 epochs). Raising
    # --ckpt_every is purely an I/O optimisation for long runs: it changes how often
    # weights are persisted, never the trained model or the optimisation trajectory.
    parser.add_argument('--ckpt_every', type=int, default=1,
                        help="Save latest_cve.pth every N epochs (and on the final epoch). "
                             "Does not affect best_cve.pth, which is always saved on improvement.")
    parser.add_argument('--log_recon_every', type=int, default=50,
                        help="Log reconstruction figures every N epochs; 0 disables them.")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--target_val_loss', type=float, default=None,
                        help='Stop training once validation loss is <= this threshold (after KL annealing).')

    # Student encoder capacity overrides
    parser.add_argument('--encoder_blocks', type=int, default=None,
                        help='Override number of Transformer encoder blocks for the student (e.g., 4 or 6)')
    parser.add_argument('--dim_feedforward', type=int, default=None,
                        help='Override student feedforward dimension (e.g., 512 or 1024)')
    parser.add_argument('--enc_hidden_dim', type=int, default=None,
                        help='Override student encoder hidden dimension (e.g., 128)')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Override number of attention heads for the student encoder (e.g., 8)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout probability for the student encoder (e.g., 0.05)')

    args = parser.parse_args()
    os.environ['MI_ESTIMATOR_TYPE'] = args.mi_estimator
    print(f"Mutual Information Estimator selected: {args.mi_estimator}")

    # Load base dataset and model configurations
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_cfg = DATASET_CONFIGS[args.dataset].copy()
    teacher_model_cfg = SHARED_MODEL_CONFIG.copy()

    set_seed(args.seed)

    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.smoke_test:
        args.epochs = 2
        args.max_train_batches = 2
        args.max_val_batches = 2
        if args.max_rows is None:
            args.max_rows = 5000

    mi_loss_weight = resolve_mi_loss_weight(args.mi_loss_weight, args.kl_weight)
    compute_mi = mi_loss_weight > 0.0

    # Try to load companion config.json for the pretrained VAE to ensure architectural alignment
    vae_dir = os.path.dirname(args.vae_checkpoint)
    config_path = os.path.join(vae_dir, 'config.json')
    if os.path.exists(config_path):
        print(f"Found VAE config file: {config_path}. Overriding teacher model and dataset configuration.")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                vae_config = json.load(f)
                if 'model' in vae_config:
                    # Merge model configuration for teacher
                    for k, v in vae_config['model'].items():
                        teacher_model_cfg[k] = v
                if 'dataset' in vae_config:
                    # Preserve the exact feature selection and preprocessing used by
                    # this teacher, including overrides and dropped columns.
                    dataset_cfg.update(vae_config['dataset'])
        except Exception as e:
            raise RuntimeError(f'Cannot load teacher configuration: {config_path}') from e

    # Build student model configuration (inherits from teacher_model_cfg with optional overrides)
    student_model_cfg = teacher_model_cfg.copy()
    student_encoder_kwargs = (teacher_model_cfg.get('encoder_kwargs') or {}).copy()

    if args.encoder_blocks is not None:
        student_encoder_kwargs['transformer_encoder_blocks'] = args.encoder_blocks
        print(f"Student Architecture Override: transformer_encoder_blocks = {args.encoder_blocks}")
    if args.dim_feedforward is not None:
        student_encoder_kwargs['dim_feedforward'] = args.dim_feedforward
        print(f"Student Architecture Override: dim_feedforward = {args.dim_feedforward}")
    if args.enc_hidden_dim is not None:
        student_model_cfg['enc_hidden_dim'] = args.enc_hidden_dim
        print(f"Student Architecture Override: enc_hidden_dim = {args.enc_hidden_dim}")
    if args.n_heads is not None:
        student_encoder_kwargs['n_heads'] = args.n_heads
        print(f"Student Architecture Override: n_heads = {args.n_heads}")
    if args.dropout is not None:
        student_encoder_kwargs['dropout'] = args.dropout
        print(f"Student Architecture Override: dropout = {args.dropout}")

    student_model_cfg['encoder_kwargs'] = student_encoder_kwargs

    if getattr(args, 'stride', None) is not None:
        dataset_cfg['stride'] = args.stride
    if getattr(args, 'sampling_rate_seconds', None) is not None:
        dataset_cfg['sampling_rate_seconds'] = args.sampling_rate_seconds
        print(f"Dataset Override: sampling_rate_seconds = {args.sampling_rate_seconds}")

    # Load dataset structures
    train_dataset, test_dataset = load_csv_dataset(
        data_root=dataset_cfg['data_root'],
        normal_csv=dataset_cfg['normal_csv'],
        attack_csv=dataset_cfg['attack_csv'],
        label_column=dataset_cfg['label_column'],
        timestamp_columns=dataset_cfg['timestamp_columns'],
        x_prefixes=dataset_cfg['x_prefixes'],
        y_prefixes=dataset_cfg['y_prefixes'],
        window_size=dataset_cfg['window_size'],
        stride=dataset_cfg['stride'],
        x_override=dataset_cfg.get('x_override'),
        y_override=dataset_cfg.get('y_override'),
        max_rows=args.max_rows,
        downsample_rate=dataset_cfg.get('sampling_rate_seconds', 1),
        scaler_type=dataset_cfg.get('scaler_type', 'minmax'),
        downsample_mode=dataset_cfg.get('downsample_mode', 'median'),
        drop_columns=dataset_cfg.get('drop_columns', [])
    )

    if train_dataset is None or test_dataset is None:
        raise RuntimeError("Failed to load dataset. Check paths and CSV names.")

    print("\n" + "="*50)
    print("WADI CVE Student Training Pipeline Commencing")
    print(f"VAE Checkpoint: {args.vae_checkpoint}")
    print(f"Device: {device}")
    print(f"KL Weight (beta): {args.kl_weight}")
    print(f"MI Weight: {mi_loss_weight} (config: {args.mi_loss_weight})")
    print(f"AMP: {'disabled' if args.no_amp else 'enabled'}")
    print("="*50 + "\n")

    # Resolve features dimensions dynamically from train_dataset indices
    x_dim = len(train_dataset.x_indices)
    y_dim = len(train_dataset.y_indices)

    # Initialize pre-trained VAE model first (using teacher's exact configuration)
    pretrained_tspvae = TALONTeacher(
        latent_dim=teacher_model_cfg['latent_dim'],
        input_dim=y_dim,
        sequence_length=dataset_cfg['window_size'],
        patch_length=dataset_cfg['patch_length'],
        enc_hidden_dim=teacher_model_cfg['enc_hidden_dim'],
        dec_hidden_dim=teacher_model_cfg['dec_hidden_dim'],
        gp_time_kernel=teacher_model_cfg['gp_time_kernel'],
        rank_c=teacher_model_cfg['rank_c'],
        gp_jitter=teacher_model_cfg['gp_jitter'],
        bandwidth=teacher_model_cfg['bandwidth'],
        posterior_tc_banded=teacher_model_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=teacher_model_cfg['tc_channel_bandwidth'],
        encoder_kwargs=teacher_model_cfg['encoder_kwargs'],
        decoder_kwargs=teacher_model_cfg['decoder_kwargs'],
        discrete_mask=train_dataset.y_discrete_mask,
        bce_loss_weight=teacher_model_cfg.get('bce_loss_weight', 1.0)
    )

    # Load weights
    print(f"Loading pre-trained VAE weights from {args.vae_checkpoint}...")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    pretrained_tspvae.load_state_dict(vae_ckpt['model_state_dict'])
    pretrained_tspvae.to(device)
    with open(args.vae_checkpoint, 'rb') as checkpoint_file:
        teacher_hash = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(checkpoint_file.read()).hexdigest()
    teacher_provenance = {
        'path': os.path.abspath(args.vae_checkpoint),
        'sha256': teacher_hash,
        'epoch': vae_ckpt.get('epoch'),
        'best_val_loss': vae_ckpt.get('best_val_loss'),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as config_file:
        json.dump({'dataset': dataset_cfg, 'model': student_model_cfg,
                   'teacher_model': teacher_model_cfg, 'teacher_provenance': teacher_provenance,
                   'train_args': vars(args), 'resolved_mi_loss_weight': mi_loss_weight},
                  config_file, indent=2)
    print(f'Teacher verified: epoch={teacher_provenance["epoch"]}, SHA256={teacher_hash}', flush=True)
    print(f'Feature dimensions: X={x_dim}, Y={y_dim}; training windows={len(train_dataset)}', flush=True)

    # Freeze pre-trained components
    for p in pretrained_tspvae.parameters():
        p.requires_grad = False

    # Precompute priors for the entire train_dataset using pre-trained VAE
    priors = precompute_priors(pretrained_tspvae, train_dataset, device)
    
    # Wrap train_dataset with our picklable PriorCachedDataset wrapper
    wrapped_train_dataset = PriorCachedDataset(train_dataset, priors)

    # Perform splits on wrapped dataset
    val_size = int(len(train_dataset) * args.val_split)
    train_size = max(1, len(train_dataset) - val_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_split, val_split = random_split(wrapped_train_dataset, [train_size, val_size], generator=generator)

    # Create optimized data loaders
    use_pin = (device.type == 'cuda')
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin)

    # Initialize TALONStudent conditioning student (using student_model_cfg with customized architecture)
    model = TALONStudent(
        pretrained_tspvae=pretrained_tspvae,
        conditioning_input_dim=x_dim,
        enc_hidden_dim=student_model_cfg['enc_hidden_dim'],
        dec_hidden_dim=student_model_cfg['dec_hidden_dim'],
        posterior_tc_banded=student_model_cfg['tc_banded_enabled'],
        bandwidth=student_model_cfg['bandwidth'],
        tc_channel_bandwidth=student_model_cfg['tc_channel_bandwidth'],
        encoder_kwargs=student_model_cfg['encoder_kwargs'],
        kl_direction=args.kl_direction,
        compute_mi=compute_mi,
        mi_loss_weight=mi_loss_weight
    )
    model.to(device)
    model.teacher_provenance = teacher_provenance

    # Only optimize student's parameters
    opt_params = get_param_groups_with_weight_decay(
        model, weight_decay=args.weight_decay
    )
    betas_tuple = tuple(args.betas)
    if args.optimizer == 'yogi':
        try:
            import torch_optimizer as optim_ext
            optimizer = optim_ext.Yogi(opt_params, lr=args.learning_rate, betas=betas_tuple, eps=args.eps)
            print(f"Using Yogi optimizer (lr={args.learning_rate}, betas={betas_tuple}, eps={args.eps})")
        except ImportError:
            print("'torch_optimizer' module not found. Install via 'pip install torch-optimizer'. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate, betas=betas_tuple, eps=args.eps)
    else:
        optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate, betas=betas_tuple, eps=args.eps)
        print(f"Using AdamW optimizer (lr={args.learning_rate}, betas={betas_tuple}, eps={args.eps})")

    # Initialize mixed precision GradScaler
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and not args.no_amp) else None

    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize TensorBoard SummaryWriter
    tb_log_dir = os.path.join("runs", f"{args.dataset}_CVE_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=tb_log_dir)

    best_val_loss = float('inf')
    start_epoch = 1

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            if getattr(args, 'vae_checkpoint', None):
                # Retain the freshly loaded teacher weights from --vae_checkpoint; only resume student weights
                student_keys = {k: v for k, v in state_dict.items() if not k.startswith(('pretrained_model', 'pretrained_decoder', 'pretrained_encoder', 'pretrained_gp_prior'))}
                model.load_state_dict(student_keys, strict=False)
                print("Retained frozen teacher weights from --vae_checkpoint and resumed student weights from checkpoint.")
            else:
                model.load_state_dict(state_dict)
            if 'optimizer_state_dict' in checkpoint and not args.reset_optimizer:
                safe_load_optimizer_state(optimizer, checkpoint['optimizer_state_dict'])
            elif args.reset_optimizer:
                print("'--reset_optimizer' flag set. Resetting optimizer momentum buffers while keeping model weights.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Loaded model weights from checkpoint. Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
        else:
            print(f"Checkpoint path '{args.resume}' not found. Starting training from scratch.")

    kl_anneal_epochs = TRAINING_CONFIG.get('kl_annealing_epochs', 50)

    for epoch in range(start_epoch, args.epochs + 1):
        effective_kl_weight = min(args.kl_weight, args.kl_weight * (float(epoch) / float(kl_anneal_epochs)))

        train_metrics = run_epoch(
            model, train_loader, device, optimizer, scaler=scaler,
            kl_weight=effective_kl_weight, recon_weight=args.recon_weight,
            max_batches=args.max_train_batches, grad_clip_norm=args.grad_clip_norm
        )
        
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, device, None, scaler=scaler,
                kl_weight=effective_kl_weight, recon_weight=args.recon_weight,
                max_batches=args.max_val_batches
            )

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon']:.4f}, ReconTarg: {train_metrics['recon_target']:.4f}, KL: {train_metrics['kl']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon']:.4f}, ReconTarg: {val_metrics['recon_target']:.4f}, KL: {val_metrics['kl']:.4f})")
        print(f"MI Term: {'enabled' if train_metrics['mi_enabled'] else 'disabled'} | "
              f"Train Estimate: {train_metrics['mi']:.4f} | Train Loss Term: {train_metrics['mi_loss_term']:.4f} | "
              f"Val Estimate: {val_metrics['mi']:.4f} | Val Loss Term: {val_metrics['mi_loss_term']:.4f}")

        # Log to Tensorboard
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        writer.add_scalar('Reconstruction/Train', train_metrics['recon'], epoch)
        writer.add_scalar('Reconstruction/Val', val_metrics['recon'], epoch)
        writer.add_scalar('KL/Train', train_metrics['kl'], epoch)
        writer.add_scalar('KL/Val', val_metrics['kl'], epoch)
        writer.add_scalar('MI/Enabled', 1.0 if train_metrics['mi_enabled'] else 0.0, epoch)
        writer.add_scalar('MI/TrainEstimate', train_metrics['mi'], epoch)
        writer.add_scalar('MI/ValEstimate', val_metrics['mi'], epoch)
        writer.add_scalar('MI/TrainLossTerm', train_metrics['mi_loss_term'], epoch)
        writer.add_scalar('MI/ValLossTerm', val_metrics['mi_loss_term'], epoch)
        if args.log_recon_every > 0 and (epoch % args.log_recon_every == 0 or epoch == args.epochs):
            log_reconstructions(model, val_loader, device, writer, epoch)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(model, optimizer, epoch, best_val_loss, os.path.join(args.output_dir, 'best_cve.pth'))
            print(f"New best validation loss! Saved model to best_cve.pth")

        # Save the latest checkpoint to resume from if interrupted. Written every
        # --ckpt_every epochs (default 1 = every epoch, the original behaviour) and
        # always on the final epoch, so a completed run is never missing its tail state.
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], os.path.join(args.output_dir, 'latest_cve.pth'))

        if args.target_val_loss is not None and val_metrics['loss'] <= args.target_val_loss and epoch >= kl_anneal_epochs:
            print(f"Target validation loss {args.target_val_loss} reached ({val_metrics['loss']:.4f} <= {args.target_val_loss}) at epoch {epoch}! Stopping training early.")
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], os.path.join(args.output_dir, 'latest_cve.pth'))
            break

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()

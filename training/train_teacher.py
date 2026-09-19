"""Train the TALON teacher (Stage 1): TALONTeacher on Y alone, for SWaT or WADI.

Usage: python training/train_teacher.py --dataset {SWaT,WADI} [hyperparameter overrides]
See the corresponding results/*/config.json for the exact hyperparameters used to
produce the released numbers.
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


class NoOpWriter:
    """Drop-in stand-in for SummaryWriter that does nothing, for --no_tensorboard runs."""
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass

import sys

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.spatial_benchmark_config import DATASET_CONFIGS, SHARED_MODEL_CONFIG, TRAINING_CONFIG
from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from utils.EnhancedTSPLossFunctions import enhanced_tspvae_loss
from utils.optimizer_utils import get_param_groups_with_weight_decay, safe_load_optimizer_state


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_cfg, input_dim, dataset_cfg, discrete_mask=None, bce_loss_weight=1.0):
    return TALONTeacher(
        latent_dim=model_cfg['latent_dim'],
        input_dim=input_dim,
        sequence_length=dataset_cfg['window_size'],
        patch_length=dataset_cfg['patch_length'],
        patch_embedder=None,
        freeze_embedder=True,
        enc_hidden_dim=model_cfg['enc_hidden_dim'],
        dec_hidden_dim=model_cfg['dec_hidden_dim'],
        gp_time_kernel=model_cfg['gp_time_kernel'],
        rank_c=model_cfg['rank_c'],
        gp_jitter=model_cfg['gp_jitter'],
        bandwidth=model_cfg['bandwidth'],
        posterior_tc_banded=model_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=model_cfg['tc_channel_bandwidth'],
        encoder_kwargs=model_cfg['encoder_kwargs'],
        decoder_kwargs=model_cfg['decoder_kwargs'],
        discrete_mask=discrete_mask,
        bce_loss_weight=bce_loss_weight,
    )



def make_loaders(dataset_cfg, batch_size, num_workers, seed, val_split, max_rows=None):
    if dataset_cfg['type'] != 'csv':
        raise NotImplementedError(f"Only csv datasets supported in this script. Got: {dataset_cfg['type']}")

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
        max_rows=max_rows,
        downsample_rate=dataset_cfg.get('sampling_rate_seconds', 1),
        scaler_type=dataset_cfg.get('scaler_type', 'minmax'),
        downsample_mode=dataset_cfg.get('downsample_mode', 'median')
    )

    if train_dataset is None or test_dataset is None:
        raise RuntimeError("Failed to load dataset. Check paths and CSV names.")

    val_size = int(len(train_dataset) * val_split)
    train_size = max(1, len(train_dataset) - val_size)

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset


def forward_batch(model, y, device):
    y = y.float().to(device)
    batch_size, channels, seq_len = y.shape
    irrelevant_mask = torch.ones(batch_size, seq_len, channels, device=device, dtype=torch.bool)
    output = model(y, irrelevant_mask=irrelevant_mask)
    return output


def run_epoch(model, loader, device, optimizer=None, loss_weights=None, max_batches=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        y = batch[1]
        if is_train:
            optimizer.zero_grad()

        output = forward_batch(model, y, device)
        loss_dict = enhanced_tspvae_loss(
            output,
            target=y.to(device),
            alpha=loss_weights['alpha'],
            beta=loss_weights['beta'],
        )

        loss = loss_dict['total_loss']
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict['reconstruction_loss'].item()
        total_kl += loss_dict['kl_loss'].item()
        n_batches += 1

    denom = max(1, n_batches)
    return {
        'loss': total_loss / denom,
        'recon': total_recon / denom,
        'kl': total_kl / denom,
    }


def log_reconstructions(model, loader, device, writer, epoch, num_samples=3, num_channels=None):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from torchvision.transforms.functional import to_tensor
    import numpy as np

    # Retrieve dataset metadata for channel names and discrete tags
    dataset = loader.dataset
    underlying = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    feature_names = underlying.metadata.get('feature_names', None) if hasattr(underlying, 'metadata') else None
    y_indices = getattr(underlying, 'y_indices', None)
    y_discrete_mask = getattr(underlying, 'y_discrete_mask', None)

    model.eval()
    plotted = 0
    with torch.inference_mode():
        for batch in loader:
            y = batch[1]
            batch_size, channels, seq_len = y.shape
            
            # Feed raw [B, C, T] tensor directly to the model (no flattening)
            y_gpu = y.float().to(device)
            output = model(y_gpu)
            
            # Extract reconstructed signal
            y_hat = output.x_hat if hasattr(output, 'x_hat') else output['x_hat']
            
            channels_to_plot = channels if num_channels is None else min(channels, num_channels)
            for i in range(min(batch_size, num_samples - plotted)):
                for c in range(channels_to_plot):
                    target = y[i, c].cpu().numpy()
                    recon = y_hat[i, c].cpu().numpy()
                    
                    # Resolve channel metadata
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


def save_checkpoint(model, optimizer, epoch, best_val, path):
    torch.save(
        {
            'epoch': epoch,
            'best_val_loss': best_val,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train TALONTeacher on Y only (WADI)")
    parser.add_argument('--dataset', type=str, default='WADI')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=TRAINING_CONFIG['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default=TRAINING_CONFIG['weight_decay'])
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/wadi_vae')
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Max rows to read from each CSV (for fast iteration)')
    parser.add_argument('--alpha', type=float, default=TRAINING_CONFIG.get('alpha', 1.0),
                        help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=TRAINING_CONFIG.get('beta', 1.0),
                        help='KL divergence weight')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to base config.json to inherit model and dataset architecture from')
    parser.add_argument('--sampling_rate_seconds', type=int, default=None,
                        help='Override sampling rate / downsample rate in seconds')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging entirely (for quick timing/profiling runs)')

    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_cfg = DATASET_CONFIGS[args.dataset].copy()
    model_cfg = SHARED_MODEL_CONFIG.copy()

    # Per-dataset model dimensions. SHARED_MODEL_CONFIG is sized for the 14+ target-channel
    # datasets; a dataset whose target channel count differs (e.g. GHL_full's 4 sensors) must
    # scale latent_dim/rank_c with it, following SWaT's convention of latent_dim ~ channel
    # count. A resume config, loaded below, still wins over this.
    if dataset_cfg.get('model_overrides'):
        model_cfg.update(dataset_cfg['model_overrides'])
        print(f"Model overrides for {args.dataset}: {dataset_cfg['model_overrides']}")

    # Load base or companion config.json to match checkpoint architecture and dataset dimensions
    cfg_to_load = args.config_path
    if not cfg_to_load and args.resume and os.path.exists(args.resume):
        resume_dir = os.path.dirname(args.resume)
        cfg_to_load = os.path.join(resume_dir, 'config.json')

    if cfg_to_load and os.path.exists(cfg_to_load):
        print(f"Found config: {cfg_to_load}. Overriding model and dataset dimensions.")
        try:
            with open(cfg_to_load, 'r', encoding='utf-8') as f:
                saved_cfg = json.load(f)
                if 'model' in saved_cfg:
                    model_cfg.update(saved_cfg['model'])
                if 'dataset' in saved_cfg:
                    dataset_cfg.update(saved_cfg['dataset'])
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    if getattr(args, 'sampling_rate_seconds', None) is not None:
        dataset_cfg['sampling_rate_seconds'] = args.sampling_rate_seconds
        print(f"Dataset Override: sampling_rate_seconds = {args.sampling_rate_seconds}")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Apply smoke_test overrides BEFORE loading data
    if args.smoke_test:
        args.epochs = 2  # Set to 2 to verify multi-epoch resume behaviour
        args.max_train_batches = 2
        args.max_val_batches = 2
        if args.max_rows is None:
            args.max_rows = 5000

    train_loader, val_loader, test_loader, train_dataset = make_loaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        max_rows=args.max_rows,
    )

    print("\n" + "="*50)
    print("WADI/SWaT Training Pipeline Commencing")
    
    # 1. Log dropped columns
    dropped_cols = train_dataset.metadata.get('dropped_columns', [])
    print(f"Dropped Channels ({len(dropped_cols)} total):")
    if dropped_cols:
        for c in dropped_cols:
            print(f"   - {c}")
    else:
        print("   - None")
        
    # 2. Log discrete vs continuous columns in Y (sensors)
    y_names = [train_dataset.metadata['feature_names'][i] for i in train_dataset.y_indices]
    discrete_y = [name for name, is_disc in zip(y_names, train_dataset.y_discrete_mask) if is_disc]
    continuous_y = [name for name, is_disc in zip(y_names, train_dataset.y_discrete_mask) if not is_disc]
    
    print(f"\nTarget Sensor Channels (Y) Classification ({len(y_names)} total):")
    print(f"   - Discrete/Binary Channels ({len(discrete_y)} total):")
    if discrete_y:
        for c in discrete_y:
            print(f"      - {c}")
    else:
        print("      - None")
        
    print(f"   - Continuous Channels ({len(continuous_y)} total):")
    if len(continuous_y) <= 15:
        for c in continuous_y:
            print(f"      - {c}")
    else:
        print(f"      - [Showing first 15 of {len(continuous_y)}]:")
        for c in continuous_y[:15]:
            print(f"      - {c}")
        print("      - ...")
    print("="*50 + "\n")

    bce_loss_weight = TRAINING_CONFIG.get('bce_loss_weight', 1.0)
    model = build_model(
        model_cfg, 
        len(train_dataset.y_indices), 
        dataset_cfg, 
        discrete_mask=train_dataset.y_discrete_mask,
        bce_loss_weight=bce_loss_weight
    ).to(device)

    param_groups = get_param_groups_with_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
    )

    best_val = float('inf')
    best_train_loss = float('inf')
    spike_threshold = 5.0
    start_epoch = 1

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            safe_load_optimizer_state(optimizer, checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val = checkpoint.get('best_val_loss', float('inf'))
            print(f"Loaded checkpoint. Resuming from epoch {start_epoch} with best val loss {best_val:.4f}")
        else:
            print(f"Checkpoint path '{args.resume}' not found. Starting training from scratch.")

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, args.dataset, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorBoard SummaryWriter (or a no-op stand-in for quick profiling runs)
    if args.no_tensorboard:
        writer = NoOpWriter()
        print("TensorBoard logging disabled (--no_tensorboard).")
    else:
        tb_log_dir = os.path.join("runs", f"{args.dataset}_VAE_run_{run_id}")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logging enabled. Run 'tensorboard --logdir runs' to view progress.")
        print(f"Log directory: {tb_log_dir}")
    print(f"Loss Weighting Configuration: alpha={args.alpha} | beta={args.beta}")

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({'dataset': dataset_cfg, 'model': model_cfg, 'train_args': vars(args)}, f, indent=2)

    history = []
    import time
    print_freq = TRAINING_CONFIG.get('print_frequency', 10)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            loss_weights={'alpha': args.alpha, 'beta': args.beta},
            max_batches=args.max_train_batches,
        )

        # Check for massive deterioration (spike protection)
        if train_metrics['loss'] > spike_threshold * best_train_loss and epoch > 5:
            print(f"Massive spike detected! (Train Loss {train_metrics['loss']:.4f} > {spike_threshold} * {best_train_loss:.4f}). Resetting to best model...", flush=True)
            best_ckpt_path = os.path.join(output_dir, 'best_vae.pth')
            if os.path.exists(best_ckpt_path):
                try:
                    best_ckpt = torch.load(best_ckpt_path, map_location=device)
                    model.load_state_dict(best_ckpt['model_state_dict'])
                    if 'optimizer_state_dict' in best_ckpt:
                        safe_load_optimizer_state(optimizer, best_ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Failed to load best checkpoint for reset: {e}", flush=True)
            else:
                print("No checkpoint found to reset to. Continuing...", flush=True)
            
            # Log the spike event to TensorBoard
            writer.add_scalar('Spike/Reset_Event', 1, epoch)
            continue  # Skip validation and saving for this spiked epoch

        if train_metrics['loss'] < best_train_loss:
            best_train_loss = train_metrics['loss']
        
        # Log normal event to TensorBoard
        writer.add_scalar('Spike/Reset_Event', 0, epoch)

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            loss_weights={'alpha': args.alpha, 'beta': args.beta},
            max_batches=args.max_val_batches,
        )

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Reconstruction/train', train_metrics['recon'], epoch)
        writer.add_scalar('KL/train', train_metrics['kl'], epoch)

        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Reconstruction/val', val_metrics['recon'], epoch)
        writer.add_scalar('KL/val', val_metrics['kl'], epoch)

        # Log reconstructions to TensorBoard
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            log_reconstructions(model, val_loader, device, writer, epoch)

        history.append({'epoch': epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            if epoch % 10 == 0 or epoch == args.epochs:
                save_checkpoint(model, optimizer, epoch, best_val, os.path.join(output_dir, 'best_vae.pth'))

        # Save the latest checkpoint every 10 epochs to resume from if interrupted (checkpoint
        # I/O was measured to be a significant fraction of epoch time for this small model)
        if epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], os.path.join(output_dir, 'latest_vae.pth'))
        epoch_duration = time.time() - epoch_start_time
        if epoch == 1 or epoch % print_freq == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | train {train_metrics['loss']:.4f} | "
                f"val {val_metrics['loss']:.4f} | recon {val_metrics['recon']:.4f} | "
                f"Time: {epoch_duration:.2f}s"
            )

    writer.close()
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == '__main__':
    main()

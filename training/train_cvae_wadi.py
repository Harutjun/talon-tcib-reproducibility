"""
========================================================================================
TRAIN THE "JOINT CVAE" SINGLE-STAGE ABLATION BASELINE ON WADI
========================================================================================
Trains `models.JointCVAE.JointCVAE` end-to-end on the standard ELBO:

        L = alpha * E_q[ ||y - y_hat(x,z)||^2_w ]  +  beta * KL( q(z|x,y) || p(z) )

with alpha / beta / lr / weight_decay / batch_size / architecture matching TALON's
real WADI configuration at `results/wadi_vae/WADI/BestFullChannels/config.json`, so the only
difference between this baseline and TALON is the architecture (single-stage joint vs.
two-stage decoupled), not the capacity or the optimisation recipe.

Model Capacity:
  - TALON Student: 33,065,972 parameters
  - Joint CVAE:    33,230,305 parameters (matches within ~0.5%)

Usage:
------
  # Smoke test
  & 'C:\\Users\\arik1\\.conda\\envs\\CVAE1\\python.exe' train_wadi_joint_cvae.py --smoke_test

  # Real run (e.g. 200 epochs, early stopping patience 50)
  & 'C:\\Users\\arik1\\.conda\\envs\\CVAE1\\python.exe' train_wadi_joint_cvae.py --epochs 200 --patience 50
========================================================================================
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from datasets.LocalTSAD import load_csv_dataset
from models.JointCVAE import JointCVAE
from utils.optimizer_utils import get_param_groups_with_weight_decay, safe_load_optimizer_state

TALON_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'results', 'wadi_vae', 'WADI', 'BestFullChannels', 'config.json')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_talon_config():
    with open(TALON_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_datasets(d_cfg, stride, max_rows=None):
    return load_csv_dataset(
        data_root=os.path.join(PROJECT_ROOT, d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'],
        attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'],
        timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'],
        y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'],
        stride=stride,
        x_override=d_cfg.get('x_override'),
        y_override=d_cfg.get('y_override'),
        drop_columns=d_cfg.get('drop_columns', []),
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median'),
        max_rows=max_rows,
    )


def build_model(cfg, x_dim, y_dim, discrete_mask):
    d_cfg, m_cfg = cfg['dataset'], cfg['model']
    return JointCVAE(
        x_dim=x_dim,
        y_dim=y_dim,
        sequence_length=d_cfg['window_size'],
        patch_length=d_cfg['patch_length'],
        latent_dim=m_cfg['latent_dim'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'],
        dec_hidden_dim=m_cfg['dec_hidden_dim'],
        bandwidth=m_cfg['bandwidth'],
        gp_time_kernel=m_cfg['gp_time_kernel'],
        rank_c=m_cfg['rank_c'],
        gp_jitter=m_cfg['gp_jitter'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'],
        decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=discrete_mask,
        bce_loss_weight=1.0,
    )


def run_epoch(model, loader, device, optimizer=None, alpha=1.0, beta=0.033, max_batches=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    tot, tot_r, tot_k, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = batch[0].to(device, non_blocking=True).permute(0, 2, 1).float()
            y = batch[1].to(device, non_blocking=True).permute(0, 2, 1).float()

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(x, y, deterministic=not is_train)
            recon = out.reconstruction_loss.mean()
            kl = out.KL_Loss.mean()
            loss = alpha * recon + beta * kl

            if is_train:
                loss.backward()
                optimizer.step()

            tot += loss.item()
            tot_r += recon.item()
            tot_k += kl.item()
            n += 1

    d = max(1, n)
    return {'loss': tot / d, 'recon': tot_r / d, 'kl': tot_k / d}


def save_checkpoint(model, optimizer, epoch, best_val, path):
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_val,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def main():
    cfg = load_talon_config()
    ta = cfg['train_args']

    p = argparse.ArgumentParser(description="Train Joint CVAE ablation on WADI")
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=ta.get('batch_size', 2048))
    p.add_argument('--learning_rate', type=float, default=ta.get('learning_rate', 1e-4))
    p.add_argument('--weight_decay', type=float, default=ta.get('weight_decay', 0.1))
    p.add_argument('--alpha', type=float, default=ta.get('alpha', 1.0))
    p.add_argument('--beta', type=float, default=ta.get('beta', 0.033))
    p.add_argument('--val_split', type=float, default=ta.get('val_split', 0.1))
    p.add_argument('--seed', type=int, default=ta.get('seed', 42))
    p.add_argument('--stride', type=int, default=cfg['dataset'].get('stride', 1))
    p.add_argument('--patience', type=int, default=40,
                   help='Stop after this many epochs with no val-loss improvement')
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--max_hours', type=float, default=2.0,
                   help='Hard wall-clock budget in hours')
    p.add_argument('--output_dir', type=str, default='results/wadi_joint_cvae')
    p.add_argument('--print_freq', type=int, default=1)
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("JOINT CVAE (single-stage ablation) -- WADI")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    set_seed(args.seed)
    d_cfg = cfg['dataset']

    train_full, test_ds = build_datasets(
        d_cfg, stride=args.stride,
        max_rows=5000 if args.smoke_test else None
    )
    x_dim = len(train_full.x_indices)
    y_dim = len(train_full.y_indices)
    print(f"[data] X channels = {x_dim}, Y channels = {y_dim}, "
          f"train windows = {len(train_full)}, test windows = {len(test_ds)}")
    print(f"[data] window={d_cfg['window_size']} patch={d_cfg['patch_length']} stride={args.stride} "
          f"downsample={d_cfg['sampling_rate_seconds']}s/{d_cfg['downsample_mode']} scaler={d_cfg['scaler_type']}")
    n_disc = int(np.sum(train_full.y_discrete_mask))
    print(f"[data] discrete Y channels: {n_disc} / {y_dim}")

    val_size = int(len(train_full) * args.val_split)
    train_size = max(1, len(train_full) - val_size)
    gen = torch.Generator().manual_seed(args.seed)
    train_split, val_split = random_split(train_full, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[data] train/val windows = {train_size}/{val_size}  "
          f"({len(train_loader)} train batches @ bs={args.batch_size})")

    model = build_model(cfg, x_dim, y_dim, train_full.y_discrete_mask).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] trainable parameters: {n_params:,}")

    param_groups = get_param_groups_with_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        safe_load_optimizer_state(optimizer, ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val_loss', float('inf'))
        out_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"[resume] from {args.resume} @ epoch {start_epoch}, best_val={best_val:.4f}")
        print(f"[resume] continuing in directory: {out_dir}")
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(PROJECT_ROOT, args.output_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'talon_config_source': TALON_CONFIG_PATH,
                   'dataset': d_cfg, 'model': cfg['model'],
                   'train_args': vars(args), 'trainable_parameters': n_params},
                  f, indent=2)

    best_path = os.path.join(out_dir, 'best_joint_cvae.pth')
    latest_path = os.path.join(out_dir, 'latest_joint_cvae.pth')

    print(f"[train] writing to: {out_dir}")
    print(f"[train] budget: max {args.epochs} epochs, patience {args.patience}, "
          f"max_hours {args.max_hours}")

    t_start = time.time()
    patience_left = args.patience
    history = []
    hist_path = os.path.join(out_dir, 'history.json')
    if os.path.exists(hist_path):
        try:
            with open(hist_path, 'r') as f:
                history = json.load(f)
            print(f"[resume] loaded {len(history)} previous epoch records from history.json")
        except Exception:
            history = []

    max_train_batches = 2 if args.smoke_test else None
    max_val_batches = 2 if args.smoke_test else None
    max_epochs = 2 if args.smoke_test else args.epochs

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer=optimizer,
                       alpha=args.alpha, beta=args.beta, max_batches=max_train_batches)
        va = run_epoch(model, val_loader, device, optimizer=None,
                       alpha=args.alpha, beta=args.beta, max_batches=max_val_batches)
        dt = time.time() - t0

        improved = (best_val - va['loss']) > args.min_delta
        if improved:
            best_val = va['loss']
            save_checkpoint(model, optimizer, epoch, best_val, best_path)
            patience_left = args.patience
            tag = "  * BEST"
        else:
            patience_left -= 1
            tag = f"  (patience {patience_left})"

        save_checkpoint(model, optimizer, epoch, va['loss'], latest_path)

        rec = {'epoch': epoch, 'elapsed_sec': time.time() - t_start,
               'epoch_sec': dt, 'train_loss': tr['loss'], 'train_recon': tr['recon'],
               'train_kl': tr['kl'], 'val_loss': va['loss'], 'val_recon': va['recon'],
               'val_kl': va['kl'], 'best_val': best_val}
        history.append(rec)

        if epoch % args.print_freq == 0 or improved or epoch == max_epochs:
            print(f"Epoch {epoch:04d} | dt {dt:5.2f}s | "
                  f"train {tr['loss']:9.4f} (r {tr['recon']:8.4f}, kl {tr['kl']:8.4f}) | "
                  f"val {va['loss']:9.4f} (r {va['recon']:8.4f}, kl {va['kl']:8.4f}) | "
                  f"best {best_val:9.4f}{tag}")

        if patience_left <= 0:
            print(f"[stop] patience exhausted at epoch {epoch}. Best val: {best_val:.4f}")
            break

        if args.max_hours is not None:
            hours = (time.time() - t_start) / 3600.0
            if hours >= args.max_hours:
                print(f"[stop] wall-clock budget reached ({hours:.2f}h >= {args.max_hours}h) at epoch {epoch}")
                break

    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    total_time = (time.time() - t_start) / 60.0
    print("=" * 90)
    print(f"Done. Trained {len(history)} epochs in {total_time:.1f} min. Best val loss: {best_val:.4f}")
    print(f"Best checkpoint:   {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()

"""
========================================================================================
TRAIN THE "JOINT CVAE" SINGLE-STAGE ABLATION BASELINE ON SWaT
========================================================================================
Trains `models.JointCVAE.JointCVAE` end-to-end on the standard ELBO

        L = alpha * E_q[ ||y - y_hat(x,z)||^2_w ]  +  beta * KL( q(z|x,y) || p(z) )

with alpha / beta / lr / weight_decay / batch_size / architecture taken VERBATIM from
TALON's real SWaT configuration at `results/swat_cve/BestFull/config.json`, so the only
difference between this baseline and TALON is the architecture (single-stage joint vs.
two-stage decoupled), not the capacity or the optimisation recipe.

Differences from `training/train_wadi_vae.py` (TALON stage 1), all deliberate:
  * the model is JointCVAE (sees X and Y), not EnhancedTSPVAE (sees Y only);
  * early stopping on a validation-loss plateau replaces the fixed 24,000-epoch run
    (see --patience / --min_delta), because a blind 24k-epoch run is not affordable here;
  * the KL is computed with a vectorised batched-Cholesky implementation instead of the
    original per-sample Python loop. `--verify_kl` proves the two agree numerically.

Usage
-----
  # numerical check that the vectorised KL == TALON's reference loop implementation
  python training/train_swat_joint_cvae.py --verify_kl

  # short timing probe before committing to a long run
  python training/train_swat_joint_cvae.py --smoke_test

  # real run
  python training/train_swat_joint_cvae.py --epochs 24000 --patience 1500
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.LocalTSAD import load_csv_dataset
from models.JointCVAE import JointCVAE
from utils.optimizer_utils import get_param_groups_with_weight_decay, safe_load_optimizer_state

TALON_CONFIG_PATH = os.path.join(project_root, 'results', 'swat_cve', 'BestFull', 'config.json')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_talon_config():
    with open(TALON_CONFIG_PATH) as f:
        return json.load(f)


def build_datasets(d_cfg, stride, max_rows=None):
    return load_csv_dataset(
        data_root=os.path.join(project_root, d_cfg['data_root']),
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


# ---------------------------------------------------------------------------------
# KL equivalence check against the reference (TALON) implementation
# ---------------------------------------------------------------------------------
def verify_kl(device):
    """Assert the vectorised KL equals EnhancedTSPVAE.compute_kl_divergence exactly."""
    from models.EnhancedTSPVAE import EnhancedTSPVAE

    cfg = load_talon_config()
    torch.manual_seed(0)
    model = build_model(cfg, x_dim=26, y_dim=25, discrete_mask=np.zeros(25, dtype=bool)).to(device)
    model.train()

    B, T, Cx, Cy = 8, cfg['dataset']['window_size'], 26, 25
    x = torch.randn(B, T, Cx, device=device)
    y = torch.randn(B, T, Cy, device=device)
    mu, params, _ = model.encode(x, y)

    mine = model.compute_kl_divergence(mu, params)

    # Reference: call TALON's own unbound method against our module (it only touches
    # self.gp_prior, self.encoder.bandwidth / .tc_channel_bandwidth, self.posterior_tc_banded)
    ref = EnhancedTSPVAE.compute_kl_divergence(model, mu, params, None)

    diff = (mine - ref).abs().max().item()
    rel = (diff / (ref.abs().max().item() + 1e-12))
    print(f"[verify_kl] vectorised KL vs TALON reference loop: "
          f"max |diff| = {diff:.6e}  (relative {rel:.3e})")
    print(f"[verify_kl] sample KL values (mine): {mine.detach().cpu().numpy()[:4]}")
    print(f"[verify_kl] sample KL values (ref) : {ref.detach().cpu().numpy()[:4]}")
    ok = diff < 1e-3 or rel < 1e-6
    print(f"[verify_kl] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None, alpha=1.0, beta=1.0, max_batches=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    tot, tot_r, tot_k, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = batch[0].to(device, non_blocking=True).permute(0, 2, 1).float()   # [B,T,Cx]
            y = batch[1].to(device, non_blocking=True).permute(0, 2, 1).float()   # [B,T,Cy]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(x, y, deterministic=not is_train)
            recon = out.reconstruction_loss.mean()
            kl = out.KL_Loss.mean()
            loss = alpha * recon + beta * kl

            if is_train:
                loss.backward()
                optimizer.step()

            tot += loss.item(); tot_r += recon.item(); tot_k += kl.item(); n += 1

    d = max(1, n)
    return {'loss': tot / d, 'recon': tot_r / d, 'kl': tot_k / d}


def save_checkpoint(model, optimizer, epoch, best_val, path):
    torch.save({'epoch': epoch, 'best_val_loss': best_val,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)


def main():
    cfg = load_talon_config()
    ta = cfg['train_args']

    p = argparse.ArgumentParser(description="Train Joint CVAE ablation on SWaT")
    p.add_argument('--epochs', type=int, default=ta['epochs'])
    p.add_argument('--batch_size', type=int, default=ta['batch_size'])
    p.add_argument('--learning_rate', type=float, default=ta['learning_rate'])
    p.add_argument('--weight_decay', type=float, default=ta['weight_decay'])
    p.add_argument('--alpha', type=float, default=ta['alpha'])
    p.add_argument('--beta', type=float, default=ta['beta'])
    p.add_argument('--val_split', type=float, default=ta['val_split'])
    p.add_argument('--seed', type=int, default=ta['seed'])
    p.add_argument('--stride', type=int, default=cfg['dataset']['stride'])
    p.add_argument('--patience', type=int, default=1500,
                   help='Stop after this many epochs with no val-loss improvement')
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--max_hours', type=float, default=None,
                   help='Hard wall-clock budget; stop cleanly when exceeded')
    p.add_argument('--output_dir', type=str, default='results/swat_joint_cvae')
    p.add_argument('--print_freq', type=int, default=50)
    p.add_argument('--target_val_loss', type=float, default=None, help='Stop when best val loss <= target')
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--verify_kl', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("JOINT CVAE (single-stage ablation) -- SWaT")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    if args.verify_kl:
        ok = verify_kl(device)
        sys.exit(0 if ok else 1)

    set_seed(args.seed)
    d_cfg = cfg['dataset']

    train_full, test_ds = build_datasets(d_cfg, stride=args.stride,
                                         max_rows=5000 if args.smoke_test else None)
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

    start_epoch, best_val = 1, float('inf')
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        safe_load_optimizer_state(optimizer, ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val_loss', float('inf'))
        print(f"[resume] from {args.resume} @ epoch {start_epoch}, best_val={best_val:.4f}")

    if args.smoke_test:
        args.epochs = min(args.epochs, 20)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(project_root, args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    run_cfg = {'dataset': d_cfg, 'model': cfg['model'], 'train_args': vars(args),
               'x_dim': x_dim, 'y_dim': y_dim, 'n_params': n_params,
               'source_talon_config': TALON_CONFIG_PATH}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2)
    print(f"[out] {out_dir}")
    print(f"[opt] lr={args.learning_rate} wd={args.weight_decay} alpha={args.alpha} beta={args.beta} "
          f"epochs<={args.epochs} patience={args.patience}")

    history = []
    best_train = float('inf')
    epochs_since_improve = 0
    t_start = time.time()
    stop_reason = 'max_epochs'

    for epoch in range(start_epoch, args.epochs + 1):
        te = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, args.alpha, args.beta)

        # spike protection, as in TALON's stage-1 training loop
        if epoch > 5 and tr['loss'] > 5.0 * best_train:
            print(f"[spike] epoch {epoch}: train {tr['loss']:.4f} > 5x best {best_train:.4f}; "
                  f"restoring best checkpoint", flush=True)
            bp = os.path.join(out_dir, 'best_joint_cvae.pth')
            if os.path.exists(bp):
                ck = torch.load(bp, map_location=device)
                model.load_state_dict(ck['model_state_dict'])
                safe_load_optimizer_state(optimizer, ck['optimizer_state_dict'])
            continue
        best_train = min(best_train, tr['loss'])

        va = run_epoch(model, val_loader, device, None, args.alpha, args.beta)
        dt = time.time() - te
        history.append({'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()},
                        'sec': dt})

        if va['loss'] < best_val - args.min_delta:
            best_val = va['loss']
            epochs_since_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val,
                            os.path.join(out_dir, 'best_joint_cvae.pth'))
        else:
            epochs_since_improve += 1

        if epoch == 1 or epoch % args.print_freq == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, va['loss'],
                            os.path.join(out_dir, 'latest_joint_cvae.pth'))
            with open(os.path.join(out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            el = time.time() - t_start
            print(f"Epoch {epoch:05d} | train {tr['loss']:11.4f} (rec {tr['recon']:10.4f} kl {tr['kl']:9.4f}) "
                  f"| val {va['loss']:11.4f} | best {best_val:11.4f} | stale {epochs_since_improve:5d} "
                  f"| {dt:.2f}s/ep | elapsed {el/60:.1f}m", flush=True)

        if args.target_val_loss is not None and best_val <= args.target_val_loss:
            stop_reason = f'target_val_loss reached ({best_val:.4f} <= {args.target_val_loss:.4f})'
            print(f'[stop] {stop_reason} at epoch {epoch}')
            break

        if epochs_since_improve >= args.patience:
            stop_reason = f'early_stopping (no val improvement for {args.patience} epochs)'
            print(f"[stop] {stop_reason} at epoch {epoch}")
            break
        if args.max_hours is not None and (time.time() - t_start) > args.max_hours * 3600:
            stop_reason = f'wall_clock budget of {args.max_hours}h reached'
            print(f"[stop] {stop_reason} at epoch {epoch}")
            break

    save_checkpoint(model, optimizer, history[-1]['epoch'] if history else 0,
                    best_val, os.path.join(out_dir, 'latest_joint_cvae.pth'))
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, 'run_summary.json'), 'w') as f:
        json.dump({'stop_reason': stop_reason,
                   'epochs_run': history[-1]['epoch'] if history else 0,
                   'best_val_loss': best_val,
                   'final_train_loss': history[-1]['loss'] if history else None,
                   'final_val_loss': history[-1]['val_loss'] if history else None,
                   'wall_clock_sec': time.time() - t_start,
                   'n_params': n_params}, f, indent=2)
    print(f"\nDone. stop_reason={stop_reason} best_val={best_val:.4f} "
          f"wall_clock={(time.time()-t_start)/60:.1f} min")
    print(f"Artifacts: {out_dir}")


if __name__ == '__main__':
    main()

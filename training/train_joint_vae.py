"""Train the Joint VAE baseline: a TALONTeacher-architecture model over concatenated
[X; Y], with no directional conditioning, for SWaT or WADI. Capacity matches TALON/CVAE
on each benchmark. Applies a local, vectorized (batch-parallel Cholesky) implementation of
compute_kl_divergence for training speed -- functionally equivalent to the per-sample-loop
version in models/TALONTeacher.py, not a different quantity.

Usage: python training/train_joint_vae.py --dataset {SWaT,WADI} [hyperparameter overrides]
"""

import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader, TensorDataset

# Enable TensorFloat-32 and cuDNN benchmark for max GPU throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# --------------------------------------------------------------------------------------
# Robust Cholesky Patch with Eigenvalue Projection Fallback
# --------------------------------------------------------------------------------------
import models.TALONTeacher as _etspvae
if 'models.TALONStudent' in sys.modules:
    import models.TALONStudent as _etspcve
else:
    _etspcve = None


def _patched_robust_cholesky_fn(M, jitter_start=1e-6, jitter_max=1e-1, max_tries=7):
    device = M.device
    dtype = M.dtype
    I = torch.eye(M.size(-1), device=device, dtype=dtype)
    jitter = float(jitter_start)
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(M + jitter * I)
            if not torch.isnan(L).any():
                return L
        except (RuntimeError, torch._C._LinAlgError):
            pass
        jitter = min(jitter * 10.0, float(jitter_max))
    try:
        return torch.linalg.cholesky(M + jitter_max * I)
    except (RuntimeError, torch._C._LinAlgError):
        sym_M = 0.5 * (M + M.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(sym_M)
        eigvals = torch.clamp(eigvals, min=float(jitter_start))
        M_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
        return torch.linalg.cholesky(M_psd)


_etspvae.robust_cholesky_fn = _patched_robust_cholesky_fn

import sys as _sys
_et_mod = _sys.modules['models.TALONTeacher']

def _batched_compute_kl_divergence(self, mu_batch, posterior, mask=None):
    if not isinstance(posterior, dict):
        raise RuntimeError("compute_kl_divergence requires a per-batch posterior dict")
    B, C, T = mu_batch.shape
    device = mu_batch.device

    K_t = self.gp_prior.K_t(); K_c = self.gp_prior.K_c()
    L_Kt = torch.linalg.cholesky(K_t); L_Kc = torch.linalg.cholesky(K_c)
    I_T = torch.eye(T, device=device, dtype=K_t.dtype)
    I_C = torch.eye(C, device=device, dtype=K_c.dtype)
    K_t_inv = torch.cholesky_solve(I_T, L_Kt); K_c_inv = torch.cholesky_solve(I_C, L_Kc)
    log_det_Kt = 2 * torch.sum(torch.log(torch.diag(L_Kt)))
    log_det_Kc = 2 * torch.sum(torch.log(torch.diag(L_Kc)))
    log_det_K = C * log_det_Kt + T * log_det_Kc
    jitter = float(self.gp_prior.jitter.item()) if isinstance(self.gp_prior.jitter, torch.Tensor) else float(self.gp_prior.jitter)

    time_diag = posterior.get('precision_diag', None)
    time_bands = posterior.get('precision_bands', None)
    chan_diag = posterior.get('precision_channel_diag', None)
    chan_bands = posterior.get('precision_channel_bands', None)

    has_channel = (self.posterior_tc_banded and chan_diag is not None and chan_bands is not None)

    assemble_fn = getattr(_et_mod, 'assemble_precision_from_bands_fn')
    P_t_batch = assemble_fn(
        time_diag, time_bands, bandwidth=int(getattr(self.encoder, 'bandwidth', T - 1))
    )
    try:
        L_Qt = torch.linalg.cholesky(P_t_batch + jitter * I_T)
        log_det_Qt = 2 * torch.diagonal(L_Qt, dim1=-2, dim2=-1).log().sum(dim=-1)
        Qt_inv_stack = torch.cholesky_solve(I_T.expand(B, T, T), L_Qt)
    except Exception:
        log_det_Qt_b = []
        Qt_inv_b = []
        for b in range(B):
            Qt = P_t_batch[b]
            L_Qt_b = _patched_robust_cholesky_fn(Qt, jitter_start=jitter, jitter_max=1e-1)
            log_det_Qt_b.append(2 * torch.sum(torch.log(torch.diag(L_Qt_b))))
            Qt_inv_b.append(torch.cholesky_solve(I_T, L_Qt_b))
        log_det_Qt = torch.stack(log_det_Qt_b, dim=0)
        Qt_inv_stack = torch.stack(Qt_inv_b, dim=0)

    if has_channel:
        P_c_batch = assemble_fn(
            chan_diag, chan_bands, bandwidth=int(getattr(self.encoder, 'tc_channel_bandwidth', max(0, C - 1)))
        )
        try:
            L_Qc = torch.linalg.cholesky(P_c_batch + jitter * I_C)
            log_det_Qc = 2 * torch.diagonal(L_Qc, dim1=-2, dim2=-1).log().sum(dim=-1)
            Qc_inv_stack = torch.cholesky_solve(I_C.expand(B, C, C), L_Qc)
        except Exception:
            log_det_Qc_b = []
            Qc_inv_b = []
            for b in range(B):
                Qc = P_c_batch[b]
                L_Qc_b = _patched_robust_cholesky_fn(Qc, jitter_start=jitter, jitter_max=1e-1)
                log_det_Qc_b.append(2 * torch.sum(torch.log(torch.diag(L_Qc_b))))
                Qc_inv_b.append(torch.cholesky_solve(I_C, L_Qc_b))
            log_det_Qc = torch.stack(log_det_Qc_b, dim=0)
            Qc_inv_stack = torch.stack(Qc_inv_b, dim=0)
        trace_t = torch.sum(K_t_inv * Qt_inv_stack, dim=[1, 2])
        trace_c = torch.sum(K_c_inv * Qc_inv_stack, dim=[1, 2])
        trace_term = trace_t * trace_c
        log_det_Q = C * log_det_Qt + T * log_det_Qc
    else:
        trace_term = torch.sum(K_t_inv * Qt_inv_stack, dim=[1, 2]) * torch.trace(K_c_inv)
        log_det_Q = C * log_det_Qt

    if mask is not None:
        mask_f = mask.float()
        mu_masked = mu_batch * mask_f.unsqueeze(1)
        valid_counts = mask_f.sum(dim=1)
    else:
        mu_masked = mu_batch
        valid_counts = torch.full((B,), float(T), device=device)

    mu_Kc = torch.matmul(K_c_inv, mu_masked)
    mu_Kc_Kt = torch.matmul(mu_Kc, K_t_inv)
    mahalanobis = torch.sum(mu_masked * mu_Kc_Kt, dim=[1, 2])

    n_eff = valid_counts * C
    kl_per_sample = 0.5 * (log_det_K + log_det_Q - n_eff + trace_term + mahalanobis)
    return kl_per_sample

# Patch the TALONTeacher class directly
_et_class = getattr(_et_mod, 'TALONTeacher')
_et_class.compute_kl_divergence = _batched_compute_kl_divergence

if _etspcve is not None:
    _etspcve.robust_cholesky_fn = _patched_robust_cholesky_fn
    if 'models.TALONStudent' in sys.modules:
        sys.modules['models.TALONStudent'].robust_cholesky_fn = _patched_robust_cholesky_fn

from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from utils.optimizer_utils import get_param_groups_with_weight_decay, safe_load_optimizer_state


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_config_path(dataset_name: str) -> str:
    if dataset_name.lower() == 'swat':
        return os.path.join(PROJECT_ROOT, 'results', 'swat_cve', 'BestFull', 'config.json')
    elif dataset_name.lower() == 'wadi':
        return os.path.join(PROJECT_ROOT, 'results', 'wadi_vae', 'WADI', 'BestFullChannels', 'config.json')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Must be SWaT or WADI.")


def build_vectorized_dataset(train_full, d_cfg, stride):
    """Vectorized sliding-window extraction: produces [N_windows, C_total, window_size] tensor.
    Eliminates all per-item Python overhead during training.
    """
    # Channel concatenation [T_total, C_x + C_y]
    data_xy = np.concatenate([
        train_full.data_normalized[:, train_full.x_indices],
        train_full.data_normalized[:, train_full.y_indices]
    ], axis=1)

    t_data = torch.from_numpy(data_xy).float().t()  # [C_total, T_total]
    window_size = d_cfg['window_size']
    windows = t_data.unfold(1, window_size, stride).permute(1, 0, 2).contiguous()  # [N, C_total, W]
    return windows


def build_model(cfg, total_dim, discrete_mask):
    d_cfg, m_cfg = cfg['dataset'], cfg['model']
    return TALONTeacher(
        latent_dim=m_cfg['latent_dim'],
        input_dim=total_dim,
        sequence_length=d_cfg['window_size'],
        patch_length=d_cfg['patch_length'],
        patch_embedder=None,
        freeze_embedder=True,
        enc_hidden_dim=m_cfg['enc_hidden_dim'],
        dec_hidden_dim=m_cfg['dec_hidden_dim'],
        gp_time_kernel=m_cfg['gp_time_kernel'],
        rank_c=m_cfg['rank_c'],
        gp_jitter=m_cfg.get('gp_jitter', 1e-12),
        bandwidth=m_cfg['bandwidth'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'],
        decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=discrete_mask,
        bce_loss_weight=1.0,
    )


def run_epoch(model, loader, device, optimizer=None, alpha=1.0, beta=0.033,
              grad_clip=1.0, max_batches=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    tot_loss, tot_recon, tot_kl, n_batches = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for bi, (xy,) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break

            xy = xy.to(device, non_blocking=True)
            B, C, T = xy.shape
            mask = torch.ones(B, T, C, device=device, dtype=torch.bool)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(xy, irrelevant_mask=mask)
            recon = out.reconstruction_loss.mean()
            kl = out.KL_Loss.mean()
            loss = alpha * recon + beta * kl

            if is_train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            tot_loss += loss.item()
            tot_recon += recon.item()
            tot_kl += kl.item()
            n_batches += 1

    d = max(1, n_batches)
    return {'loss': tot_loss / d, 'recon': tot_recon / d, 'kl': tot_kl / d}


def main():
    p = argparse.ArgumentParser(description="Train Pure Joint VAE on [X; Y] for SWaT or WADI")
    p.add_argument('--dataset', type=str, required=True, choices=['SWaT', 'WADI'],
                   help='Target dataset: SWaT or WADI')
    p.add_argument('--epochs', type=int, default=1000,
                   help='Target training epochs (default: 1000)')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size (default: 512 for SWaT, 2048 for WADI)')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.033)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stride', type=int, default=None,
                   help='Train stride (default: 5 for SWaT, 1 for WADI)')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--ckpt_every', type=int, default=10)
    p.add_argument('--target_val_loss', type=float, default=None, help='Stop when best val loss <= target')
    p.add_argument('--smoke_test', action='store_true',
                   help='Run 5 batches to verify forward/backward/saving')
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    ds_name = args.dataset
    cfg_path = get_dataset_config_path(ds_name)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    d_cfg = cfg['dataset']
    m_cfg = cfg['model']

    if args.batch_size is None:
        args.batch_size = 512 if ds_name == 'SWaT' else 2048
    if args.stride is None:
        args.stride = d_cfg.get('stride', 5 if ds_name == 'SWaT' else 1)
    if args.output_dir is None:
        args.output_dir = f"results/{ds_name.lower()}_joint_pure_vae"

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, 'train.log')

    def log(msg: str):
        print(msg, flush=True)
        with open(log_file_path, 'a', encoding='utf-8') as lf:
            lf.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log("=" * 90)
    log(f"TRAINING PURE JOINT VAE on [X; Y] -- Dataset: {ds_name}")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log(f"GPU Optimizations: TF32=True, cuDNN.benchmark=True, Vectorized DataLoader, Pinned Memory")
    log("=" * 90)

    set_seed(args.seed)

    log("[1/4] Loading and vectorizing dataset...")
    train_full, test_ds = load_csv_dataset(
        data_root=os.path.join(PROJECT_ROOT, d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'],
        attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'],
        timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'],
        y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'],
        stride=args.stride,
        x_override=d_cfg.get('x_override'),
        y_override=d_cfg.get('y_override'),
        drop_columns=d_cfg.get('drop_columns', []),
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median'),
        max_rows=2000 if args.smoke_test else None
    )

    x_dim = len(train_full.x_indices)
    y_dim = len(train_full.y_indices)
    total_dim = x_dim + y_dim
    x_disc = getattr(train_full, 'x_discrete_mask', np.zeros(x_dim, dtype=bool))
    y_disc = getattr(train_full, 'y_discrete_mask', np.zeros(y_dim, dtype=bool))
    joint_discrete_mask = np.concatenate([x_disc, y_disc])
    n_disc = int(np.sum(joint_discrete_mask))

    log(f"  X channels: {x_dim} (discrete: {int(np.sum(x_disc))})")
    log(f"  Y channels: {y_dim} (discrete: {int(np.sum(y_disc))})")
    log(f"  Total joint channels: {total_dim} (discrete: {n_disc})")

    # Vectorized window extraction
    windows = build_vectorized_dataset(train_full, d_cfg, args.stride)
    n_total_windows = len(windows)
    log(f"  Pre-materialized contiguous windows: {windows.shape} ({windows.element_size() * windows.nelement() / (1024**2):.1f} MB)")

    # Train / Val split
    val_size = int(n_total_windows * args.val_split)
    train_size = max(1, n_total_windows - val_size)

    # Deterministic permutation for reproducibility
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total_windows)
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    train_tensor = windows[train_idx]
    val_tensor = windows[val_idx]

    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    log(f"  Split: {train_size} train ({len(train_loader)} batches @ bs={args.batch_size}), {val_size} val ({len(val_loader)} batches)")

    log("[2/4] Building model...")
    model = build_model(cfg, total_dim, joint_discrete_mask).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    log(f"  TALONTeacher trainable parameters: {n_params:,}")

    param_groups = get_param_groups_with_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        log(f"  Resuming from checkpoint: {args.resume}")
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        safe_load_optimizer_state(optimizer, ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val_loss = ck.get('best_val_loss', float('inf'))
        log(f"  Resumed at epoch {start_epoch}, previous best val: {best_val_loss:.4f}")

    best_ckpt_path = os.path.join(out_dir, 'best_joint_vae.pth')
    latest_ckpt_path = os.path.join(out_dir, 'latest_joint_vae.pth')
    history_path = os.path.join(out_dir, 'history.json')

    run_config = {
        'dataset': ds_name,
        'config_source': cfg_path,
        'dataset_cfg': d_cfg,
        'model_cfg': m_cfg,
        'train_args': vars(args),
        'total_channels': total_dim,
        'trainable_parameters': n_params,
        'start_time': datetime.now().isoformat()
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []

    log(f"[3/4] Starting training for {args.epochs} epochs (alpha={args.alpha}, beta={args.beta})...")
    max_b = 5 if args.smoke_test else None
    t0_train = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0_ep = time.time()

        train_res = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            alpha=args.alpha, beta=args.beta, grad_clip=args.grad_clip,
            max_batches=max_b
        )

        val_res = run_epoch(
            model, val_loader, device, optimizer=None,
            alpha=args.alpha, beta=args.beta,
            max_batches=max_b
        )

        ep_time = time.time() - t0_ep
        is_best = val_res['loss'] < best_val_loss

        if is_best:
            best_val_loss = val_res['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_res['loss'],
                'val_recon': val_res['recon'],
                'val_kl': val_res['kl'],
                'best_val_loss': best_val_loss,
                'config': run_config,
            }, best_ckpt_path)

        if epoch % args.ckpt_every == 0 or epoch == args.epochs or is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_res['loss'],
                'best_val_loss': best_val_loss,
                'config': run_config,
            }, latest_ckpt_path)

        record = {
            'epoch': epoch,
            'epoch_time': ep_time,
            'total_time': time.time() - t0_train,
            'train_loss': train_res['loss'],
            'train_recon': train_res['recon'],
            'train_kl': train_res['kl'],
            'val_loss': val_res['loss'],
            'val_recon': val_res['recon'],
            'val_kl': val_res['kl'],
            'best_val_loss': best_val_loss,
            'is_best': is_best,
        }
        history.append(record)

        if epoch % 20 == 0 or is_best or epoch == 1 or args.smoke_test:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        best_marker = " [*BEST*]" if is_best else ""
        if epoch <= 10 or epoch % 10 == 0 or is_best or args.smoke_test:
            log(f"Epoch {epoch:4d}/{args.epochs} ({ep_time:.2f}s) | "
                f"Train Loss: {train_res['loss']:.4f} (R: {train_res['recon']:.4f}, KL: {train_res['kl']:.2f}) | "
                f"Val Loss: {val_res['loss']:.4f} (R: {val_res['recon']:.4f}, KL: {val_res['kl']:.2f}){best_marker}")

        if args.target_val_loss is not None and best_val_loss <= args.target_val_loss:
            log(f'[TARGET REACHED] Best val loss {best_val_loss:.4f} <= target {args.target_val_loss:.4f}. Halting.')
            break

        if args.smoke_test and epoch >= 2:
            log("[SMOKE TEST] Passed 2 epochs successfully. Halting.")
            break

    total_duration = time.time() - t0_train
    log(f"[4/4] Training complete in {total_duration/60:.2f} mins. Best val loss: {best_val_loss:.4f}")
    log(f"Saved best model to: {best_ckpt_path}")


if __name__ == '__main__':
    main()

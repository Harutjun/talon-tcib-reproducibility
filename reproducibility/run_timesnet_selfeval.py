"""
========================================================================================
TimesNet SELF-EVALUATION ON OUR OWN SWaT/WADI TEST DATA -- FAITHFUL OFFICIAL REPRODUCTION
========================================================================================
SOURCE: https://github.com/thuml/Time-Series-Library (official repository)
Architecture vendored in reproducibility/baselines/timesnet.py.
Trains TimesNet with MSE reconstruction loss and evaluates via evaluate_all_metrics.
"""

import os
import sys
import json
import time
import random
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

BASELINES_DIR = PROJECT_ROOT / 'reproducibility' / 'baselines'
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'reproducibility'))

from datasets.LocalTSAD import load_csv_dataset
from timesnet import Model as TimesNetModel
from evaluate_swat_bestfull import evaluate_all_metrics

RESULTS_DIR = PROJECT_ROOT / 'reproducibility' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

_forced = os.environ.get('BASELINE_DEVICE', '').strip().lower()
if _forced in ('cpu', 'cuda'):
    DEVICE = torch.device(_forced)
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else "")
      + "  (override with env var BASELINE_DEVICE=cpu|cuda)", flush=True)

DATASETS = {
    'SWaT': PROJECT_ROOT / 'results' / 'swat_cve' / 'BestFull' / 'config.json',
    'WADI': PROJECT_ROOT / 'results' / 'wadi_vae' / 'WADI' / 'BestFullChannels' / 'config.json',
}

WIN_SIZE = 100
STRIDE = 1
BATCH_SIZE = 128
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
D_MODEL = 64
D_FF = 64
E_LAYERS = 3
TOP_K = 3
NUM_KERNELS = 6
DROPOUT = 0.0
VAL_FRACTION = 0.1
PATIENCE = 3


def load_flat_data(dataset_name: str):
    cfg_path = DATASETS[dataset_name]
    with open(cfg_path) as f:
        cfg = json.load(f)
    d_cfg = cfg['dataset']

    train_dataset, test_dataset = load_csv_dataset(
        data_root=os.path.join(str(PROJECT_ROOT), d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'],
        attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'],
        timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'],
        y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'],
        stride=10,
        x_override=d_cfg.get('x_override'),
        y_override=d_cfg.get('y_override'),
        drop_columns=d_cfg.get('drop_columns'),
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median'),
    )

    n_feats = train_dataset.data_normalized.shape[1]
    print(f"[{dataset_name}] train={train_dataset.data_normalized.shape}, "
          f"test={test_dataset.data_normalized.shape}, feats={n_feats}, "
          f"test anomalies={int(test_dataset.labels.sum())}/{len(test_dataset.labels)}", flush=True)

    trainD = torch.tensor(train_dataset.data_normalized, dtype=torch.float32)
    testD = torch.tensor(test_dataset.data_normalized, dtype=torch.float32)
    test_labels = test_dataset.labels.astype(int)
    return trainD, testD, test_labels, n_feats


def build_windows_overlap(data: torch.Tensor, win_size: int, stride: int) -> torch.Tensor:
    windows = data.unfold(0, win_size, stride)
    return windows.permute(0, 2, 1).contiguous()


def build_windows_blocks(data: torch.Tensor, win_size: int) -> torch.Tensor:
    T = data.shape[0]
    n_blocks = -(-T // win_size)
    pad_len = n_blocks * win_size - T
    if pad_len > 0:
        data = torch.cat([data, data[-1:].repeat(pad_len, 1)], dim=0)
    return data.view(n_blocks, win_size, -1)


def per_job_path(dataset_name: str, seed=None) -> Path:
    suffix = f'_seed{seed}' if seed is not None else ''
    return RESULTS_DIR / f'baseline_selfeval_{dataset_name}_TimesNet{suffix}.json'


def per_job_checkpoint_path(dataset_name: str, seed=None) -> Path:
    suffix = f'_seed{seed}' if seed is not None else ''
    return RESULTS_DIR / f'timesnet_{dataset_name}{suffix}_checkpoint.pth'


def run_one(dataset_name, trainD_flat, testD_flat, test_labels, n_feats, seed=None, resume=True):
    print(f"\n{'='*80}\nTimesNet on {dataset_name}" + (f" (seed={seed})" if seed is not None else "") + f"\n{'='*80}", flush=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    cfg = SimpleNamespace(task_name='anomaly_detection', seq_len=WIN_SIZE, label_len=0, pred_len=0,
                           top_k=TOP_K, d_model=D_MODEL, d_ff=D_FF, num_kernels=NUM_KERNELS,
                           e_layers=E_LAYERS, enc_in=n_feats, c_out=n_feats,
                           embed='timeF', freq='h', dropout=DROPOUT, num_class=0)
    model = TimesNetModel(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    trainD_flat = trainD_flat.to(DEVICE)
    testD_flat = testD_flat.to(DEVICE)

    n_val = int(len(trainD_flat) * VAL_FRACTION)
    train_core, val_core = trainD_flat[:-n_val], trainD_flat[-n_val:]

    train_windows = build_windows_overlap(train_core, WIN_SIZE, STRIDE)
    val_windows = build_windows_overlap(val_core, WIN_SIZE, STRIDE)

    best_val_loss = float('inf')
    best_state = None
    patience_left = PATIENCE
    epochs_trained = 0
    start_epoch = 0

    ckpt_path = per_job_checkpoint_path(dataset_name, seed=seed)
    if resume and ckpt_path.exists():
        print(f"  [RESUME] Found checkpoint at {ckpt_path}, loading...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_state = ckpt.get('best_state', {k: v.detach().clone() for k, v in model.state_dict().items()})
        patience_left = ckpt.get('patience_left', PATIENCE)
        start_epoch = ckpt.get('epoch', 0)
        epochs_trained = ckpt.get('epochs_trained', start_epoch)
        print(f"  [RESUME] Resuming from epoch {start_epoch}/{MAX_EPOCHS} (best_val_loss={best_val_loss:.6f})", flush=True)

    n_train = train_windows.shape[0]
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        train_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = train_windows[idx]

            optimizer.zero_grad()
            rec = model(batch, None, None, None)
            loss = criterion(rec, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        epochs_trained = epoch + 1
        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            val_losses = []
            n_val_w = val_windows.shape[0]
            for start in range(0, n_val_w, BATCH_SIZE):
                batch = val_windows[start:start + BATCH_SIZE]
                rec = model(batch, None, None, None)
                val_loss_batch = criterion(rec, batch)
                val_losses.append(val_loss_batch.item())
            val_loss = float(np.mean(val_losses))

        improved = val_loss < best_val_loss - 1e-6
        print(f'  [epoch {epoch+1}/{MAX_EPOCHS}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}' +
              ('  (best)' if improved else f'  (best={best_val_loss:.6f}, patience {PATIENCE - patience_left + 1}/{PATIENCE})'), flush=True)
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_state': best_state,
                'patience_left': patience_left,
                'epochs_trained': epochs_trained,
            }, ckpt_path)
            print(f'  [CHECKPOINT] Saved best checkpoint to {ckpt_path}', flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'  [EARLY STOP] no val improvement for {PATIENCE} epochs, stopping at epoch {epochs_trained}', flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def score_blocks(data_flat):
        blocks = build_windows_blocks(data_flat, WIN_SIZE)
        scores = []
        with torch.no_grad():
            for start in range(0, blocks.shape[0], BATCH_SIZE):
                batch = blocks[start:start + BATCH_SIZE]
                rec = model(batch, None, None, None)
                err = torch.mean((rec - batch) ** 2, dim=-1)
                scores.append(err.detach().cpu().numpy().reshape(-1))
        return np.concatenate(scores, axis=0)

    test_scores_padded = score_blocks(testD_flat)
    scores = test_scores_padded[:testD_flat.shape[0]]

    metrics = evaluate_all_metrics(test_labels, scores)
    metrics['epochs_trained'] = epochs_trained
    metrics['best_val_loss'] = best_val_loss
    elapsed = time.time() - t0
    print(f"  --> AUC-ROC={metrics['AUC_ROC']:.4f}  AUC-PR={metrics['AUC_PR']:.4f}  "
          f"Point-F1={metrics['Point_F1']:.4f}  VUS-ROC={metrics.get('VUS_ROC', float('nan')):.4f}  "
          f"VUS-PR={metrics.get('VUS_PR', float('nan')):.4f}  "
          f"Aff-F1={metrics.get('Affiliation_F1', float('nan')):.4f}  "
          f"PATE={metrics.get('PATE_AUC_PR', float('nan')):.4f}  epochs={epochs_trained}  ({elapsed:.1f}s)", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS.keys()), required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--force-restart', action='store_true')
    args = parser.parse_args()

    p = per_job_path(args.dataset, seed=args.seed)
    if p.exists() and not args.force_restart:
        print(f"[SKIP] {args.dataset}/TimesNet" + (f" seed={args.seed}" if args.seed is not None else "") + f": already done ({p})", flush=True)
        return

    ckpt_p = per_job_checkpoint_path(args.dataset, seed=args.seed)
    if args.force_restart and ckpt_p.exists():
        ckpt_p.unlink(missing_ok=True)
        print(f"  [FORCE-RESTART] Removed existing checkpoint {ckpt_p}", flush=True)

    trainD_flat, testD_flat, test_labels, n_feats = load_flat_data(args.dataset)
    metrics = run_one(args.dataset, trainD_flat, testD_flat, test_labels, n_feats, seed=args.seed, resume=not args.force_restart)
    with open(p, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVE] {p}", flush=True)


if __name__ == '__main__':
    main()

"""
========================================================================================
GDN SELF-EVALUATION ON OUR OWN SWaT/WADI TEST DATA -- FAITHFUL OFFICIAL REPRODUCTION
========================================================================================
Closes the same literature gap as run_baseline_selfeval.py (DAGMM/OmniAnomaly/MAD-GAN), for GDN,
which was initially scoped OUT of that pass on the mistaken assumption that only its model
architecture (reproducibility/baselines/gdn.py) was available locally. Direct inspection of the
official repository (github.com/d-ailin/GDN: main.py, train.py, test.py, evaluate.py, run.sh,
util/*.py -- all fetched and read directly, not summarized) confirmed a complete, genuinely
reproducible training+scoring pipeline exists; this script reimplements it faithfully against
TALON's own preprocessed SWaT/WADI data.

WHY A SEPARATE SCRIPT FROM run_baseline_selfeval.py: GDN is a FORECASTING model (predict the next
single timestep from a `slide_win`-step context window), trained and scored with a genuinely
different procedure than DAGMM/OmniAnomaly/MAD-GAN's reconstruction-based approach:
  - Windowing: verbatim semantics of the official `datasets/TimeDataset.py` (`make_windows` below)
    -- context window immediately preceding index i, target and anomaly label both at index i.
  - Loss: plain forecast MSE (`train.py`'s `loss_func`), not a reconstruction/adversarial/VAE loss.
  - Validation split: a single RANDOM CONTIGUOUS BLOCK carved out of the training windows
    (`get_loaders_indices` below, verbatim semantics of `main.py`'s `Main.get_loaders()`), not the
    chronological-last-10% convention used for the other three self-evaluated baselines.
  - Anomaly score: per-sensor forecast error normalized by that sensor's own median/IQR, a 4-point
    trailing moving average, then the single highest-scoring sensor at each timestep -- verbatim
    from the official `evaluate.py` (`get_err_scores`/`get_err_median_and_iqr`, `topk=1` case of
    `get_best_performance_data`).

METHODOLOGICAL NOTE -- disclosed test-statistic dependency (see Thesis.tex Appendix D): the
official `evaluate.py` computes each sensor's median/IQR normalization constants from the TEST
set's OWN forecast errors -- `get_err_scores` is called with the test set supplying BOTH the values
being scored and the reference statistics. This is not test-LABEL leakage (labels never enter this
computation, only the unlabeled residual distribution), but it is a real dependency on the test
distribution that departs from this thesis's train-only calibration convention used for TALON and
for the other three self-evaluated baselines. It is reproduced here exactly as officially published
-- this script implements the "faithful reproduction, disclosed" choice, not a silently-modified
train-calibrated variant.

Detection metrics (AUC-ROC/AUC-PR/Point-F1/VUS-ROC/VUS-PR/Affiliation-F1/PATE) are computed with the
SAME `evaluate_all_metrics()` pipeline used for every other self-evaluated baseline and for TALON
itself, applied directly to the continuous per-timestep score above. GDN's own official
oracle-best-threshold point-metric code (`evaluate.py`'s `get_best_performance_data`/
`get_val_performance_data`, which sweep 400 thresholds against the TEST LABELS to report the best
F1) is deliberately NOT used, for the same reason it isn't used for the other three baselines:
consistency of the metric computation itself across every row is what makes the numbers genuinely
comparable to TALON's and to each other.

OFFICIAL HYPERPARAMETERS (from `run.sh`, matching the configuration already cited in Thesis.tex
Appendix D's FLOPs profiling of GDN -- confirmed consistent, not re-tuned here): seed=5,
batch_size=32, slide_win=5, slide_stride=1, dim=64, topk=5 (graph sparsification, distinct from the
scoring-time topk=1 sensor aggregation above), out_layer_num=1, out_layer_inter_dim=128,
val_ratio=0.2, decay=0, epoch=30 (cap), early_stop_win=15 (on validation MSE loss, matching
`train.py`'s own early stopping -- GDN, unlike DAGMM/OmniAnomaly/MAD-GAN, already has a principled,
official early-stopping criterion, so no departure from the official recipe was needed here). The
official model itself trains in float32 (never `.double()`'d in the official code), reproduced
as-is here.

USAGE:
------
  ...\\python.exe reproducibility/run_gdn_selfeval.py --dataset SWaT
  ...\\python.exe reproducibility/run_gdn_selfeval.py --dataset WADI
  (then, to fold GDN into the combined summary alongside the other three baselines)
  ...\\python.exe reproducibility/run_baseline_selfeval.py --combine
========================================================================================
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import iqr as scipy_iqr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

BASELINES_DIR = PROJECT_ROOT / 'reproducibility' / 'baselines'
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'reproducibility'))

from datasets.LocalTSAD import load_csv_dataset
from gdn import GDN, fully_connected_edge_index
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

# Official run.sh hyperparameters -- verbatim, not re-tuned.
SEED = 5
BATCH_SIZE = 32
SLIDE_WIN = 5
SLIDE_STRIDE = 1
DIM = 64
TOPK_GRAPH = 5
OUT_LAYER_NUM = 1
OUT_LAYER_INTER_DIM = 128
VAL_RATIO = 0.2
DECAY = 0.0
EPOCH_CAP = 30
EARLY_STOP_WIN = 15


def load_flat_data(dataset_name: str):
    """Identical config.json-driven loading to run_baseline_selfeval.py's own load_flat_data --
    same CSVs, same train/test split, same per-channel min-max scaling fit on the training set."""
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
    return (train_dataset.data_normalized.astype(np.float64),
            test_dataset.data_normalized.astype(np.float64),
            test_dataset.labels.astype(int), n_feats)


def make_windows(data_ct: np.ndarray, labels_t: np.ndarray, slide_win: int, stride: int, is_train: bool):
    """Verbatim semantics of the official TimeDataset.process(): data_ct is [n_channels, T]
    (channels x time). ft = data_ct[:, i-slide_win:i] (context), tar = data_ct[:, i] (target),
    label = labels_t[i] (the target timestep's own label) -- training uses `stride`, eval always
    uses stride 1 (range(slide_win, T)), exactly as the official code does."""
    n_channels, T = data_ct.shape
    rng = range(slide_win, T, stride) if is_train else range(slide_win, T)
    xs, ys, labs = [], [], []
    for i in rng:
        xs.append(data_ct[:, i - slide_win:i])
        ys.append(data_ct[:, i])
        labs.append(labels_t[i])
    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    lab = torch.tensor(np.array(labs), dtype=torch.float32)
    return x, y, lab


def get_loaders_indices(n_windows: int, seed: int, val_ratio: float):
    """Verbatim semantics of the official Main.get_loaders(): a single random contiguous block
    (size val_ratio * N) carved out of the first (1 - val_ratio) * N windows is validation; every
    other window is training. Matches the official repo's own (seeded) random.randrange call."""
    rnd = random.Random(seed)
    train_use_len = int(n_windows * (1 - val_ratio))
    val_use_len = int(n_windows * val_ratio)
    val_start = rnd.randrange(train_use_len)
    all_idx = np.arange(n_windows)
    val_idx = all_idx[val_start:val_start + val_use_len]
    train_idx = np.concatenate([all_idx[:val_start], all_idx[val_start + val_use_len:]])
    return train_idx, val_idx


def run_epoch_forward(model, x, y, edge_index, batch_size, device, train_mode, optimizer=None):
    """Mirrors the official train.py/test.py per-batch loop (Adam step if train_mode, else
    no_grad). Returns (avg_loss, predictions, ground_truth) -- predictions/ground_truth only
    populated in eval mode (train mode returns None, None, matching how they're unused there)."""
    loss_fn = nn.MSELoss(reduction='mean')
    n = x.shape[0]
    order = np.random.permutation(n) if train_mode else np.arange(n)
    model.train() if train_mode else model.eval()
    total_loss, n_batches = 0.0, 0
    preds, gts = [], []
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        xb = x[idx].to(device).float()
        yb = y[idx].to(device).float()
        eb = edge_index.to(device).long()
        if train_mode:
            optimizer.zero_grad()
            out = model(xb, eb).float()
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(xb, eb).float()
                loss = loss_fn(out, yb)
        total_loss += loss.item()
        n_batches += 1
        if not train_mode:
            preds.append(out.detach().cpu().numpy())
            gts.append(yb.detach().cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    if not train_mode:
        return avg_loss, np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)
    return avg_loss, None, None


def get_err_scores(predict: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Verbatim formula from the official evaluate.py's get_err_scores + get_err_median_and_iqr,
    called with the SAME array as both the values being scored and the reference for the
    median/IQR (the official code's own convention when scoring the test set -- see module
    docstring's methodological note). predict/gt: [T, n_channels]. Returns smoothed per-sensor
    normalized error scores, same shape."""
    delta = np.abs(predict - gt)
    err_mid = np.median(delta, axis=0)
    err_iqr = scipy_iqr(delta, axis=0)
    epsilon = 1e-2
    err_scores = (delta - err_mid) / (np.abs(err_iqr) + epsilon)
    smoothed = np.zeros_like(err_scores)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed[i] = np.mean(err_scores[i - before_num:i + 1], axis=0)
    return smoothed


def run_one(dataset_name: str, n_feats: int, train_ct: np.ndarray, test_ct: np.ndarray, test_labels: np.ndarray):
    print(f"\n{'='*80}\nGDN on {dataset_name}\n{'='*80}", flush=True)
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    train_labels_zero = np.zeros(train_ct.shape[1], dtype=np.float64)
    x_all, y_all, _ = make_windows(train_ct, train_labels_zero, SLIDE_WIN, SLIDE_STRIDE, is_train=True)
    train_idx, val_idx = get_loaders_indices(len(x_all), SEED, VAL_RATIO)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    print(f"  windows: train={len(x_train)}, val={len(x_val)}", flush=True)

    x_test, y_test, lab_test = make_windows(test_ct, test_labels.astype(np.float64), SLIDE_WIN, 1, is_train=False)

    edge_index = fully_connected_edge_index(n_feats)
    model = GDN([edge_index], n_feats, dim=DIM, input_dim=SLIDE_WIN, out_layer_num=OUT_LAYER_NUM,
                out_layer_inter_dim=OUT_LAYER_INTER_DIM, topk=TOPK_GRAPH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=DECAY)

    best_val_loss = float('inf')
    best_state = None
    stop_count = 0
    epochs_trained = 0
    for epoch in range(EPOCH_CAP):
        train_loss, _, _ = run_epoch_forward(model, x_train, y_train, edge_index, BATCH_SIZE, DEVICE, train_mode=True, optimizer=optimizer)
        val_loss, _, _ = run_epoch_forward(model, x_val, y_val, edge_index, BATCH_SIZE, DEVICE, train_mode=False)
        epochs_trained = epoch + 1
        improved = val_loss < best_val_loss
        print(f"  epoch {epoch+1}/{EPOCH_CAP}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
              + ("  (best)" if improved else f"  (best={best_val_loss:.6f}, stop {stop_count+1}/{EARLY_STOP_WIN})"), flush=True)
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= EARLY_STOP_WIN:
                print(f"  [EARLY STOP] no val improvement for {EARLY_STOP_WIN} epochs, stopping at epoch {epochs_trained}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_pred, test_gt = run_epoch_forward(model, x_test, y_test, edge_index, BATCH_SIZE, DEVICE, train_mode=False)
    err_scores = get_err_scores(test_pred, test_gt)          # [T_test, n_channels]
    final_scores = np.max(err_scores, axis=1)                # topk=1 sensor aggregation (official)

    test_labels_aligned = lab_test.numpy().astype(int)
    metrics = evaluate_all_metrics(test_labels_aligned, final_scores)
    metrics['epochs_trained'] = epochs_trained
    metrics['best_val_loss'] = best_val_loss
    elapsed = time.time() - t0
    print(f"  --> AUC-ROC={metrics['AUC_ROC']:.4f}  AUC-PR={metrics['AUC_PR']:.4f}  "
          f"Point-F1={metrics['Point_F1']:.4f}  VUS-ROC={metrics.get('VUS_ROC', float('nan')):.4f}  "
          f"VUS-PR={metrics.get('VUS_PR', float('nan')):.4f}  "
          f"Aff-F1={metrics.get('Affiliation_F1', float('nan')):.4f}  "
          f"PATE={metrics.get('PATE_AUC_PR', float('nan')):.4f}  epochs={epochs_trained}  ({elapsed:.1f}s)", flush=True)
    return metrics


def per_job_path(dataset_name: str) -> Path:
    """Same naming convention as run_baseline_selfeval.py's per_job_path, so its --combine picks
    this up automatically alongside DAGMM/OmniAnomaly/MAD_GAN."""
    return RESULTS_DIR / f'baseline_selfeval_{dataset_name}_GDN.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS.keys()), required=True)
    args = parser.parse_args()

    p = per_job_path(args.dataset)
    if p.exists():
        print(f"[SKIP] {args.dataset}/GDN: already done ({p})", flush=True)
        return

    train_data, test_data, test_labels, n_feats = load_flat_data(args.dataset)
    train_ct = train_data.T  # [n_channels, T], matching official TimeDataset's own convention
    test_ct = test_data.T
    metrics = run_one(args.dataset, n_feats, train_ct, test_ct, test_labels)
    with open(p, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVE] {p}", flush=True)


if __name__ == '__main__':
    main()

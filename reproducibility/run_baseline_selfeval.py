"""
========================================================================================
SELF-EVALUATION OF DAGMM, OmniAnomaly, MAD-GAN ON OUR OWN SWaT/WADI TEST DATA
========================================================================================
Fills a real gap: DAGMM, OmniAnomaly, and MAD-GAN's own papers, and every modern benchmark
paper we could locate (PatchAD, PatchTrAD, PATE2024, TSB-AD, GDN's own paper), report ONLY
Point-F1/Precision/Recall for these three methods on SWaT/WADI -- never AUC-ROC, VUS-ROC,
VUS-PR, Affiliation-F1, or PATE, because they are 2018-2019-era CPS-anomaly-detection methods
that predate this sub-field's adoption of threshold-independent/range-based metrics.

This script closes that gap by evaluating them ourselves from official repositories: it trains
each method's OFFICIAL architecture using its OFFICIAL training/scoring procedure (both vendored
verbatim from imperial-qore/TranAD's own src/models.py and main.py -- see
reproducibility/baselines/tranad_bundle.py for exact provenance) on the EXACT same preprocessed
SWaT/WADI data already used for TALON's own headline results (same CSV files, same train/test
split, same per-channel min-max scaling fit on the training set only, via
`datasets.LocalTSAD.load_csv_dataset` with the same config.json parameters as
`evaluate_swat_bestfull.py`), then scores the resulting per-timestep reconstruction error with
the IDENTICAL metric pipeline used for TALON (`evaluate_swat_bestfull.evaluate_all_metrics`:
AUC-ROC/AUC-PR/Point-F1/VUS-ROC/VUS-PR/Affiliation-F1/PATE, via TSB_AD's `get_metrics` and the
`pate` package). This is what makes the resulting numbers genuinely comparable to TALON's,
unlike e.g. TSB-AD's own SWaT numbers (a completely different per-episode protocol that produces
VUS-PR values 5-10x lower for the same methods).

SCOPE -- why these three and not GDN/USAD:
  - GDN's official repository (d-ailin/GDN) is a forecasting model with a genuinely different
    training/scoring procedure than these three's reconstruction-based approach, so it has its
    own dedicated script instead (run_gdn_selfeval.py).
  - USAD is deliberately NOT re-run here: it has its own dedicated script instead
    (run_usad_published.py), which reproduces the original paper's own training setup rather
    than TranAD's reimplementation vendored in tranad_bundle.py.

TRAINING PROCEDURE FIDELITY: the per-model loss functions and window-construction logic below
(`convert_to_windows`, the DAGMM/OmniAnomaly/MAD-GAN branches of `backprop`) are copied near-verbatim
from TranAD's own `main.py`/`src/models.py` (AdamW + StepLR(5, 0.9), same per-model loss formulas) --
the same official benchmark suite's own training recipe for these three methods, not a from-scratch
reimplementation. Two substantive differences from TranAD's own `main.py`:
  (1) Data source: instead of TranAD's own `preprocess.py` output, we feed in TALON's own preprocessed
      arrays so every baseline sees the identical, already-published-elsewhere test split.
  (2) Epoch count: TranAD's own main.py trains every model for a fixed 5 epochs regardless of
      architecture. A pilot run showed DAGMM's training loss still visibly falling at epoch 5 on this
      data -- a fixed budget tuned (implicitly) for TranAD's own architecture, not these three very
      different and much smaller models, risks under-training some of them, which would unfairly
      *disadvantage* these baselines rather than favor them. Instead, each model trains until its own
      validation loss plateaus: the last VAL_FRACTION=10% of the (nominal-only) training data is held
      out chronologically as a validation split -- matching TALON's own `val_split: 0.1` training
      convention -- and training stops after PATIENCE=3 consecutive epochs with no improvement in
      validation loss (capped at MAX_EPOCHS=50), reverting to the best-validation-loss checkpoint
      before scoring the test set. Zero test-label information is used anywhere in this stopping
      criterion. `epochs_trained` is recorded per model/dataset in the output JSON for transparency.

USAGE:
------
  Sequential (all 6 dataset/model jobs, one at a time):
    python reproducibility/run_baseline_selfeval.py

  Parallel (one job per process -- each model is tiny, so the GPU has ample headroom to run
  several at once; each job writes its OWN result file, so there is no shared-state race
  condition between concurrently-running processes):
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset SWaT --model DAGMM
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset SWaT --model OmniAnomaly
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset SWaT --model MAD_GAN
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset WADI --model DAGMM
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset WADI --model OmniAnomaly
    ...\\python.exe reproducibility/run_baseline_selfeval.py --dataset WADI --model MAD_GAN

  After some/all of the above finish (safe to run any time, even mid-run, to check progress):
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
from torch.optim.lr_scheduler import StepLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

BASELINES_DIR = PROJECT_ROOT / 'reproducibility' / 'baselines'
sys.path.insert(0, str(BASELINES_DIR))

from datasets.LocalTSAD import load_csv_dataset
import tranad_bundle as tb

sys.path.insert(0, str(PROJECT_ROOT / 'reproducibility'))
from evaluate_swat_bestfull import evaluate_all_metrics

RESULTS_DIR = PROJECT_ROOT / 'reproducibility' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

_forced = os.environ.get('BASELINE_DEVICE', '').strip().lower()
if _forced in ('cpu', 'cuda'):
    DEVICE = torch.device(_forced)
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    torch.set_num_threads(1)  # each window is a tiny op; per-op thread-pool overhead dominates
    # otherwise (measured: ~5x faster single-threaded than PyTorch's default multi-threading here)
print(f"[DEVICE] Using {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else "")
      + "  (override with env var BASELINE_DEVICE=cpu|cuda)", flush=True)

HEARTBEAT_EVERY = 5000  # print a liveness/progress line every N windows within a long epoch

# Train-until-convergence, not TranAD's own fixed 5-epoch default: a smoke test showed DAGMM's
# training loss still visibly falling at epoch 5 on this data (under-trained by a fixed budget
# picked for TranAD's own architecture, not these three), while MAD-GAN's adversarial loss is
# non-monotonic by nature. Early-stopping on a held-out slice of the NOMINAL training data --
# chronological last 10%, matching TALON's own `val_split: 0.1` training convention -- lets each
# method train to its own natural convergence point with zero test-label leakage, rather than
# imposing one architecture's convergence speed on all three or hand-picking a bigger fixed number.
VAL_FRACTION = 0.1
MAX_EPOCHS = 50
PATIENCE = 3  # stop after this many consecutive epochs with no validation-loss improvement

DATASETS = {
    'SWaT': PROJECT_ROOT / 'results' / 'swat_cve' / 'BestFull' / 'config.json',
    'WADI': PROJECT_ROOT / 'results' / 'wadi_vae' / 'WADI' / 'BestFullChannels' / 'config.json',
}

MODEL_CLASSES = {
    'DAGMM': tb.DAGMM,
    'OmniAnomaly': tb.OmniAnomaly,
    'MAD_GAN': tb.MAD_GAN,
    'TranAD': tb.TranAD,
    'Random': None,
}


def load_flat_data(dataset_name: str):
    """Loads train/test as flat [T, F] normalized arrays + per-timestep labels, using the EXACT
    same config.json parameters (CSV paths, column overrides, scaling, downsampling) as
    evaluate_swat_bestfull.py and every other TALON evaluation script."""
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

    trainD = torch.tensor(train_dataset.data_normalized, dtype=torch.float64)
    testD = torch.tensor(test_dataset.data_normalized, dtype=torch.float64)
    test_labels = test_dataset.labels.astype(int)
    return trainD, testD, test_labels, n_feats


def convert_to_windows(data: torch.Tensor, w_size: int) -> torch.Tensor:
    """Verbatim semantics of TranAD main.py's convert_to_windows(): window j = data[j-w_size:j]
    (or left-padded with repeats of the first row when j < w_size) -- so window j's LAST element is
    data[j-1], NEVER data[j] itself.

    BUG FIXED (found while validating the TranAD self-eval numbers against literature): an earlier
    version of this function built window j ending at data[j] (verified numerically: window[j][-1]
    == data[j] for every j, via direct construction on a toy 0..9 sequence). That's harmless for
    DAGMM/OmniAnomaly/MAD-GAN, whose loss reconstructs the ENTIRE window against itself (every
    position contributes equally, so which end is nominally "the target" doesn't matter to their
    task) -- but it is a genuine data leak for TranAD specifically, whose loss is computed only
    against `elem = window[-1]`: the value being reconstructed was also sitting inside the model's
    own input. This let TranAD trivially copy its answer instead of genuinely predicting from prior
    context, and explains why MORE training made it WORSE, not better (best validation
    reconstruction loss improved from 0.00513 to 0.00122 across 5->50 epochs while every detection
    metric collapsed -- the signature of a model exploiting a shortcut more efficiently, not
    ordinary overfitting). Fixed by shifting the input by one position before windowing, so the
    windowed value at position j is built from data up to and including j-1 only, exactly matching
    official semantics (verified numerically against the same toy sequence: window[j][-1] ==
    data[j-1] for every j, with the same all-data[0] boundary behavior official uses for j < w_size).
    """
    T = data.shape[0]
    shifted = torch.cat([data[0:1], data[:-1]], dim=0)  # shifted[k] = data[k-1] (data[0] for k=0)
    padded = torch.cat([data[0:1].repeat(w_size - 1, 1), shifted], dim=0)
    windows = padded.unfold(0, w_size, 1).permute(0, 2, 1)  # [T, w_size, feats]
    return windows.reshape(T, -1)


def _heartbeat(i, n, t0):
    if i > 0 and i % HEARTBEAT_EVERY == 0:
        rate = i / (time.time() - t0)
        eta = (n - i) / max(rate, 1e-9)
        print(f'    ...{i}/{n} windows ({rate:.0f}/s, ETA {eta:.0f}s)', flush=True)


def backprop_dagmm(epoch, model, data, optimizer, scheduler, training):
    l = nn.MSELoss(reduction='none')
    if training:
        l1s, l2s = [], []
        t0 = time.time()
        for i, d in enumerate(data):
            _, x_hat, z, gamma = model(d)
            l1, l2 = l(x_hat, d), l(gamma, d)
            l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
            loss = torch.mean(l1) + torch.mean(l2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            _heartbeat(i, len(data), t0)
        scheduler.step()
        print(f'  Epoch {epoch}\tL1={np.mean(l1s):.6f}\tL2={np.mean(l2s):.6f}\t({time.time()-t0:.1f}s)', flush=True)
        return
    else:
        ae1s = []
        with torch.no_grad():
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
        ae1s = torch.stack(ae1s)
        loss = l(ae1s, data)
        return loss.detach().cpu().numpy()


def backprop_omnianomaly(epoch, model, data, optimizer, scheduler, training):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    if training:
        mses, klds = [], []
        hidden = None
        t0 = time.time()
        for i, d in enumerate(data):
            y_pred, mu, logvar, hidden = model(d, hidden if i else None)
            MSE = l(y_pred, d)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
            loss = MSE + model.beta * KLD
            mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            hidden = hidden.detach()
            _heartbeat(i, len(data), t0)
        scheduler.step()
        print(f'  Epoch {epoch}\tMSE={np.mean(mses):.6f}\tKLD={np.mean(klds):.6f}\t({time.time()-t0:.1f}s)', flush=True)
        return
    else:
        y_preds = []
        hidden = None
        with torch.no_grad():
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
        y_pred = torch.stack(y_preds)
        MSE = l(y_pred, data)
        return MSE.detach().cpu().numpy()


def backprop_madgan(epoch, model, data, optimizer, scheduler, training):
    l = nn.MSELoss(reduction='none')
    bcel = nn.BCELoss(reduction='mean')
    msel = nn.MSELoss(reduction='mean')
    real_label = torch.tensor([0.9], dtype=torch.float64, device=data.device)
    fake_label = torch.tensor([0.1], dtype=torch.float64, device=data.device)
    if training:
        mses, gls, dls = [], [], []
        t0 = time.time()
        for i, d in enumerate(data):
            model.discriminator.zero_grad()
            _, real, fake = model(d)
            dl = bcel(real, real_label) + bcel(fake, fake_label)
            dl.backward()
            model.generator.zero_grad()
            optimizer.step()
            z, _, fake = model(d)
            mse = msel(z, d)
            gl = bcel(fake, real_label)
            tl = gl + mse
            tl.backward()
            model.discriminator.zero_grad()
            optimizer.step()
            mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
            _heartbeat(i, len(data), t0)
        print(f'  Epoch {epoch}\tMSE={np.mean(mses):.6f}\tG={np.mean(gls):.6f}\tD={np.mean(dls):.6f}\t({time.time()-t0:.1f}s)', flush=True)
        return
    else:
        outputs = []
        with torch.no_grad():
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
        outputs = torch.stack(outputs)
        loss = l(outputs, data)
        return loss.detach().cpu().numpy()


def backprop_tranad(epoch, model, data, optimizer, scheduler, training):
    l = nn.MSELoss(reduction='none')
    feats = model.n_feats
    T = data.shape[0]
    windows_T_W_F = data.view(T, model.n_window, feats)
    bs = model.batch if training else 1024
    
    if training:
        l1s = []
        t0 = time.time()
        n = epoch + 1
        for i in range(0, T, bs):
            batch_windows = windows_T_W_F[i:i+bs].permute(1, 0, 2)
            local_bs = batch_windows.shape[1]
            elem = batch_windows[-1, :, :].view(1, local_bs, feats)
            
            z = model(batch_windows, elem)
            if isinstance(z, tuple):
                l1 = (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
            else:
                l1 = l(z, elem)
                
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            _heartbeat(i, T, t0)
            
        scheduler.step()
        print(f'  Epoch {epoch}\tL1={np.mean(l1s):.6f}\t({time.time()-t0:.1f}s)', flush=True)
        return
    else:
        losses = []
        with torch.no_grad():
            for i in range(0, T, bs):
                batch_windows = windows_T_W_F[i:i+bs].permute(1, 0, 2)
                local_bs = batch_windows.shape[1]
                elem = batch_windows[-1, :, :].view(1, local_bs, feats)
                z = model(batch_windows, elem)
                if isinstance(z, tuple):
                    z = z[1]
                batch_loss = l(z, elem)[0] # shape [local_bs, feats]
                losses.append(batch_loss)
        losses = torch.cat(losses, dim=0)
        return losses.detach().cpu().numpy()


BACKPROP = {
    'DAGMM': backprop_dagmm,
    'OmniAnomaly': backprop_omnianomaly,
    'MAD_GAN': backprop_madgan,
    'TranAD': backprop_tranad,
}


def run_one(dataset_name, model_name, trainD_flat, testD_flat, test_labels, n_feats, seed=None):
    print(f"\n{'='*80}\n{model_name} on {dataset_name}" + (f" (seed={seed})" if seed is not None else "") + f"\n{'='*80}", flush=True)
    if seed is not None:
        # Same convention as TALON's own multi-seed sweeps (final_table_rows.py,
        # nmc_sweep_pipeline.py): seeding here controls weight init and any stochastic
        # ops during training, so repeated calls with different seeds are genuinely independent
        # trials rather than accidentally sharing whatever the ambient RNG state happens to be.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if model_name == 'Random':
        scores = np.random.uniform(0, 1, size=len(test_labels))
        metrics = evaluate_all_metrics(test_labels, scores)
        print(f"  --> AUC-ROC={metrics['AUC_ROC']:.4f}  AUC-PR={metrics['AUC_PR']:.4f}  Point-F1={metrics['Point_F1']:.4f}  VUS-ROC={metrics.get('VUS_ROC', float('nan')):.4f}  VUS-PR={metrics.get('VUS_PR', float('nan')):.4f}  Aff-F1={metrics.get('Affiliation_F1', float('nan')):.4f}  PATE={metrics.get('PATE_AUC_PR', float('nan')):.4f}", flush=True)
        return metrics

    t0 = time.time()
    model_class = MODEL_CLASSES[model_name]
    model = model_class(n_feats).double().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, 5, 0.9)
    backprop = BACKPROP[model_name]

    trainD_flat, testD_flat = trainD_flat.to(DEVICE), testD_flat.to(DEVICE)

    # Chronological train/val split of the NOMINAL training data only (last VAL_FRACTION held
    # out) -- zero test-label leakage, matches TALON's own val_split=0.1 training convention.
    n_val = int(len(trainD_flat) * VAL_FRACTION)
    train_core, val_core = trainD_flat[:-n_val], trainD_flat[-n_val:]

    if model_name == 'OmniAnomaly':
        # OmniAnomaly's own forward() consumes ONE raw timestep at a time and carries temporal
        # context via its GRU's recurrent `hidden` state instead of explicit windowing -- per
        # TranAD's own main.py, OmniAnomaly is deliberately excluded from convert_to_windows().
        trainD, valD, testD = train_core, val_core, testD_flat
    else:
        trainD = convert_to_windows(train_core, model.n_window)
        valD = convert_to_windows(val_core, model.n_window)
        testD = convert_to_windows(testD_flat, model.n_window)

    best_val_loss = float('inf')
    best_state = None
    patience_left = PATIENCE
    epochs_trained = 0
    max_epochs = 5 if model_name == 'TranAD' else MAX_EPOCHS
    for e in range(max_epochs):
        print(f"  [epoch {e+1}/{MAX_EPOCHS}]", flush=True)
        backprop(e, model, trainD, optimizer, scheduler, training=True)
        epochs_trained = e + 1

        model.eval()
        val_loss = float(np.mean(backprop(e, model, valD, optimizer, scheduler, training=False)))
        model.train()
        improved = val_loss < best_val_loss - 1e-6
        print(f'    val_loss={val_loss:.6f}' + ('  (best)' if improved else f'  (best={best_val_loss:.6f}, patience {PATIENCE - patience_left + 1}/{PATIENCE})'), flush=True)
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'  [EARLY STOP] no val improvement for {PATIENCE} epochs, stopping at epoch {epochs_trained}', flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    loss = backprop(0, model, testD, optimizer, scheduler, training=False)  # [T, feats]
    scores = np.mean(loss, axis=1)  # per-timestep score, same aggregation as TranAD main.py

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


MASTER_RESULTS_PATH = RESULTS_DIR / 'baseline_selfeval_results.json'


def per_job_path(dataset_name, model_name, seed=None):
    """Each (dataset, model[, seed]) job owns exactly one file -- this is what makes it safe to run
    several jobs concurrently as separate processes: no two processes ever write the same file,
    so there is no read-modify-write race of the kind a single shared JSON would have. Seeded runs
    (used for multi-seed sweeps) get their own suffixed filename so they never collide with, or
    overwrite, the original unseeded single-run result for the same (dataset, model)."""
    suffix = f'_seed{seed}' if seed is not None else ''
    return RESULTS_DIR / f'baseline_selfeval_{dataset_name}_{model_name}{suffix}.json'


# GDN is trained by the separate run_gdn_selfeval.py (a genuinely different forecasting-based
# pipeline -- see that script's own module docstring), but writes to this same per_job_path
# naming convention, so it's included here purely for --combine/print_summary bookkeeping, not
# because this script knows how to train it.
ALL_MODEL_NAMES = list(MODEL_CLASSES.keys()) + ['GDN']


def run_job(dataset_name, model_name, trainD_flat=None, testD_flat=None, test_labels=None, n_feats=None, seed=None):
    if trainD_flat is None:
        trainD_flat, testD_flat, test_labels, n_feats = load_flat_data(dataset_name)
    metrics = run_one(dataset_name, model_name, trainD_flat, testD_flat, test_labels, n_feats, seed=seed)
    p = per_job_path(dataset_name, model_name, seed=seed)
    with open(p, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVE] {p}", flush=True)
    return metrics


def print_summary(all_results):
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}", flush=True)
    for dataset_name in DATASETS:
        for model_name in ALL_MODEL_NAMES:
            m = all_results.get(dataset_name, {}).get(model_name)
            if m is None:
                print(f"{dataset_name:6s} {model_name:12s}  [not finished yet]", flush=True)
                continue
            print(f"{dataset_name:6s} {model_name:12s} "
                  f"AUC-ROC={m['AUC_ROC']:.4f} AUC-PR={m['AUC_PR']:.4f} Point-F1={m['Point_F1']:.4f} "
                  f"VUS-ROC={m.get('VUS_ROC', float('nan')):.4f} VUS-PR={m.get('VUS_PR', float('nan')):.4f} "
                  f"Aff-F1={m.get('Affiliation_F1', float('nan')):.4f} PATE={m.get('PATE_AUC_PR', float('nan')):.4f} "
                  f"epochs={m.get('epochs_trained', '?')}", flush=True)


def combine_results():
    """Scans every per-job file that exists so far (regardless of whether it ran sequentially or
    as one of several parallel processes) and (re)writes the single combined master JSON. Safe to
    call at any time, including while other jobs are still running -- it only ever reads files
    other processes have already finished writing."""
    combined = {}
    for dataset_name in DATASETS:
        for model_name in ALL_MODEL_NAMES:
            p = per_job_path(dataset_name, model_name)
            if p.exists():
                with open(p) as f:
                    combined.setdefault(dataset_name, {})[model_name] = json.load(f)
    with open(MASTER_RESULTS_PATH, 'w') as f:
        json.dump(combined, f, indent=2)
    n_done = sum(len(v) for v in combined.values())
    print(f"[COMBINE] {n_done}/{len(DATASETS) * len(ALL_MODEL_NAMES)} jobs done -> wrote {MASTER_RESULTS_PATH}", flush=True)
    print_summary(combined)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS.keys()),
                         help='Run only this dataset (must be paired with --model). Omit both '
                              '--dataset and --model to run all 6 jobs sequentially in one process.')
    parser.add_argument('--model', choices=list(MODEL_CLASSES.keys()), help='See --dataset.')
    parser.add_argument('--seed', type=int, default=None,
                         help='Set an explicit random seed (weight init + stochastic training ops) '
                              'and write to a seed-suffixed result file instead of the default '
                              'unseeded one. Used for multi-seed sweeps; omit for a single run.')
    parser.add_argument('--combine', action='store_true',
                         help='Merge whatever per-job result files exist so far into the combined '
                              'master JSON and print a status summary; trains nothing.')
    args = parser.parse_args()

    if args.combine:
        combine_results()
        return

    if bool(args.dataset) != bool(args.model):
        parser.error('--dataset and --model must be given together')

    if args.dataset and args.model:
        # Single-job mode: safe to launch many of these as separate concurrent processes.
        p = per_job_path(args.dataset, args.model, seed=args.seed)
        if p.exists():
            print(f"[SKIP] {args.dataset}/{args.model}" + (f" seed={args.seed}" if args.seed is not None else "") + f": already done ({p})", flush=True)
            return
        run_job(args.dataset, args.model, seed=args.seed)
        return

    # Default: run everything sequentially in this one process (loads each dataset's data once,
    # reused across its 3 models).
    for dataset_name in DATASETS:
        models_left = [m for m in MODEL_CLASSES if not per_job_path(dataset_name, m).exists()]
        if not models_left:
            print(f"[SKIP] {dataset_name}: all models already done", flush=True)
            continue
        trainD_flat, testD_flat, test_labels, n_feats = load_flat_data(dataset_name)
        for model_name in models_left:
            run_job(dataset_name, model_name, trainD_flat, testD_flat, test_labels, n_feats)

    combine_results()


if __name__ == '__main__':
    main()

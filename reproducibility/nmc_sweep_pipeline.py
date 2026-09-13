"""
========================================================================================
MONTE CARLO SAMPLE-SIZE SWEEP UNDER THE SHIPPED PIPELINE (SWaT AND WADI)
========================================================================================
Replaces two earlier per-dataset sweep scripts that scored with the clipped
importance-sampling estimator (clamping log q_phi - log q_psi to [-10, 10]) and dataset-specific
aggregation/fusion choices no longer in the pipeline (WADI's 0.85 median + 0.15 causal blend
with Smooth-Max fusion; SWaT's rate-limited drift correction). This script uses the shipped
configuration for whichever dataset it is pointed at:

    score      -log p(Y|X) ~= -log[(1/K) sum_k p(Y|z_k)],  z_k ~ q_psi(z|X)   (no ratio)
    aggregate  plain median over the overlapping windows covering each timestep
    consensus  SPEC branch (SWaT student decode, WADI teacher decode), channel-precision
               weighted and train-calibrated
    fusion     S(t) = z_CNLL(t) + lambda * z_cons(t),  lambda = 1  (both benchmarks)
    OOD clamp  SPEC value (SWaT 1.0, WADI 10.0)

WHY ONE EXTRACTION PER SEED SUFFICES
    The estimator reduces over the K samples only at the very last step (a logsumexp), so a
    single K=200 pass that RETAINS the per-sample log p(Y|z_k) matrix can be sliced to any
    N_MC <= 200. Cost is O(seeds), not O(seeds x grid). The consensus term and the train-side
    calibration constants do not depend on N_MC and are computed once.

Slicing to the first N_MC columns is a valid draw because the samples are i.i.d. given the
window, so no re-sampling is needed per grid point.
========================================================================================
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
from runtime_guard import cap_threads as _cap, preflight as _preflight
_cap()

import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = PROJECT_ROOT / 'reproducibility'
for _p in (str(PROJECT_ROOT), str(REPRO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

from final_table_rows import (SPEC, COLS, extract, chan_weights, load_dataset_and_model,
                              SWAT_LAMBDA)
from evaluate_swat_stage2_ablations import build_student
from rebuild_swat_ablation_bestfull import median_aggregate, build_consensus_variance
from evaluate_swat_bestfull import evaluate_all_metrics

N_MC_GRID = [50, 75, 100, 125, 150, 175, 200]


def extract_persample(cve, ds, W, device, K, ood, K_sub=50):
    """`final_table_rows.extract`, but returning the per-sample log p(Y|z_k) matrix
    [n_windows, K] instead of reducing it, plus both decodes for the consensus term."""
    total_len = len(ds.labels)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    logp, yh_s, yh_t, slices, cur = [], [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            tm = torch.ones(xb.size(0), W, device=device, dtype=torch.bool)
            B, C = xb.size(0), yb.size(2)
            nv, var = W * C, 1.0 / C
            const = (nv / 2.0) * math.log(2.0 * math.pi * var)
            for _ in range(B):
                ws, we = cur * 10, cur * 10 + W
                if we <= total_len:
                    slices.append((ws, we))
                cur += 1
            acc = []
            for p in range(math.ceil(K / K_sub)):
                k = min(K_sub, K - p * K_sub)
                xr, yr = xb.repeat_interleave(k, 0), yb.repeat_interleave(k, 0)
                mr = tm.repeat_interleave(k, 0)
                yc = torch.clamp(yr, -ood, ood) if ood is not None else yr
                s = cve(x_condition=xr, y_patches=yc, mode='testRandom', y_target=yr,
                        time_mask_full=mr, ood_threshold=ood)
                mse = s['reconstruction_loss_condition'].view(B, k)
                acc.append(-(mse * nv) / (2.0 * var) - const)
            logp.append(torch.cat(acc, 1).cpu())
            sm = cve(x_condition=xb, y_patches=yb, mode='test', y_target=yb,
                     time_mask_full=tm, ood_threshold=ood)
            yh_s.append(sm['y_hat_condition'].cpu())
            yh_t.append(sm['y_hat_target'].cpu())
    return {'logp': torch.cat(logp, 0), 'y_hat_student': torch.cat(yh_s, 0),
            'y_hat_teacher': torch.cat(yh_t, 0),
            'slices': slices, 'total_len': total_len}


def main():
    _preflight(label="nmc_sweep_wadi")
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='WADI')
    ap.add_argument('--n_seeds', type=int, default=10)
    ap.add_argument('--K_test', type=int, default=200)
    ap.add_argument('--K_train', type=int, default=50)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    seeds = [42 + i * 1337 for i in range(args.n_seeds)]
    if args.out is None:
        args.out = 'results/nmc_sweep_%s_corrected.json' % args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DS = args.dataset
    sp = SPEC[DS]
    cons_key = 'y_hat_%s' % sp['cons']
    print("N_MC SWEEP | %s | device=%s | seeds=%s" % (DS, device, seeds), flush=True)
    print("grid=%s | ood=%s | consensus=%s | lambda=%s\n"
          % (N_MC_GRID, sp['ood'], sp['cons'], SWAT_LAMBDA), flush=True)

    train_ds, test_ds, W, y_true, vae, ck, m_cfg = load_dataset_and_model(DS, device)
    cve = build_student(ck, vae, m_cfg, device)[0]
    cw = chan_weights(train_ds, cve, device, sp['ood'])
    cve.channel_weights.data.copy_(torch.tensor(cw, dtype=torch.float32, device=device))

    # --- train calibration: independent of N_MC, so done once ---
    tr = extract(cve, train_ds, W, device, args.K_train, sp['ood'],
                 max_batches=sp['train_batches'])
    mu, sd = float(np.mean(tr['cnll'])), float(np.std(tr['cnll']))
    tr_sl, tr_len = tr['slices'], len(tr['slices']) * 10 + 100
    trc = build_consensus_variance(tr[cons_key], tr_sl, tr_len, cw)
    cnt = np.zeros(tr_len, dtype=int)
    for (a, b) in tr_sl:
        cnt[a:b] += 1
    mc, sc = float(np.mean(trc[cnt > 1])), float(np.std(trc[cnt > 1]))
    print("calibration: cnll mu=%.4f sd=%.4f | cons mu=%.5f sd=%.5f\n" % (mu, sd, mc, sc),
          flush=True)
    del tr
    torch.cuda.empty_cache()

    y = np.asarray(y_true)
    sweep = {str(k): [] for k in N_MC_GRID}
    for i, s in enumerate(seeds):
        t0 = time.time()
        torch.manual_seed(s)
        np.random.seed(s)
        te = extract_persample(cve, test_ds, W, device, args.K_test, sp['ood'])
        cons = (build_consensus_variance(te[cons_key], te['slices'],
                                         te['total_len'], cw) - mc) / (sc + 1e-6)
        for k in N_MC_GRID:
            cn = -(torch.logsumexp(te['logp'][:, :k], dim=1) - math.log(k)).numpy()
            z1 = (median_aggregate(cn, te['slices'], te['total_len']) - mu) / (sd + 1e-6)
            m = evaluate_all_metrics(y, z1 + SWAT_LAMBDA * cons)
            sweep[str(k)].append({kk: m.get(kk) for kk, _ in COLS})
        last = sweep['200'][-1]
        print("  seed %2d/%d (=%d) done in %.0fs  [K=200: F1=%.4f AUCROC=%.4f]"
              % (i + 1, len(seeds), s, time.time() - t0, last['Point_F1'], last['AUC_ROC']),
              flush=True)
        del te
        torch.cuda.empty_cache()

    summ = {}
    for k in N_MC_GRID:
        rows = sweep[str(k)]
        summ[str(k)] = {}
        for kk, _ in COLS:
            v = [r[kk] for r in rows if r[kk] is not None]
            if v:
                summ[str(k)][kk] = {'mean': float(np.mean(v)),
                                    'std': float(np.std(v, ddof=1)), 'n': len(v)}

    with open(PROJECT_ROOT / args.out, 'w') as f:
        json.dump({'dataset': DS, 'seeds': seeds, 'grid': N_MC_GRID, 'ood': sp['ood'],
                   'consensus_branch': sp['cons'], 'lambda': SWAT_LAMBDA,
                   'per_seed': sweep, 'summary': summ}, f, indent=2)

    print("\n" + "=" * 110, flush=True)
    print("%6s" % "N_MC" + "".join("%14s" % p for _, p in COLS), flush=True)
    for k in N_MC_GRID:
        cells = []
        for kk, _ in COLS:
            if kk in summ[str(k)]:
                cells.append("%8.4f+-%.4f" % (summ[str(k)][kk]['mean'],
                                              summ[str(k)][kk]['std']))
            else:
                cells.append("%14s" % "--")
        print("%6d" % k + "".join(cells), flush=True)
    print("\n[SAVED] %s" % (PROJECT_ROOT / args.out), flush=True)


if __name__ == '__main__':
    main()

"""
STATUS: ACTIVE

========================================================================================
MODEL-FREE BASELINES FOR SWaT AND WADI
========================================================================================
While auditing why a randomly-initialised student scored as well as it did on SWaT, the
squared distance of Y from its nominal training mean turned out to reach AUC-ROC 0.788 and
AUC-PR 0.693 -- competitive with several deep baselines in the paper's Table I, and ahead of
GDN on both. No model of any kind is involved.

That is a fact about the benchmark, not about any method, and it belongs in the comparison
table: a reader cannot judge whether a deep detector is doing useful work without knowing what
a one-line statistic achieves on the same data. This script computes three such baselines
across the full 7-metric spectrum for both datasets.

    mean_L2       mean_c (Y_tc - mu_c)^2                  raw distance from the nominal mean
    zscore_L2     mean_c ((Y_tc - mu_c) / sigma_c)^2      scale-corrected; the fair comparison,
                                                          since TALON's pipeline applies its own
                                                          per-channel precision weighting
    max_abs_z     max_c |Y_tc - mu_c| / sigma_c           worst-channel excursion

Each is reported raw (per timestep) and smoothed with a centred rolling median of width equal
to the model's window (100), because TALON aggregates a window-level score across overlapping
windows and an unsmoothed point statistic would otherwise be handicapped by comparison.

mu and sigma come from the TRAINING split only -- these baselines are strictly train-calibrated,
the same discipline the paper claims for its own operating point.
========================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch                       # before pandas: DLL load order in this conda env
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = PROJECT_ROOT / 'reproducibility'
for _p in (str(PROJECT_ROOT), str(REPRO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

from datasets.LocalTSAD import load_csv_dataset
from evaluate_swat_bestfull import evaluate_all_metrics

# Point Precision/Recall are reported alongside the ranked metrics so this row can sit next
# to every other row in the table.
METRICS = [('Point_F1', 'PointF1'), ('Precision', 'Prec'), ('Recall', 'Rec'),
           ('AUC_ROC', 'AUC-ROC'), ('AUC_PR', 'AUC-PR'),
           ('VUS_ROC', 'VUS-ROC'), ('VUS_PR', 'VUS-PR'), ('Affiliation_F1', 'Affil'),
           ('PATE_AUC_PR', 'PATE')]

CFG = {
    'SWaT': 'results/swat_cve/BestFull/config.json',
    'WADI': 'results/wadi_vae/WADI/BestFullChannels/config.json',
}
# Uniform-random scores, averaged over this many draws. A single draw is noisy on the
# range-aware metrics; the mean over draws is the reference row's value.
RANDOM_DRAWS = 10
RANDOM_SEED = 42
# TALON's reported rows under the corrected conditional score (10-seed mean, K=200), as
# produced by `final_table_rows.py`. These are the values in the thesis and paper SOTA
# tables; the model-free baselines below are compared against them.
TALON_REFERENCE = {
    'SWaT': dict(Point_F1=.7539, Precision=.8786, Recall=.6603, AUC_ROC=.8976, AUC_PR=.7789,
                 VUS_ROC=.8183, VUS_PR=.6238, Affiliation_F1=.7288, PATE_AUC_PR=.8005),
    'WADI': dict(Point_F1=.5056, Precision=.4432, Recall=.5885, AUC_ROC=.8155, AUC_PR=.3388,
                 VUS_ROC=.7591, VUS_PR=.3845, Affiliation_F1=.7178, PATE_AUC_PR=.4088),
}
PUBLISHED_TALON = TALON_REFERENCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['SWaT', 'WADI'])
    ap.add_argument('--out', type=str, default='results/trivial_baselines.json')
    args = ap.parse_args()

    out = {}
    for ds in args.datasets:
        d = json.load(open(PROJECT_ROOT / CFG[ds]))['dataset']
        tr, te = load_csv_dataset(
            data_root=os.path.join(str(PROJECT_ROOT), d['data_root']),
            normal_csv=d['normal_csv'], attack_csv=d['attack_csv'],
            label_column=d['label_column'], timestamp_columns=d['timestamp_columns'],
            x_prefixes=d['x_prefixes'], y_prefixes=d['y_prefixes'],
            window_size=d['window_size'], stride=10,
            x_override=d.get('x_override'), y_override=d.get('y_override'),
            drop_columns=d.get('drop_columns', []),
            scaler_type=d.get('scaler_type', 'minmax'),
            downsample_rate=d.get('sampling_rate_seconds', 10),
            downsample_mode=d.get('downsample_mode', 'median'))

        Ytr = np.asarray(tr.data_normalized)[:, tr.y_indices].astype(np.float64)
        Yte = np.asarray(te.data_normalized)[:, te.y_indices].astype(np.float64)
        y = np.asarray(te.labels)[:Yte.shape[0]]
        W = d['window_size']

        mu = Ytr.mean(axis=0)
        sd = Ytr.std(axis=0) + 1e-8
        dev = Yte - mu
        z = dev / sd

        cand = {
            'mean_L2': (dev ** 2).mean(axis=1),
            'zscore_L2': (z ** 2).mean(axis=1),
            'max_abs_z': np.abs(z).max(axis=1),
        }

        print(f"\n{'='*104}\n  {ds}: model-free baselines "
              f"(Y {Yte.shape}, anomaly rate {y.mean():.3f}, window {W})\n{'='*104}", flush=True)
        print(f"  {'baseline':26s}" + "".join(f"{p:>10s}" for _, p in METRICS), flush=True)
        res = {}
        for nm, s in cand.items():
            for smooth in (False, True):
                v = (pd.Series(s).rolling(W, center=True, min_periods=1).median().values
                     if smooth else s)
                key = nm + ('_smoothed' if smooth else '')
                m = evaluate_all_metrics(y, v)
                res[key] = {k: m.get(k) for k, _ in METRICS}
                print(f"  {key:26s}" +
                      "".join(f"{m.get(k, float('nan')):10.4f}" for k, _ in METRICS), flush=True)
        # Uniform-random reference: the floor every metric must be read against. On a
        # benchmark as sparse as GHL_full (0.7% positives) this is what makes an absolute
        # AUC-PR of 0.12 interpretable.
        rng = np.random.default_rng(RANDOM_SEED)
        draws = [evaluate_all_metrics(y, rng.random(len(y))) for _ in range(RANDOM_DRAWS)]
        rand_row = {}
        for k, _ in METRICS:
            v = [d[k] for d in draws if d.get(k) is not None]
            if v:
                rand_row[k] = float(np.mean(v))
        res['random_classifier'] = rand_row
        print(f"  {'random_classifier':26s}" +
              "".join(f"{rand_row.get(k, float('nan')):10.4f}" for k, _ in METRICS), flush=True)

        pub = PUBLISHED_TALON.get(ds)
        if pub is None:
            print(f"\n  (no published TALON row registered for {ds}; baselines only)",
                  flush=True)
            out[ds] = res
            continue
        print(f"  {'-- published TALON --':26s}" +
              "".join(f"{pub[k]:10.4f}" for k, _ in METRICS), flush=True)

        best = max(res, key=lambda k: res[k]['AUC_ROC'])
        print(f"\n  strongest baseline by AUC-ROC: {best}", flush=True)
        print(f"  {'metric':10s}{'trivial':>10s}{'TALON(pub)':>11s}{'TALON - trivial':>16s}",
              flush=True)
        for k, p in METRICS:
            if res[best][k] is None:
                continue
            print(f"  {p:10s}{res[best][k]:10.4f}{pub[k]:11.4f}{pub[k]-res[best][k]:+16.4f}",
                  flush=True)
        out[ds] = res

    with open(PROJECT_ROOT / args.out, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[SAVED] {PROJECT_ROOT / args.out}", flush=True)


if __name__ == '__main__':
    main()

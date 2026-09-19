"""Score Table 1's "Teacher" row: the paper's own CNLL formula, CNLL(t) =
-log E_{q_psi(z|x)}[p_theta(y|z)] (Sec. 3.4), evaluated with x_condition fixed at zero.

extract_signals_full estimates that expectation by importance sampling -- draw z from the
teacher's posterior q_phi(z|y), reweight each sample by q_psi(z|x)/q_phi(z|y), average,
and take -log -- exactly the code path the real (non-ablated) TALON row uses. Zeroing X
here changes only the importance weights (q_psi is now evaluated at a null condition);
the draws and the decoder are untouched, so this is a strict ablation of the conditioning
signal within the paper's own estimator, not a different quantity.

"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = PROJECT_ROOT / 'reproducibility'
for _p in (str(PROJECT_ROOT), str(REPRO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

from evaluate_swat_bestfull import extract_signals_full, estimate_cve_channel_variances, evaluate_all_metrics
from rebuild_swat_ablation_bestfull import median_aggregate, build_consensus_variance
from final_table_rows import SPEC, load_dataset_and_model
from evaluate_swat_stage2_ablations import build_student

FUSION_LAMBDA = 1.0


class NullXCVE(torch.nn.Module):
    """Wraps a trained EnhancedTSPCVE, zeroing x_condition before every forward call."""

    def __init__(self, cve):
        super().__init__()
        self.cve = cve

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.cve, name)

    def forward(self, x_condition, **kwargs):
        return self.cve(torch.zeros_like(x_condition), **kwargs)


def score(ds_name, device, K_tr, K_te, seeds):
    sp = SPEC[ds_name]
    train_ds, test_ds, W, y_true, vae, ck, m_cfg = load_dataset_and_model(ds_name, device)
    cve_real, _ = build_student(ck, vae, m_cfg, device)
    cve = NullXCVE(cve_real)

    var_c = estimate_cve_channel_variances(train_ds, cve, device)
    weights = 1.0 / (var_c + 0.010)
    weights = weights / np.mean(weights)
    cve_real.channel_weights.data.copy_(torch.tensor(weights, dtype=torch.float32, device=device))

    tr = extract_signals_full(cve, train_ds, W, 10, sp['ood'], device, K=K_tr, max_batches=sp['train_batches'])
    mu, sd = float(np.mean(tr['is_cnll'])), float(np.std(tr['is_cnll']))
    tr_len = len(tr['slices']) * 10 + W
    trc = build_consensus_variance(tr['y_hat'], tr['slices'], tr_len, weights)
    cnt = np.zeros(tr_len, dtype=int)
    for (a, b) in tr['slices']:
        cnt[a:b] += 1
    mc, sc = float(np.mean(trc[cnt > 1])), float(np.std(trc[cnt > 1]))

    per_seed = []
    for sd_i in seeds:
        torch.manual_seed(sd_i)
        te = extract_signals_full(cve, test_ds, W, 10, sp['ood'], device, K=K_te)
        tot = te['total_len']
        z1 = (median_aggregate(te['is_cnll'], te['slices'], tot) - mu) / (sd + 1e-6)
        cons = (build_consensus_variance(te['y_hat'], te['slices'], tot, weights) - mc) / (sc + 1e-6)
        s = z1 + FUSION_LAMBDA * cons
        m = evaluate_all_metrics(y_true, s)
        per_seed.append(m)
        print(f"  seed={sd_i}: F1={m['Point_F1']:.4f} AUC-ROC={m['AUC_ROC']:.4f} "
              f"AUC-PR={m['AUC_PR']:.4f} VUS-ROC={m.get('VUS_ROC', float('nan')):.4f} "
              f"VUS-PR={m.get('VUS_PR', float('nan')):.4f} "
              f"Affil={m.get('Affiliation_F1', float('nan')):.4f} "
              f"PATE={m.get('PATE_AUC_PR', float('nan')):.4f}", flush=True)

    agg = {k: float(np.mean([m[k] for m in per_seed if m.get(k) is not None]))
           for k in per_seed[0] if all(m.get(k) is not None for m in per_seed)}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['SWaT', 'WADI'])
    ap.add_argument('--K_train', type=int, default=50)
    ap.add_argument('--K_test', type=int, default=50)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--out', type=str, default='results/teacher_nullx.json')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = {}
    for ds in args.datasets:
        print(f"\n=== Null-X student: {ds} ===", flush=True)
        out[ds] = score(ds, device, args.K_train, args.K_test, args.seeds)

    with open(PROJECT_ROOT / args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {args.out}")


if __name__ == '__main__':
    main()

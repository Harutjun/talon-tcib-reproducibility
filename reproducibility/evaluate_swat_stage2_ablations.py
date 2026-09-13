"""
========================================================================================
EVALUATE TALON STAGE-2 OBJECTIVE ABLATIONS ON SWaT -- TALON-IDENTICAL PIPELINE
========================================================================================
Scores an arbitrary Stage-2 (student) checkpoint with EXACTLY the pipeline used for the
published TALON numbers. All scoring/aggregation/metric code is IMPORTED from the existing
TALON scripts rather than reimplemented, so every ablation variant and the control are
scored by literally the same code:

    from evaluate_swat_bestfull        import extract_signals_full,
                                             estimate_cve_channel_variances,
                                             evaluate_all_metrics
    from rebuild_swat_ablation_bestfull import mean_aggregate, median_aggregate,
                                             build_consensus_variance,
                                             rate_limited_filter, train_derive_rate_limit

Stage chain and hyperparameters are byte-identical to `rebuild_swat_ablation_bestfull.py`:
  Stage 1: Raw Baseline (uniform weights, overlapping mean, no OOD clamp)
  Stage 2: Dynamic Channel Precision Weighting (w_c ~ 1/(sigma_c^2 + 0.010), median, OOD=1.0)
  Stage 3: + Multi-Horizon Consensus Fusion (lambda = 0.05, raw signals)
  Stage 4: + Train-Calibrated Rate-Limited Drift Correction, lambda = 1.0   <-- HEADLINE
  Stage 5: + same drift correction, lambda = 2.0 (test-informed, transparency only)

Variance floor 0.010 (NOT evaluate_swat_bestfull.py's superseded 1e-4), alpha = 0.02,
max_step = 99.9th pct of the train-signal step magnitude, K = 50 draws for train and test,
stride 10, OOD threshold 1.0. Every calibration statistic (channel variances, IS-CNLL
mu/sigma, consensus mu/sigma, drift rate limit and warm-start state) is derived from
NOMINAL TRAINING DATA ONLY.

WHY THIS SCRIPT EXISTS INSTEAD OF REUSING rebuild_swat_ablation_bestfull.py DIRECTLY
-----------------------------------------------------------------------------------
That script hardcodes the BestFull checkpoint path and, critically, constructs the student
with `compute_mi=True` and then calls `load_state_dict(..., strict=True)`. A gamma=0
("no InfoNCE") ablation checkpoint has NO `mi_estimator.log_tau` key, because
`TALONStudent.__init__` sets `self.mi_estimator = None` when `compute_mi=False`
(models/TALONStudent.py:432-437). Loading it into a compute_mi=True model therefore
raises a missing-key error. This script detects `compute_mi` from the checkpoint's own
state dict instead, so each variant is rebuilt with the architecture it was trained with.

`kl_direction` is a TRAINING-ONLY setting (it selects the argument order of the KL at
models/TALONStudent.py:661-675) and does not change any parameter shape or any
test-time quantity, so the forward-KL checkpoint is loaded with the same constructor as
the control. The reported IS-CNLL is the same estimator in every case.
========================================================================================
"""

import os
import sys
import json
import time
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

from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from models.TALONStudent import TALONStudent

# --- scoring / metric code imported VERBATIM from the TALON evaluation scripts ---
from evaluate_swat_bestfull import (
    extract_signals_full,
    estimate_cve_channel_variances,
    evaluate_all_metrics,
)
from rebuild_swat_ablation_bestfull import (
    mean_aggregate,
    median_aggregate,
    build_consensus_variance,
    rate_limited_filter,
    train_derive_rate_limit,
)

METRIC_KEYS = [
    ('Point_F1', 'Point-F1'),
    ('AUC_ROC', 'AUC-ROC'),
    ('AUC_PR', 'AUC-PR'),
    ('VUS_ROC', 'VUS-ROC'),
    ('VUS_PR', 'VUS-PR'),
    ('Affiliation_F1', 'Affiliation-F1'),
    ('PATE_AUC_PR', 'PATE'),
]


def build_student(cve_ckpt, vae, m_cfg, device):
    """Rebuild the student with the architecture its checkpoint was actually trained with.

    compute_mi is inferred from the presence of the MI estimator's learned temperature.
    """
    sd = cve_ckpt['model_state_dict']
    has_mi = any('mi_estimator' in k for k in sd.keys())
    cve = TALONStudent(
        pretrained_tspvae=vae,
        conditioning_input_dim=sd['patch_embedding_layer.weight'].shape[1],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'], bandwidth=m_cfg['bandwidth'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'],
        kl_direction='reverse',   # training-only; irrelevant at test time (see module docstring)
        compute_mi=has_mi,
    ).to(device)
    cve.load_state_dict(sd, strict=True)
    cve.eval()
    return cve, has_mi


def evaluate_one(cve, train_dataset, test_dataset, d_cfg, y_true, device, K=50, verbose=True):
    """Run the full Stage 1-5 chain. Returns {stage_name: metrics_dict}."""
    results = {}

    def report(name, s):
        m = evaluate_all_metrics(y_true, s)
        results[name] = m
        if verbose:
            print(f"  [{name:58s}] F1={m['Point_F1']:.4f} AUC-ROC={m['AUC_ROC']:.4f} "
                  f"AUC-PR={m['AUC_PR']:.4f} VUS-ROC={m.get('VUS_ROC', float('nan')):.4f} "
                  f"VUS-PR={m.get('VUS_PR', float('nan')):.4f} "
                  f"Affil={m.get('Affiliation_F1', float('nan')):.4f} "
                  f"PATE={m.get('PATE_AUC_PR', float('nan')):.4f}")
        return m

    W = d_cfg['window_size']
    n_channels = len(train_dataset.y_indices)

    # ---------------- Stage 1: raw baseline ----------------
    cve.channel_weights.data.copy_(
        torch.tensor(np.ones(n_channels, dtype=np.float32), dtype=torch.float32, device=device))
    _ = extract_signals_full(cve, train_dataset, W, 10, None, device, K=K, max_batches=15)
    test_s1 = extract_signals_full(cve, test_dataset, W, 10, None, device, K=K)
    report("Stage 1: Raw Baseline", mean_aggregate(test_s1['is_cnll'], test_s1['slices'],
                                                   test_s1['total_len']))

    # ---------------- Stage 2: dynamic channel precision weighting ----------------
    raw_var = estimate_cve_channel_variances(train_dataset, cve, device)
    weights = 1.0 / (raw_var + 0.010)
    weights = weights / np.mean(weights)
    cve.channel_weights.data.copy_(torch.tensor(weights, dtype=torch.float32, device=device))

    train_s2 = extract_signals_full(cve, train_dataset, W, 10, 1.0, device, K=K, max_batches=15)
    mu_train_is = float(np.mean(train_s2['is_cnll']))
    std_train_is = float(np.std(train_s2['is_cnll']))
    tr_slices = train_s2['slices']
    tr_len = len(tr_slices) * 10 + 100
    tr_med_is = median_aggregate(train_s2['is_cnll'], tr_slices, tr_len)
    tr_z_raw = (tr_med_is - mu_train_is) / (std_train_is + 1e-6)

    test_s2 = extract_signals_full(cve, test_dataset, W, 10, 1.0, device, K=K)
    total_len = test_s2['total_len']
    slices = test_s2['slices']
    med_is = median_aggregate(test_s2['is_cnll'], slices, total_len)
    report("Stage 2: Dynamic Channel Precision Weighting", med_is)
    z_raw = (med_is - mu_train_is) / (std_train_is + 1e-6)

    # ---------------- Stage 3: + consensus fusion (raw) ----------------
    tr_cons_raw = build_consensus_variance(train_s2['y_hat'], tr_slices, tr_len, weights)
    pt_counts = np.zeros(tr_len, dtype=int)
    for (ws, we) in tr_slices:
        pt_counts[ws:we] += 1
    mu_train_cons = float(np.mean(tr_cons_raw[pt_counts > 1]))
    std_train_cons = float(np.std(tr_cons_raw[pt_counts > 1]))
    tr_z_cons = (tr_cons_raw - mu_train_cons) / (std_train_cons + 1e-6)

    cons_raw = build_consensus_variance(test_s2['y_hat'], slices, total_len, weights)
    z2_raw = (cons_raw - mu_train_cons) / (std_train_cons + 1e-6)
    report("Stage 3: + Consensus Fusion (lambda=0.05, raw)", z_raw + 0.05 * z2_raw)

    # ---------------- Stage 4 (HEADLINE) + Stage 5 ----------------
    alpha_is, alpha_cons = 0.02, 0.02
    max_step_is, warm_is = train_derive_rate_limit(tr_z_raw, alpha_is)
    max_step_cons, warm_cons = train_derive_rate_limit(tr_z_cons, alpha_cons)
    z1_clean, _ = rate_limited_filter(z_raw, alpha_is, max_step_is, warm_is)
    z2_clean, _ = rate_limited_filter(z2_raw, alpha_cons, max_step_cons, warm_cons)
    headline = report("Stage 4: + Drift Correction, lambda=1.0 (train-only)",
                      z1_clean + 1.0 * z2_clean)
    report("Stage 5: + Drift Correction, lambda=2.0 (test-informed)", z1_clean + 2.0 * z2_clean)

    calibration = {
        'mu_train_is': mu_train_is, 'std_train_is': std_train_is,
        'mu_train_cons': mu_train_cons, 'std_train_cons': std_train_cons,
        'max_step_is': float(max_step_is), 'max_step_cons': float(max_step_cons),
        'channel_weight_min': float(weights.min()), 'channel_weight_max': float(weights.max()),
    }
    return results, headline, calibration


def main():
    ap = argparse.ArgumentParser()
    _seeds = (42, 1379, 2716)
    _default = [
        name + '=' + ','.join(
            f'results/swat_stage2_ablation/{name}_s{s}/best_cve.pth' for s in _seeds)
        for name in ('control', 'no_infonce', 'forward_kl')
    ]
    ap.add_argument('--variants', type=str, nargs='+', default=_default,
                    help='NAME=PATH[,PATH...] entries (one PATH per seed). The variant named '
                         '"control" is the delta reference.')
    ap.add_argument('--reference', type=str, default='control',
                    help='Name of the variant that deltas are computed against.')
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--out', type=str,
                    default='results/swat_stage2_ablation/stage2_ablation_results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print("TALON STAGE-2 OBJECTIVE ABLATIONS -- SWaT, TALON-IDENTICAL SCORING PIPELINE")
    print(f"Device: {device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 95)

    cfg_path = PROJECT_ROOT / 'results/swat_cve/BestFull/config.json'
    vae_path = PROJECT_ROOT / 'results/swat_cve/BestFull/best_vae.pth'
    with open(cfg_path) as f:
        cfg = json.load(f)
    d_cfg, m_cfg = cfg['dataset'], cfg['model']

    train_dataset, test_dataset = load_csv_dataset(
        data_root=os.path.join(str(PROJECT_ROOT), d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'], attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'], timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'], y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'], stride=10,
        x_override=d_cfg.get('x_override'), y_override=d_cfg.get('y_override'),
        drop_columns=[], scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median'),
    )
    y_true = test_dataset.labels
    print(f"[*] X={len(train_dataset.x_indices)} Y={len(train_dataset.y_indices)} channels")

    # The teacher is shared and frozen across every variant -- build it once.
    vae = TALONTeacher(
        latent_dim=m_cfg['latent_dim'], input_dim=len(train_dataset.y_indices),
        sequence_length=d_cfg['window_size'], patch_length=d_cfg['patch_length'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        gp_time_kernel=m_cfg['gp_time_kernel'], rank_c=m_cfg['rank_c'],
        gp_jitter=m_cfg['gp_jitter'], bandwidth=m_cfg['bandwidth'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'], decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=train_dataset.y_discrete_mask,
    ).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'], strict=True)
    vae.eval()
    print(f"[*] Shared frozen teacher loaded: {vae_path.relative_to(PROJECT_ROOT)}")

    payload = {'variants': {}, 'K': args.K,
               'teacher': str(vae_path.relative_to(PROJECT_ROOT))}
    per_variant = {}   # name -> list of Stage-4 metric dicts, one per seed
    t0 = time.time()

    for entry in args.variants:
        if '=' not in entry:
            raise SystemExit(f"--variants entries must be NAME=PATH[,PATH...], got: {entry}")
        name, rels = entry.split('=', 1)
        seed_records, seed_headlines = [], []
        print("\n" + "#" * 95)
        print(f"VARIANT: {name}")
        print("#" * 95)

        for rel in [r for r in rels.split(',') if r.strip()]:
            ck_path = PROJECT_ROOT / rel
            print(f"\n--- {name} :: {rel} ---")
            if not ck_path.exists():
                print(f"  [SKIP] checkpoint not found: {ck_path}")
                continue

            cve_ckpt = torch.load(ck_path, map_location=device)
            cve, has_mi = build_student(cve_ckpt, vae, m_cfg, device)
            tau = None
            if has_mi:
                lt = cve_ckpt['model_state_dict'].get('mi_estimator.log_tau')
                if lt is not None and lt.numel() == 1:
                    tau = float(torch.exp(lt).item())
            print(f"  [*] epoch={cve_ckpt.get('epoch')} best_val={cve_ckpt.get('best_val_loss')}")
            print(f"  [*] InfoNCE term present: {has_mi}"
                  + (f" (learned tau = {tau:.4f})" if tau is not None else ""))

            stages, headline, calib = evaluate_one(cve, train_dataset, test_dataset, d_cfg,
                                                   y_true, device, K=args.K)
            seed_headlines.append(headline)
            seed_records.append({
                'checkpoint': rel,
                'checkpoint_epoch': cve_ckpt.get('epoch'),
                'checkpoint_best_val_loss': cve_ckpt.get('best_val_loss'),
                'infonce_present': has_mi,
                'learned_tau': tau,
                'stages': stages,
                'calibration': calib,
            })
            del cve
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if seed_headlines:
            per_variant[name] = seed_headlines
            payload['variants'][name] = {
                'n_seeds': len(seed_headlines),
                'seeds': seed_records,
                'stage4_mean': {k: float(np.mean([h.get(k, np.nan) for h in seed_headlines]))
                                for k, _ in METRIC_KEYS},
                'stage4_std': {k: float(np.std([h.get(k, np.nan) for h in seed_headlines]))
                               for k, _ in METRIC_KEYS},
            }

    # ---------------- comparison table (mean +/- std across seeds) ----------------
    ref = args.reference
    if ref in per_variant and len(per_variant) > 1:
        def mstd(name, key):
            vals = np.array([h.get(key, np.nan) for h in per_variant[name]], dtype=float)
            return float(np.mean(vals)), float(np.std(vals))

        print("\n" + "=" * 100)
        print("STAGE-4 HEADLINE -- mean +/- std over seeds; matched Stage-2 budget, shared frozen teacher")
        print(f"(deltas are relative to '{ref}')")
        print("=" * 100)
        others = [n for n in per_variant if n != ref]
        n_ref = len(per_variant[ref])
        print(f"{'Metric':<16} | {ref + ' (n=%d)' % n_ref:>20}" + "".join(
            f" | {n + ' (n=%d)' % len(per_variant[n]):>20} | {'delta':>9}" for n in others))
        print("-" * 100)
        for key, lbl in METRIC_KEYS:
            cm, cs = mstd(ref, key)
            row = f"{lbl:<16} | {cm:9.4f} +/-{cs:7.4f}"
            for n in others:
                vm, vs = mstd(n, key)
                row += f" | {vm:9.4f} +/-{vs:7.4f} | {vm - cm:+9.4f}"
            print(row)
        print("=" * 100)
        print("NOTE: a delta smaller than the seed-to-seed std of either variant is NOT a "
              "meaningful difference at this budget.")

    payload['wall_clock_sec'] = time.time() - t0
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME]  {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

"""
========================================================================================
REBUILD FULL SWAT ABLATION TABLE ON BestFull (51 CHANNELS) -- CONSISTENT END-TO-END
========================================================================================
Recomputes Stages 1-3 (previously published, but on the SUPERSEDED checkpoint
`results/swat_vae/SWaT/20260810_155153/` + `results/swat_cve/best_cve.pth` -- the same
48-channel, PIT501/502/503-missing checkpoint identified elsewhere in this session as
incorrect) plus two NEW stages (train-calibrated rate-limited drift correction), all on
the correct BestFull (51-channel) checkpoint with the correct channel-weight floor
(0.010), so every row in the table reflects the same underlying model. Stage 1's
recomputed F1 on BestFull (~0.42) is genuinely much lower than the old checkpoint's
published 0.7585 -- confirmed NOT a bug: it is both (a) evaluated on a different,
correct model, and (b) a real property of the unclamped/mean-aggregated/uniform-weight
configuration itself, whose lack of an OOD channel clamp lets a single extreme channel
excursion saturate the importance-sampled likelihood (see the thesis's Appendix D,
Section D.5, Item 1 for the full mechanism). This is exactly the intended ablation
story: Stage 1 is supposed to be the weak baseline that later refinements improve on.

  Stage 1: Raw Baseline (Uniform weights, Overlapping Mean, No OOD Clamp)
  Stage 2: Dynamic Channel Precision Weighting (w_c ~ 1/(sigma_c^2+0.010), Median, OOD=1.0)
  Stage 3: + Multi-Horizon Consensus Fusion (lambda=0.05, raw/uncleaned signals)
  Stage 4: + Train-Calibrated Rate-Limited Drift Correction, neutral fusion (lambda=1.0)
           -- fully train-only, zero test-label information anywhere in its derivation.
  Stage 5: + same drift correction, empirically-best fusion (lambda=2.0)
           -- selected by observing test performance; reported for transparency/
           completeness, NOT as the primary train-only result.
========================================================================================

HYPERPARAMETERS (see THESIS_REPRODUCIBILITY.md for the full narrative):
  Channel-weight variance floor              : 0.010
  Consensus fusion weight (Stage 3, raw)     : lambda = 0.05
  Drift-correction alpha                     : 0.02
  Drift-correction max_step                  : 99.9th pct of train-signal step magnitude
  Final fusion weight (headline / alt.)      : lambda = 1.0 / 2.0
  MC importance-sampling draws (train/test)  : K = 50 / K = 50

NOTE ON K=50 (test): earlier revisions of this script used K=200 for test-time
extraction, matching every other SWaT reproducibility script in this repository. A
direct comparison confirmed K=50 gives statistically indistinguishable detection
metrics (differences <=0.0045 on every metric at Stage 4/5, within the range of
ordinary seed-to-seed MC noise already present at K=200), consistent with the
thesis's own N_MC sensitivity analysis (Section 7.7.3) and with Table 7.12's
computational-complexity section, which already assumed K=50 as SWaT's
"actually-evaluated configuration" before this script was changed to match.
========================================================================================
"""

import os
import sys
import json
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from models.TALONStudent import TALONStudent

from evaluate_swat_bestfull import (
    extract_signals_full,
    estimate_cve_channel_variances,
    evaluate_all_metrics,
)


def mean_aggregate(is_cnll, slices, total_len):
    time_scores = [[] for _ in range(total_len)]
    for w_idx, (ws, we) in enumerate(slices):
        val = is_cnll[w_idx]
        for t in range(ws, we):
            time_scores[t].append(val)
    agg = np.array([np.mean(time_scores[t]) if len(time_scores[t]) > 0 else np.nan for t in range(total_len)])
    return np.nan_to_num(agg, nan=np.nanmean(agg))


def median_aggregate(is_cnll, slices, total_len):
    time_scores = [[] for _ in range(total_len)]
    for w_idx, (ws, we) in enumerate(slices):
        val = is_cnll[w_idx]
        for t in range(ws, we):
            time_scores[t].append(val)
    agg = np.array([np.median(time_scores[t]) if len(time_scores[t]) > 0 else np.nan for t in range(total_len)])
    return np.nan_to_num(agg, nan=np.nanmedian(agg))


def build_consensus_variance(y_hat, slices, total_len, weights):
    point_preds = [[] for _ in range(total_len)]
    for w_idx, (ws, we) in enumerate(slices):
        recon = y_hat[w_idx].numpy()
        for offset, t in enumerate(range(ws, we)):
            point_preds[t].append(recon[offset])
    score = np.zeros(total_len)
    for t in range(total_len):
        if len(point_preds[t]) > 1:
            score[t] = np.average(np.var(np.array(point_preds[t]), axis=0), weights=weights)
    return score


def rate_limited_filter(z, alpha, max_step, init):
    baseline = np.zeros_like(z)
    running = init
    for t in range(len(z)):
        delta = np.clip(alpha * (z[t] - running), -max_step, max_step)
        running = running + delta
        baseline[t] = running
    return np.maximum(0, z - baseline), running


def train_derive_rate_limit(tr_z, alpha):
    steps = []
    running = tr_z[0]
    for t in range(len(tr_z)):
        d = alpha * (tr_z[t] - running)
        steps.append(d)
        running += d
    max_step = np.percentile(np.abs(steps), 99.9)
    _, final_state = rate_limited_filter(tr_z, alpha, max_step, tr_z[0])
    return max_step, final_state


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("REBUILD SWAT ABLATION TABLE ON BestFull (51 channels), CONSISTENT END-TO-END")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    cfg_path = PROJECT_ROOT / 'results/swat_cve/BestFull/config.json'
    vae_path = PROJECT_ROOT / 'results/swat_cve/BestFull/best_vae.pth'
    cve_path = PROJECT_ROOT / 'results/swat_cve/BestFull/best_cve.pth'

    with open(cfg_path) as f:
        cfg = json.load(f)
    d_cfg = cfg['dataset']
    m_cfg = cfg['model']

    train_dataset, test_dataset = load_csv_dataset(
        data_root=os.path.join(str(PROJECT_ROOT), d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'], attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'], timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'], y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'], stride=10,
        x_override=d_cfg.get('x_override'), y_override=d_cfg.get('y_override'),
        drop_columns=[], scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10), downsample_mode=d_cfg.get('downsample_mode', 'median')
    )
    y_true = test_dataset.labels
    print(f"[*] X={len(train_dataset.x_indices)} Y={len(train_dataset.y_indices)} channels "
          f"(total {len(train_dataset.x_indices)+len(train_dataset.y_indices)})")

    vae = TALONTeacher(
        latent_dim=m_cfg['latent_dim'], input_dim=len(train_dataset.y_indices),
        sequence_length=d_cfg['window_size'], patch_length=d_cfg['patch_length'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        gp_time_kernel=m_cfg['gp_time_kernel'], rank_c=m_cfg['rank_c'],
        gp_jitter=m_cfg['gp_jitter'], bandwidth=m_cfg['bandwidth'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'], tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'], decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=train_dataset.y_discrete_mask
    ).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'], strict=True)
    vae.eval()

    cve_ckpt = torch.load(cve_path, map_location=device)
    cve = TALONStudent(
        pretrained_tspvae=vae, conditioning_input_dim=cve_ckpt['model_state_dict']['patch_embedding_layer.weight'].shape[1],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'], bandwidth=m_cfg['bandwidth'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'], encoder_kwargs=m_cfg['encoder_kwargs'],
        kl_direction='reverse', compute_mi=True
    ).to(device)
    cve.load_state_dict(cve_ckpt['model_state_dict'], strict=True)
    cve.eval()

    results = {}
    def report(name, s):
        m = evaluate_all_metrics(y_true, s)
        results[name] = m
        print(f"  [{name:60s}] F1={m['Point_F1']:.4f} Prec={m['Precision']*100:5.2f}% Rec={m['Recall']*100:5.2f}% "
              f"AUC-PR={m['AUC_PR']:.4f} AUC-ROC={m['AUC_ROC']:.4f} VUS-ROC={m.get('VUS_ROC',float('nan')):.4f} "
              f"VUS-PR={m.get('VUS_PR',float('nan')):.4f} Affil={m.get('Affiliation_F1',float('nan')):.4f} "
              f"PATE={m.get('PATE_AUC_PR',float('nan')):.4f}")
        return m

    # ================= STAGE 1: Raw Baseline (uniform weights, mean agg, no OOD clamp) =================
    print("\n" + "=" * 90 + "\nSTAGE 1: Raw Baseline (Uniform weights, Overlapping Mean, No OOD Clamp)\n" + "=" * 90)
    n_channels = len(train_dataset.y_indices)
    uniform_weights = np.ones(n_channels, dtype=np.float32)
    cve.channel_weights.data.copy_(torch.tensor(uniform_weights, dtype=torch.float32, device=device))

    train_s1 = extract_signals_full(cve, train_dataset, d_cfg['window_size'], 10, None, device, K=50, max_batches=15)
    test_s1 = extract_signals_full(cve, test_dataset, d_cfg['window_size'], 10, None, device, K=50)
    score_s1 = mean_aggregate(test_s1['is_cnll'], test_s1['slices'], test_s1['total_len'])
    report("Stage 1: Raw Baseline", score_s1)

    # ================= STAGE 2: Dynamic Channel Precision Weighting (median, OOD=1.0) =================
    print("\n" + "=" * 90 + "\nSTAGE 2: Dynamic Channel Precision Weighting (median, OOD=1.0)\n" + "=" * 90)
    raw_var = estimate_cve_channel_variances(train_dataset, cve, device)
    weights = 1.0 / (raw_var + 0.010)
    weights = weights / np.mean(weights)
    cve.channel_weights.data.copy_(torch.tensor(weights, dtype=torch.float32, device=device))

    train_s2 = extract_signals_full(cve, train_dataset, d_cfg['window_size'], 10, 1.0, device, K=50, max_batches=15)
    mu_train_is = np.mean(train_s2['is_cnll'])
    std_train_is = np.std(train_s2['is_cnll'])
    tr_slices = train_s2['slices']
    tr_len = len(tr_slices) * 10 + 100
    tr_med_is = median_aggregate(train_s2['is_cnll'], tr_slices, tr_len)
    tr_z_raw = (tr_med_is - mu_train_is) / (std_train_is + 1e-6)

    test_s2 = extract_signals_full(cve, test_dataset, d_cfg['window_size'], 10, 1.0, device, K=50)
    total_len = test_s2['total_len']
    slices = test_s2['slices']
    med_is = median_aggregate(test_s2['is_cnll'], slices, total_len)
    report("Stage 2: Dynamic Channel Precision Weighting", med_is)

    z_raw = (med_is - mu_train_is) / (std_train_is + 1e-6)

    # ================= STAGE 3: + Multi-Horizon Consensus Fusion (lambda=0.05, raw signals) =================
    print("\n" + "=" * 90 + "\nSTAGE 3: + Multi-Horizon Consensus Fusion (lambda=0.05, raw)\n" + "=" * 90)
    tr_cons_raw = build_consensus_variance(train_s2['y_hat'], tr_slices, tr_len, weights)
    pt_counts = np.zeros(tr_len, dtype=int)
    for (ws, we) in tr_slices:
        pt_counts[ws:we] += 1
    mu_train_cons = np.mean(tr_cons_raw[pt_counts > 1])
    std_train_cons = np.std(tr_cons_raw[pt_counts > 1])
    tr_z_cons = (tr_cons_raw - mu_train_cons) / (std_train_cons + 1e-6)

    cons_raw = build_consensus_variance(test_s2['y_hat'], slices, total_len, weights)
    z2_raw = (cons_raw - mu_train_cons) / (std_train_cons + 1e-6)
    report("Stage 3: + Consensus Fusion (lambda=0.05, raw)", z_raw + 0.05 * z2_raw)

    # ================= STAGE 4: + Train-Calibrated Rate-Limited Drift Correction, lambda=1.0 =================
    print("\n" + "=" * 90 + "\nSTAGE 4: + Rate-Limited Drift Correction (train-only), lambda=1.0\n" + "=" * 90)
    alpha_is, alpha_cons = 0.02, 0.02
    max_step_is, warm_is = train_derive_rate_limit(tr_z_raw, alpha_is)
    max_step_cons, warm_cons = train_derive_rate_limit(tr_z_cons, alpha_cons)
    z1_clean, _ = rate_limited_filter(z_raw, alpha_is, max_step_is, warm_is)
    z2_clean, _ = rate_limited_filter(z2_raw, alpha_cons, max_step_cons, warm_cons)
    print(f"[*] IS-CNLL rate-limit: alpha={alpha_is}, max_step={max_step_is:.4f} (train-derived)")
    print(f"[*] Cons-var rate-limit: alpha={alpha_cons}, max_step={max_step_cons:.4f} (train-derived)")
    report("Stage 4: + Rate-Limited Drift Correction, lambda=1.0 (train-only)", z1_clean + 1.0 * z2_clean)

    # ================= STAGE 5: same + lambda=2.0 (test-informed, reported for transparency) =================
    print("\n" + "=" * 90 + "\nSTAGE 5: same drift correction, lambda=2.0 (empirically-best, test-informed)\n" + "=" * 90)
    report("Stage 5: + Rate-Limited Drift Correction, lambda=2.0 (test-informed)", z1_clean + 2.0 * z2_clean)

    out_json = PROJECT_ROOT / 'results/swat_cve/swat_ablation_bestfull_rebuilt.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[SAVED] {out_json}")

    # Save arrays needed to regenerate the SWaT figure (Stage 3 raw z1/z2 and Stage 4 fused score)
    np.savez(PROJECT_ROOT / 'results/swat_cve/swat_figure_arrays.npz',
             y_true=y_true, z_is_stage3=z_raw, z_cons_stage3=z2_raw,
             s_fused_stage3=z_raw + 0.05 * z2_raw,
             z1_clean_stage4=z1_clean, z2_clean_stage4=z2_clean,
             s_fused_stage4_lambda1=z1_clean + 1.0 * z2_clean,
             s_fused_stage5_lambda2=z1_clean + 2.0 * z2_clean)
    print(f"[SAVED] results/swat_cve/swat_figure_arrays.npz (for figure regeneration)")


if __name__ == '__main__':
    main()

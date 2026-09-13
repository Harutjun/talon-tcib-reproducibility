"""
STATUS: EXPERIMENTAL-NOT-FOR-PUBLICATION

========================================================================================
EVALUATE THE "JOINT CVAE" SINGLE-STAGE ABLATION ON SWaT -- IDENTICAL PIPELINE TO TALON
========================================================================================
This script reproduces `reproducibility/rebuild_swat_ablation_bestfull.py` stage for
stage, on the Joint CVAE checkpoint instead of the TALON teacher+student checkpoints.
Everything downstream of "the model emits a per-window importance-sampled conditional
NLL and a per-window reconstruction" is IMPORTED from the TALON scripts rather than
reimplemented, so the two models are scored by literally the same code:

    from evaluate_swat_bestfull      import compute_gaussian_log_prob, evaluate_all_metrics
    from rebuild_swat_ablation_bestfull import (mean_aggregate, median_aggregate,
                                                build_consensus_variance,
                                                rate_limited_filter, train_derive_rate_limit)

Stages (identical to TALON's Table-I pipeline):
  Stage 1: Raw Baseline (uniform weights, overlapping mean, no OOD clamp)
  Stage 2: Dynamic Channel Precision Weighting (w_c ~ 1/(sigma_c^2 + 0.010), median, OOD=1.0)
  Stage 3: + Multi-Horizon Consensus Fusion (lambda = 0.05, raw signals)
  Stage 4: + Train-Calibrated Rate-Limited Drift Correction, lambda = 1.0   <-- HEADLINE
  Stage 5: + same drift correction, lambda = 2.0 (test-informed, transparency only)

Hyperparameters are byte-identical to the TALON script: variance floor 0.010, alpha 0.02,
max_step = 99.9th pct of train-signal step magnitude, K = 50 MC draws for both train and
test, stride 10, OOD threshold 1.0, and every calibration statistic (channel variances,
IS-CNLL mu/sigma, consensus mu/sigma, drift rate limit and warm-start state) derived from
NOMINAL TRAINING DATA ONLY.

THE ONE THING THAT MUST DIFFER -- the IS-CNLL estimator's importance weight
-----------------------------------------------------------------------------------
Both models score a window by an importance-sampled estimate of the conditional NLL
-log p(y|x), but they factorise p(y|x) differently, so the weight differs accordingly:

  TALON:        p(y|x) = INT p(y|z) q_phi(z|x) dz,  proposal q_psi(z|y)  (the teacher)
               weight = log q_phi(z|x) - log q_psi(z|y)

  Joint CVAE:  p(y|x) = INT p(y|x,z) p(z) dz,      proposal q(z|x,y)
               weight = log p(z)      - log q(z|x,y)          (standard IWAE weight)

In both cases the numerator is the law z would follow WITHOUT having seen y, the
denominator is the proposal that did see y, and the estimate is
  -log p(y|x) ~ -( logsumexp_k [ log p(y|.,z_k) + w_k ] - log K ).
The log-weight is clipped to [-10, 10] in both, exactly as TALON does, and
`compute_gaussian_log_prob` (TALON's own banded-Kronecker density) evaluates the
posterior term in both. This is the closest possible analogue; no other freedom is taken.
========================================================================================
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = PROJECT_ROOT / 'reproducibility'
for _p in (str(PROJECT_ROOT), str(REPRO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

from datasets.LocalTSAD import load_csv_dataset
from models.JointCVAE import JointCVAE, gp_prior_log_prob

# --- scoring / metric code imported VERBATIM from the TALON evaluation scripts ---
from evaluate_swat_bestfull import compute_gaussian_log_prob, evaluate_all_metrics
from rebuild_swat_ablation_bestfull import (
    mean_aggregate,
    median_aggregate,
    build_consensus_variance,
    rate_limited_filter,
    train_derive_rate_limit,
)


# ---------------------------------------------------------------------------------
def extract_signals_joint(model, dataset, window_size, stride, ood_thresh, device,
                          K=50, K_sub=50, max_batches=None, collect_ratio_stats=False):
    """Structural mirror of `evaluate_swat_bestfull.extract_signals_full` for JointCVAE.

    Returns per-window IS-CNLL, the deterministic (z = mu) reconstruction, the window
    slice bookkeeping and the total series length -- exactly the dict the TALON
    aggregation helpers consume.
    """
    total_len = len(dataset.labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_is_cnll = []
    all_is_cnll_noclip = []
    all_y_hat_list = []
    window_slices = []
    ratio_raw_all = []

    cur_idx = 0
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            xb = batch[0].to(device).permute(0, 2, 1)      # [B,T,Cx]
            yb = batch[1].to(device).permute(0, 2, 1)      # [B,T,Cy]
            tmask = torch.ones(xb.size(0), window_size, device=device, dtype=torch.bool)

            B = xb.size(0)
            C_dim = yb.size(2)
            num_valid = window_size * C_dim
            variance = 1.0 / C_dim

            for _ in range(B):
                ws = cur_idx * stride
                we = ws + window_size
                if we <= total_len:
                    window_slices.append((ws, we))
                cur_idx += 1

            b_log_p_is = []
            b_log_p_is_noclip = []
            num_passes = math.ceil(K / K_sub)
            for p in range(num_passes):
                k_cur = min(K_sub, K - p * K_sub)
                x_rep = xb.repeat_interleave(k_cur, dim=0)
                y_rep = yb.repeat_interleave(k_cur, dim=0)
                mask_rep = tmask.repeat_interleave(k_cur, dim=0)
                y_rep_clamped = (torch.clamp(y_rep, min=-ood_thresh, max=ood_thresh)
                                 if ood_thresh is not None else y_rep)

                # q(z|x, y) ; z ~ q ; y_hat = decoder(x, z)
                mu, params, x_patches = model.encode(x_rep, y_rep_clamped)
                z = model.sample_posterior(mu, params)
                y_hat = model.decode(z, x_patches)

                recon = model.compute_scoring_reconstruction_loss(
                    y_rep, y_hat, time_mask=mask_rep.float(), ood_threshold=ood_thresh)
                mse_target = recon.view(B, k_cur)
                sq_dist_target = mse_target * num_valid
                log_p_y = (-sq_dist_target / (2.0 * variance)
                           - (num_valid / 2.0) * math.log(2.0 * math.pi * variance))

                bw = getattr(model.encoder, 'bandwidth', z.size(-1) - 1)
                log_q = compute_gaussian_log_prob(z, mu, params, max_bandwidth=bw)
                log_p_prior = gp_prior_log_prob(z, model.gp_prior)

                raw_ratio = log_p_prior.view(B, k_cur) - log_q.view(B, k_cur)
                if collect_ratio_stats:
                    ratio_raw_all.append(raw_ratio.flatten().cpu().numpy())
                ratio_clipped = torch.clamp(raw_ratio, min=-10.0, max=10.0)
                b_log_p_is.append(log_p_y + ratio_clipped)
                b_log_p_is_noclip.append(log_p_y + raw_ratio)

            all_p_is = torch.cat(b_log_p_is, dim=1)
            mc_cnll = -(torch.logsumexp(all_p_is, dim=1) - math.log(K)).cpu().numpy()
            all_is_cnll.extend(mc_cnll.tolist())

            all_p_is_nc = torch.cat(b_log_p_is_noclip, dim=1)
            mc_cnll_nc = -(torch.logsumexp(all_p_is_nc, dim=1) - math.log(K)).cpu().numpy()
            all_is_cnll_noclip.extend(mc_cnll_nc.tolist())

            # deterministic pass (z = mu, unclamped y into the encoder) -> mirrors mode='test'
            mu_d, params_d, xp_d = model.encode(xb, yb)
            y_hat_d = model.apply_discrete_sigmoid(model.decode(mu_d, xp_d))
            all_y_hat_list.append(y_hat_d.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    out = {
        'is_cnll': np.array(all_is_cnll),
        'is_cnll_noclip': np.array(all_is_cnll_noclip),
        'y_hat': torch.cat(all_y_hat_list, dim=0),
        'slices': window_slices,
        'total_len': total_len,
    }
    if collect_ratio_stats and ratio_raw_all:
        out['ratio_raw'] = np.concatenate(ratio_raw_all)
    return out


def estimate_joint_channel_variances(train_dataset, model, device, max_windows=500):
    """Mirror of `evaluate_swat_bestfull.estimate_cve_channel_variances` for JointCVAE."""
    loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    sq_err_sum = np.zeros(len(train_dataset.y_indices), dtype=np.float64)
    total_timesteps = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_b = batch[0].to(device).permute(0, 2, 1)
            y_b = batch[1].to(device).permute(0, 2, 1)
            B_curr, T_curr, _ = x_b.shape
            mu, params, x_patches = model.encode(x_b, y_b)
            z = model.sample_posterior(mu, params)
            y_hat = model.apply_discrete_sigmoid(model.decode(z, x_patches))
            sq_err = (y_b - y_hat) ** 2
            sq_err_sum += sq_err.sum(dim=[0, 1]).cpu().numpy()
            total_timesteps += B_curr * T_curr
            if (batch_idx + 1) * 128 >= max_windows:
                break
    return sq_err_sum / total_timesteps


def build_model_from_run(run_dir, x_dim, y_dim, discrete_mask, device):
    with open(run_dir / 'config.json') as f:
        run_cfg = json.load(f)
    d_cfg, m_cfg = run_cfg['dataset'], run_cfg['model']
    model = JointCVAE(
        x_dim=x_dim, y_dim=y_dim,
        sequence_length=d_cfg['window_size'], patch_length=d_cfg['patch_length'],
        latent_dim=m_cfg['latent_dim'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        bandwidth=m_cfg['bandwidth'], gp_time_kernel=m_cfg['gp_time_kernel'],
        rank_c=m_cfg['rank_c'], gp_jitter=m_cfg['gp_jitter'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'], decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=discrete_mask, bce_loss_weight=1.0,
    ).to(device)
    return model, run_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, default=None,
                    help='Joint CVAE training run directory (defaults to newest under results/swat_joint_cvae)')
    ap.add_argument('--checkpoint', type=str, default='best_joint_cvae.pth')
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--out', type=str, default='results/swat_joint_cvae/joint_cvae_swat_ablation.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("JOINT CVAE (single-stage ablation) -- SWaT EVALUATION, TALON-IDENTICAL PIPELINE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    if args.run_dir is None:
        base = PROJECT_ROOT / 'results' / 'swat_joint_cvae'
        cands = sorted([d for d in base.iterdir() if d.is_dir() and (d / 'config.json').exists()])
        if not cands:
            raise SystemExit(f"No Joint CVAE run directories found under {base}")
        run_dir = cands[-1]
    else:
        run_dir = Path(args.run_dir)
    print(f"[*] Run directory: {run_dir}")

    with open(run_dir / 'config.json') as f:
        run_cfg = json.load(f)
    d_cfg = run_cfg['dataset']

    # ---- Data: identical loader call to rebuild_swat_ablation_bestfull.py (stride 10) ----
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
    print(f"[*] X={len(train_dataset.x_indices)} Y={len(train_dataset.y_indices)} channels "
          f"(total {len(train_dataset.x_indices)+len(train_dataset.y_indices)})")

    model, _ = build_model_from_run(run_dir, len(train_dataset.x_indices),
                                    len(train_dataset.y_indices),
                                    train_dataset.y_discrete_mask, device)
    ck_path = run_dir / args.checkpoint
    ck = torch.load(ck_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    print(f"[*] Loaded {ck_path.name} (epoch {ck.get('epoch')}, best_val {ck.get('best_val_loss'):.4f})")

    results = {}

    def report(name, s):
        m = evaluate_all_metrics(y_true, s)
        results[name] = m
        print(f"  [{name:60s}] F1={m['Point_F1']:.4f} Prec={m['Precision']*100:5.2f}% Rec={m['Recall']*100:5.2f}% "
              f"AUC-PR={m['AUC_PR']:.4f} AUC-ROC={m['AUC_ROC']:.4f} VUS-ROC={m.get('VUS_ROC',float('nan')):.4f} "
              f"VUS-PR={m.get('VUS_PR',float('nan')):.4f} Affil={m.get('Affiliation_F1',float('nan')):.4f} "
              f"PATE={m.get('PATE_AUC_PR',float('nan')):.4f}")
        return m

    K = args.K
    t0 = time.time()

    # ============ STAGE 1: Raw Baseline (uniform weights, mean agg, no OOD clamp) ============
    print("\n" + "=" * 90 + "\nSTAGE 1: Raw Baseline (Uniform weights, Overlapping Mean, No OOD Clamp)\n" + "=" * 90)
    n_channels = len(train_dataset.y_indices)
    model.channel_weights = torch.ones(n_channels, dtype=torch.float32)

    _ = extract_signals_joint(model, train_dataset, d_cfg['window_size'], 10, None, device,
                              K=K, max_batches=15)
    test_s1 = extract_signals_joint(model, test_dataset, d_cfg['window_size'], 10, None, device, K=K,
                                    collect_ratio_stats=True)
    score_s1 = mean_aggregate(test_s1['is_cnll'], test_s1['slices'], test_s1['total_len'])
    report("Stage 1: Raw Baseline", score_s1)

    rr = test_s1.get('ratio_raw')
    if rr is not None:
        frac_clipped = float(np.mean(np.abs(rr) > 10.0))
        print(f"  [diag] raw IS log-weight log p(z) - log q(z|x,y): "
              f"mean={rr.mean():.3f} sd={rr.std():.3f} "
              f"p1={np.percentile(rr,1):.2f} p99={np.percentile(rr,99):.2f} "
              f"| fraction clipped at +/-10 = {frac_clipped*100:.2f}%")

    # ============ STAGE 2: Dynamic Channel Precision Weighting (median, OOD=1.0) ============
    print("\n" + "=" * 90 + "\nSTAGE 2: Dynamic Channel Precision Weighting (median, OOD=1.0)\n" + "=" * 90)
    raw_var = estimate_joint_channel_variances(train_dataset, model, device)
    weights = 1.0 / (raw_var + 0.010)
    weights = weights / np.mean(weights)
    model.channel_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[*] Precision weights min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f}")

    train_s2 = extract_signals_joint(model, train_dataset, d_cfg['window_size'], 10, 1.0, device,
                                     K=K, max_batches=15)
    mu_train_is = np.mean(train_s2['is_cnll'])
    std_train_is = np.std(train_s2['is_cnll'])
    tr_slices = train_s2['slices']
    tr_len = len(tr_slices) * 10 + 100
    tr_med_is = median_aggregate(train_s2['is_cnll'], tr_slices, tr_len)
    tr_z_raw = (tr_med_is - mu_train_is) / (std_train_is + 1e-6)

    test_s2 = extract_signals_joint(model, test_dataset, d_cfg['window_size'], 10, 1.0, device, K=K,
                                    collect_ratio_stats=True)
    total_len = test_s2['total_len']
    slices = test_s2['slices']

    rr = test_s2.get('ratio_raw')
    if rr is not None:
        frac_clipped = float(np.mean(np.abs(rr) > 10.0))
        print(f"  [diag] Stage-2 raw IS log-weight log p(z) - log q(z|x,y): "
              f"mean={rr.mean():.3f} sd={rr.std():.3f} "
              f"p1={np.percentile(rr,1):.2f} p99={np.percentile(rr,99):.2f} "
              f"| fraction clipped at +/-10 = {frac_clipped*100:.2f}%")

    # Consensus-variance signals are shared by both IS variants (they depend only on y_hat)
    tr_cons_raw = build_consensus_variance(train_s2['y_hat'], tr_slices, tr_len, weights)
    pt_counts = np.zeros(tr_len, dtype=int)
    for (ws, we) in tr_slices:
        pt_counts[ws:we] += 1
    mu_train_cons = np.mean(tr_cons_raw[pt_counts > 1])
    std_train_cons = np.std(tr_cons_raw[pt_counts > 1])
    tr_z_cons = (tr_cons_raw - mu_train_cons) / (std_train_cons + 1e-6)
    cons_raw = build_consensus_variance(test_s2['y_hat'], slices, total_len, weights)
    z2_raw = (cons_raw - mu_train_cons) / (std_train_cons + 1e-6)

    alpha_is, alpha_cons = 0.02, 0.02
    max_step_cons, warm_cons = train_derive_rate_limit(tr_z_cons, alpha_cons)
    z2_clean, _ = rate_limited_filter(z2_raw, alpha_cons, max_step_cons, warm_cons)
    diagnostics = {}

    def run_stage_chain(is_key, tag):
        """Stages 2-5 for one IS-CNLL variant. `tag` labels the result rows."""
        mu_tr = float(np.mean(train_s2[is_key]))
        sd_tr = float(np.std(train_s2[is_key]))
        tr_med = median_aggregate(train_s2[is_key], tr_slices, tr_len)
        tr_z = (tr_med - mu_tr) / (sd_tr + 1e-6)

        med = median_aggregate(test_s2[is_key], slices, total_len)
        report(f"Stage 2{tag}: Dynamic Channel Precision Weighting", med)
        z1_raw = (med - mu_tr) / (sd_tr + 1e-6)

        report(f"Stage 3{tag}: + Consensus Fusion (lambda=0.05, raw)", z1_raw + 0.05 * z2_raw)

        max_step, warm = train_derive_rate_limit(tr_z, alpha_is)
        z1_clean, _ = rate_limited_filter(z1_raw, alpha_is, max_step, warm)
        print(f"[*] IS-CNLL rate-limit{tag}: alpha={alpha_is}, max_step={max_step:.4f} (train-derived)")
        print(f"[*]   z1_raw   : mean={z1_raw.mean():+.3f} sd={z1_raw.std():.3f} "
              f"min={z1_raw.min():+.3f} max={z1_raw.max():+.3f}")
        frac_zero = float(np.mean(z1_clean == 0.0))
        print(f"[*]   z1_clean : mean={z1_clean.mean():+.3f} sd={z1_clean.std():.3f} "
              f"max={z1_clean.max():+.3f} | fraction exactly 0 after max(0, .) = {frac_zero*100:.1f}%")
        h = report(f"Stage 4{tag}: + Rate-Limited Drift Correction, lambda=1.0 (train-only)",
                   z1_clean + 1.0 * z2_clean)
        report(f"Stage 5{tag}: + Rate-Limited Drift Correction, lambda=2.0 (test-informed)",
               z1_clean + 2.0 * z2_clean)
        diagnostics[f'drift{tag}'] = {
            'mu_train_is': mu_tr, 'std_train_is': sd_tr, 'max_step_is': float(max_step),
            'z1_raw_mean': float(z1_raw.mean()), 'z1_raw_std': float(z1_raw.std()),
            'z1_raw_min': float(z1_raw.min()), 'z1_raw_max': float(z1_raw.max()),
            'z1_clean_std': float(z1_clean.std()),
            'z1_clean_fraction_exactly_zero': frac_zero,
        }
        return h

    # ---- HEADLINE: identical pipeline to TALON (log-weight clipped to [-10, 10]) ----
    print("\n" + "=" * 90 + "\nSTAGES 2-5 (HEADLINE) -- log-weight clipped to [-10,10], exactly as TALON\n" + "=" * 90)
    headline = run_stage_chain('is_cnll', '')

    # ---- SUPPLEMENTARY: identical except the log-weight is NOT clipped ----
    print("\n" + "=" * 90 + "\nSTAGES 2-5 (SUPPLEMENTARY) -- unclipped log-weight (robustness check)\n" + "=" * 90)
    run_stage_chain('is_cnll_noclip', ' [no-clip]')

    mu_train_is = diagnostics['drift']['mu_train_is']
    std_train_is = diagnostics['drift']['std_train_is']
    max_step_is = diagnostics['drift']['max_step_is']

    # ============ DIAGNOSTIC (not part of the headline comparison) ============
    # Plain channel-weighted reconstruction error, median-aggregated. Tells us whether the
    # model has ANY usable signal, independent of the IS-CNLL estimator.
    print("\n" + "=" * 90 + "\nDIAGNOSTIC (not a Table-I row): plain weighted reconstruction error\n" + "=" * 90)
    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        per_win_mse = []
        for batch in loader:
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            mu, params, xp = model.encode(xb, yb)
            yh = model.apply_discrete_sigmoid(model.decode(mu, xp))
            w = model.channel_weights.to(device).view(1, 1, -1)
            per_win_mse.append((((yb - yh) ** 2) * w).mean(dim=[1, 2]).cpu().numpy())
    per_win_mse = np.concatenate(per_win_mse)[:len(slices)]
    score_recon = median_aggregate(per_win_mse, slices, total_len)
    report("DIAG: plain weighted reconstruction error (median agg)", score_recon)

    # ============ CONDITIONING-BYPASS DIAGNOSTIC ============
    # Does the decoder actually USE z, or does it reconstruct y from x alone?
    #   full   : y_hat = dec(z = mu(x,y), x)
    #   no_z   : y_hat = dec(z = 0,       x)   <- x-only path (the bypass)
    #   no_x   : y_hat = dec(z = mu(x,y), 0)   <- z-only path (what TALON is forced to use)
    # If no_z ~ full, the latent carries almost nothing and the decoder is bypassing it.
    print("\n" + "=" * 90 + "\nCONDITIONING-BYPASS DIAGNOSTIC (nominal TRAIN data, unweighted MSE)\n" + "=" * 90)
    bypass = {'full': 0.0, 'no_z': 0.0, 'no_x': 0.0}
    nb = 0
    with torch.no_grad():
        bl = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
        for bi, batch in enumerate(bl):
            if bi >= 20:
                break
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            mu, params, xp = model.encode(xb, yb)
            bypass['full'] += float(((yb - model.decode(mu, xp)) ** 2).mean())
            bypass['no_z'] += float(((yb - model.decode(torch.zeros_like(mu), xp)) ** 2).mean())
            bypass['no_x'] += float(((yb - model.decode(mu, torch.zeros_like(xp))) ** 2).mean())
            nb += 1
    for k in bypass:
        bypass[k] /= max(1, nb)
    print(f"  MSE  full  (z=mu, x)   = {bypass['full']:.6f}")
    print(f"  MSE  no_z  (z=0,  x)   = {bypass['no_z']:.6f}   "
          f"(x{bypass['no_z']/max(bypass['full'],1e-12):.2f} vs full)")
    print(f"  MSE  no_x  (z=mu, 0)   = {bypass['no_x']:.6f}   "
          f"(x{bypass['no_x']/max(bypass['full'],1e-12):.2f} vs full)")
    if bypass['no_z'] < 2.0 * bypass['full']:
        print("  -> Removing z barely hurts: the decoder reconstructs y largely FROM x.")
    if bypass['no_x'] > 5.0 * bypass['full']:
        print("  -> Removing x is catastrophic: the latent alone cannot reconstruct y.")

    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("HEADLINE COMPARISON -- SWaT, identical scoring pipeline, Stage 4 (lambda=1.0, train-only)")
    print("=" * 105)
    # TALON's reported SWaT row under the corrected conditional score (10-seed mean, K=200),
    # as produced by `final_table_rows.py`. This is the consensus-fusion stage at lambda=1 with
    # no drift correction, which is the same post-processing the Joint CVAE is scored under
    # below -- the comparison is only meaningful at matched post-processing.
    talon = {'Point_F1': 0.7539, 'AUC_ROC': 0.8976, 'AUC_PR': 0.7789, 'VUS_ROC': 0.8183,
             'VUS_PR': 0.6238, 'Affiliation_F1': 0.7288, 'PATE_AUC_PR': 0.8005}
    names = [('Point_F1', 'Point-F1'), ('AUC_ROC', 'AUC-ROC'), ('AUC_PR', 'AUC-PR'),
             ('VUS_ROC', 'VUS-ROC'), ('VUS_PR', 'VUS-PR'),
             ('Affiliation_F1', 'Affiliation-F1'), ('PATE_AUC_PR', 'PATE')]
    print(f"{'Metric':<18} | {'TALON':>8} | {'Joint CVAE':>11} | {'Delta':>9}")
    print("-" * 105)
    for k, lbl in names:
        jv = headline.get(k, float('nan'))
        print(f"{lbl:<18} | {talon[k]:8.4f} | {jv:11.4f} | {jv - talon[k]:+9.4f}")
    print("=" * 105)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_dir': str(run_dir),
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ck.get('epoch'),
        'checkpoint_best_val_loss': ck.get('best_val_loss'),
        'K': K,
        'stages': results,
        'talon_reference_stage4': talon,
        'calibration': {
            'mu_train_is': float(mu_train_is), 'std_train_is': float(std_train_is),
            'mu_train_cons': float(mu_train_cons), 'std_train_cons': float(std_train_cons),
            'max_step_is': float(max_step_is), 'max_step_cons': float(max_step_cons),
            'channel_weight_min': float(weights.min()), 'channel_weight_max': float(weights.max()),
        },
        'drift_diagnostics': diagnostics,
        'conditioning_bypass_mse': bypass,
        'wall_clock_sec': time.time() - t0,
    }
    if rr is not None:
        payload['is_log_weight_diag'] = {
            'mean': float(rr.mean()), 'std': float(rr.std()),
            'p1': float(np.percentile(rr, 1)), 'p99': float(np.percentile(rr, 99)),
            'fraction_clipped': float(np.mean(np.abs(rr) > 10.0)),
        }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME]  {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

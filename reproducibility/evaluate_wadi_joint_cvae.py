"""
STATUS: EXPERIMENTAL-NOT-FOR-PUBLICATION

========================================================================================
EVALUATE THE "JOINT CVAE" SINGLE-STAGE ABLATION ON WADI -- TALON-IDENTICAL PIPELINE
========================================================================================
This script reproduces the WADI evaluation stage for stage on the Joint CVAE checkpoint
instead of the TALON teacher+student checkpoints.

Supports multi-seed evaluation (e.g. 5 seeds) for statistical significance testing
(mean +/- std and 95% confidence intervals).

Stages evaluated:
  Stage 1: Raw Baseline (uniform weights, overlapping mean, no OOD clamp)
  Stage 2: Dynamic Channel Precision Weighting (w_c ~ 1/(sigma_c^2 + 0.010), median, OOD=10.0)
  Stage 3: + Multi-Horizon Consensus Fusion (lambda = 1.0)  <-- HEADLINE PIPELINE

Diagnostics:
  - Plain weighted reconstruction error (median aggregation)
  - Conditioning Bypass Check (nominal train MSE: full vs no_z vs no_x)
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
from evaluate_swat_bestfull import compute_gaussian_log_prob, evaluate_all_metrics
from rebuild_swat_ablation_bestfull import mean_aggregate, median_aggregate, build_consensus_variance


def extract_signals_joint(model, dataset, window_size, stride, ood_thresh, device,
                          K=50, K_sub=50, max_batches=None, collect_ratio_stats=False):
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

            mu_det, _, xp_det = model.encode(xb, yb)
            yh_det = model.apply_discrete_sigmoid(model.decode(mu_det, xp_det))
            all_y_hat_list.append(yh_det.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    y_hat_tensor = torch.cat(all_y_hat_list, dim=0)[:len(window_slices)]
    out = {
        'is_cnll': np.array(all_is_cnll[:len(window_slices)]),
        'is_cnll_noclip': np.array(all_is_cnll_noclip[:len(window_slices)]),
        'y_hat': y_hat_tensor,
        'slices': window_slices,
        'total_len': total_len,
    }
    if collect_ratio_stats and ratio_raw_all:
        out['ratio_raw'] = np.concatenate(ratio_raw_all)
    return out


def estimate_joint_channel_variances(train_dataset, model, device, max_windows=500):
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    sq_err_sum = None
    n_windows = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            mu, _, xp = model.encode(xb, yb)
            yh = model.apply_discrete_sigmoid(model.decode(mu, xp))
            err = ((yb - yh) ** 2).mean(dim=[0, 1]).cpu().numpy()
            if sq_err_sum is None:
                sq_err_sum = err * xb.size(0)
            else:
                sq_err_sum += err * xb.size(0)
            n_windows += xb.size(0)
            if n_windows >= max_windows:
                break
    return sq_err_sum / max(1, n_windows)


def build_model_from_run(run_dir, x_dim, y_dim, discrete_mask, device):
    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        run_cfg = json.load(f)
    m_cfg = run_cfg['model']
    d_cfg = run_cfg['dataset']

    model = JointCVAE(
        x_dim=x_dim, y_dim=y_dim,
        sequence_length=d_cfg['window_size'],
        patch_length=d_cfg['patch_length'],
        latent_dim=m_cfg['latent_dim'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'],
        dec_hidden_dim=m_cfg['dec_hidden_dim'],
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
                    help='Joint CVAE training run directory (defaults to newest under results/wadi_joint_cvae)')
    ap.add_argument('--checkpoint', type=str, default='best_joint_cvae.pth')
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--n_seeds', type=int, default=5,
                    help='Number of random Monte Carlo evaluation seeds for statistical significance')
    ap.add_argument('--out', type=str, default='results/wadi_joint_cvae/joint_cvae_wadi_ablation.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print(f"JOINT CVAE (single-stage ablation) -- WADI EVALUATION ({args.n_seeds} SEEDS)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    if args.run_dir is None:
        base = PROJECT_ROOT / 'results' / 'wadi_joint_cvae'
        cands = sorted([d for d in base.iterdir() if d.is_dir() and (d / 'config.json').exists()])
        if not cands:
            raise SystemExit(f"No Joint CVAE run directories found under {base}")
        run_dir = cands[-1]
    else:
        run_dir = Path(args.run_dir)
    print(f"[*] Run directory: {run_dir}")

    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        run_cfg = json.load(f)
    d_cfg = run_cfg['dataset']

    eval_stride = 10
    ood_val = 10.0

    train_dataset, test_dataset = load_csv_dataset(
        data_root=os.path.join(str(PROJECT_ROOT), d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'], attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'], timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'], y_prefixes=d_cfg['y_prefixes'],
        window_size=d_cfg['window_size'], stride=eval_stride,
        x_override=d_cfg.get('x_override'), y_override=d_cfg.get('y_override'),
        drop_columns=d_cfg.get('drop_columns', []),
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
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

    K = args.K
    t0 = time.time()
    seeds = [42 + 1337 * i for i in range(args.n_seeds)]

    # Dynamic precision weights (calibrated on train)
    raw_var = estimate_joint_channel_variances(train_dataset, model, device)
    weights = 1.0 / (raw_var + 0.010)
    weights = weights / np.mean(weights)
    model.channel_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[*] Precision weights min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f}")

    train_s2 = extract_signals_joint(model, train_dataset, d_cfg['window_size'], eval_stride, ood_val, device,
                                     K=K, max_batches=12)
    mu_train_is = np.mean(train_s2['is_cnll'])
    std_train_is = np.std(train_s2['is_cnll'])

    tr_len = len(train_s2['slices']) * eval_stride + d_cfg['window_size']
    tr_cons_raw = build_consensus_variance(train_s2['y_hat'], train_s2['slices'], tr_len, weights)
    pt_counts = np.zeros(tr_len, dtype=int)
    for (ws, we) in train_s2['slices']:
        pt_counts[ws:we] += 1
    mu_train_cons = np.mean(tr_cons_raw[pt_counts > 1])
    std_train_cons = np.std(tr_cons_raw[pt_counts > 1])

    # Run multi-seed evaluation
    stage3_runs = []
    print(f"\n[EVALUATING JOINT CVAE ACROSS {len(seeds)} SEEDS: {seeds}]")
    for s_idx, sd in enumerate(seeds):
        torch.manual_seed(sd)
        np.random.seed(sd)
        test_s2 = extract_signals_joint(model, test_dataset, d_cfg['window_size'], eval_stride, ood_val, device, K=K)
        total_len = test_s2['total_len']
        slices = test_s2['slices']

        med_is = median_aggregate(test_s2['is_cnll'], slices, total_len)
        cons_raw = build_consensus_variance(test_s2['y_hat'], slices, total_len, weights)
        z1_raw = (med_is - mu_train_is) / (std_train_is + 1e-6)
        z2_raw = (cons_raw - mu_train_cons) / (std_train_cons + 1e-6)

        headline_score = z1_raw + 1.0 * z2_raw
        m = evaluate_all_metrics(y_true, headline_score)
        stage3_runs.append(m)
        print(f"  Seed {sd:5d} ({s_idx+1}/{len(seeds)}) | F1={m['Point_F1']:.4f} Prec={m['Precision']*100:5.2f}% Rec={m['Recall']*100:5.2f}% "
              f"AUC-PR={m['AUC_PR']:.4f} AUC-ROC={m['AUC_ROC']:.4f} VUS-ROC={m['VUS_ROC']:.4f} VUS-PR={m['VUS_PR']:.4f} Affil={m['Affiliation_F1']:.4f} PATE={m['PATE_AUC_PR']:.4f}")

    # Compute Joint CVAE mean +/- std and 95% CI
    metric_names = ['Point_F1', 'Precision', 'Recall', 'AUC_ROC', 'AUC_PR', 'VUS_ROC', 'VUS_PR', 'Affiliation_F1', 'PATE_AUC_PR']
    joint_stats = {}
    for k in metric_names:
        vals = [r[k] for r in stage3_runs]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = float(1.96 * std_v / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        joint_stats[k] = {'mean': mean_v, 'std': std_v, 'ci95': ci95, 'raw': vals}

    # Conditioning Bypass Diagnostic
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
    print(f"  MSE  no_z  (z=0,  x)   = {bypass['no_z']:.6f}   (x{bypass['no_z']/max(bypass['full'],1e-12):.2f} vs full)")
    print(f"  MSE  no_x  (z=mu, 0)   = {bypass['no_x']:.6f}   (x{bypass['no_x']/max(bypass['full'],1e-12):.2f} vs full)")

    # Load TALON's multi-seed stats from final_table_rows.json if available
    talon_stats = {}
    ft_path = PROJECT_ROOT / 'results' / 'final_table_rows.json'
    if ft_path.exists():
        with open(ft_path) as f:
            ft = json.load(f)
        wadi_ft = ft.get('datasets', {}).get('WADI', {}).get('students', {}).get('trained', {})
        t_runs = wadi_ft.get('per_seed', [])[:len(seeds)]
        for k in metric_names:
            vals = [r[k] for r in t_runs]
            if vals:
                talon_stats[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'ci95': float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    'raw': vals
                }

    print("\n" + "=" * 115)
    print(f"HEADLINE STATISTICAL SIGNIFICANCE COMPARISON -- WADI ({len(seeds)} SEEDS)")
    print("=" * 115)
    lbl_map = {
        'Point_F1': 'Point-F1', 'Precision': 'Precision', 'Recall': 'Recall',
        'AUC_ROC': 'AUC-ROC', 'AUC_PR': 'AUC-PR', 'VUS_ROC': 'VUS-ROC',
        'VUS_PR': 'VUS-PR', 'Affiliation_F1': 'Affiliation-F1', 'PATE_AUC_PR': 'PATE'
    }
    print(f"{'Metric':<18} | {'TALON (Mean +/- CI)':<26} | {'Joint CVAE (Mean +/- CI)':<26} | {'Delta (Joint - TALON)':>22}")
    print("-" * 115)
    for k in metric_names:
        lbl = lbl_map[k]
        ts = talon_stats.get(k, {})
        js = joint_stats[k]
        t_str = f"{ts.get('mean', float('nan')):.4f} +/- {ts.get('ci95', 0.0):.4f}" if ts else "--"
        j_str = f"{js['mean']:.4f} +/- {js['ci95']:.4f}"
        delta_str = f"{js['mean'] - ts['mean']:+9.4f}" if ts else "--"
        print(f"{lbl:<18} | {t_str:<26} | {j_str:<26} | {delta_str:>22}")
    print("=" * 115)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_dir': str(run_dir),
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ck.get('epoch'),
        'checkpoint_best_val_loss': ck.get('best_val_loss'),
        'K': K,
        'n_seeds': len(seeds),
        'seeds': seeds,
        'joint_cvae_stats': joint_stats,
        'joint_cvae_runs': stage3_runs,
        'talon_reference_stats': talon_stats,
        'conditioning_bypass_mse': bypass,
        'wall_clock_sec': time.time() - t0,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME]  {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

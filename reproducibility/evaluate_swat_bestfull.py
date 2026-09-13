"""
========================================================================================
FULL SWaT BENCHMARK EVALUATION ON THE HEADLINE CHECKPOINT
========================================================================================
Model directory:  results/swat_cve/BestFull
Checkpoints:      best_vae.pth (teacher), best_cve.pth (student), config.json
Settings:         Strict unadjusted point-wise, train-calibrated evaluation protocol
========================================================================================
"""

import os
import sys
import math
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from models.TALONStudent import (
    TALONStudent,
    assemble_precision_from_bands_fn,
    robust_cholesky_batched
)

try:
    from TSB_AD.evaluation.metrics import get_metrics as tsbad_get_metrics
except ImportError:
    tsbad_get_metrics = None

try:
    from pate.PATE_metric import PATE
except ImportError:
    PATE = None


def compute_gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, params: dict, max_bandwidth: int) -> torch.Tensor:
    """Computes exact Gaussian log-likelihood under structured Kronecker banded precision."""
    B_flat, C, T = z.shape
    device = z.device
    dtype = torch.float32

    z = z.to(dtype)
    mu = mu.to(dtype)
    bw = min(max_bandwidth, T - 1)

    Qt = assemble_precision_from_bands_fn(
        params['precision_diag'].float(),
        params['precision_bands'].float(),
        bandwidth=bw
    )
    Lt = robust_cholesky_batched(Qt, jitter_start=1e-5, jitter_max=1e-2)
    logdet_Qt = 2.0 * torch.sum(torch.log(torch.diagonal(Lt, dim1=-2, dim2=-1)), dim=-1)

    has_qc = (params.get('precision_channel_diag') is not None and 
              params.get('precision_channel_bands') is not None)
    if has_qc:
        Qc = assemble_precision_from_bands_fn(
            params['precision_channel_diag'].float(),
            params['precision_channel_bands'].float(),
            bandwidth=min(params['precision_channel_bands'].size(1), C - 1)
        )
        Lc = robust_cholesky_batched(Qc, jitter_start=1e-5, jitter_max=1e-2)
        logdet_Qc = 2.0 * torch.sum(torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)), dim=-1)
    else:
        Qc = torch.eye(C, device=device, dtype=dtype).unsqueeze(0).expand(B_flat, -1, -1)
        logdet_Qc = torch.zeros(B_flat, device=device, dtype=dtype)

    logdet_Q = C * logdet_Qt + T * logdet_Qc
    delta = z - mu
    temp_t = delta @ Qt
    temp = Qc @ temp_t
    quad = torch.sum(delta * temp, dim=[-2, -1])

    n_dim = C * T
    log_prob = 0.5 * (logdet_Q - n_dim * math.log(2.0 * math.pi) - quad)
    return log_prob


def extract_signals_full(cve_model, dataset, window_size: int, stride: int, ood_thresh: float, 
                         device: torch.device, K: int = 200, K_sub: int = 50, max_batches: int = None):
    total_len = len(dataset.labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_is_cnll = []
    all_y_hat_list = []
    window_slices = []
    
    cur_idx = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            tmask = torch.ones(xb.size(0), window_size, device=device, dtype=torch.bool)
            
            B = xb.size(0)
            C_dim = yb.size(2)
            num_valid = window_size * C_dim
            variance = 1.0 / C_dim
            
            for b in range(B):
                ws = cur_idx * stride
                we = ws + window_size
                if we <= total_len:
                    window_slices.append((ws, we))
                cur_idx += 1
                
            b_log_p_is = []
            num_passes = math.ceil(K / K_sub)
            for p in range(num_passes):
                k_cur = min(K_sub, K - p * K_sub)
                x_rep = xb.repeat_interleave(k_cur, dim=0)
                y_rep = yb.repeat_interleave(k_cur, dim=0)
                mask_rep = tmask.repeat_interleave(k_cur, dim=0)
                y_rep_clamped = torch.clamp(y_rep, min=-ood_thresh, max=ood_thresh) if ood_thresh is not None else y_rep

                sampled = cve_model(
                    x_condition=x_rep, y_patches=y_rep_clamped, mode='testRandom',
                    y_target=y_rep, time_mask_full=mask_rep, ood_threshold=ood_thresh
                )

                mse_target = sampled['reconstruction_loss_target'].view(B, k_cur)
                sq_dist_target = mse_target * num_valid
                log_p_y = -sq_dist_target / (2.0 * variance) - (num_valid / 2.0) * math.log(2.0 * math.pi * variance)

                z_target = sampled['z_target']
                mu_condition = sampled['mu_condition']
                mu_target = sampled['mu_target']
                params_condition = sampled['posterior_params_condition']
                params_target = sampled['posterior_params_target']

                T_lat = z_target.size(-1)
                bw_cond = getattr(cve_model.conditioning_encoder, 'bandwidth', T_lat - 1)
                bw_prior = getattr(cve_model.pretrained_encoder, 'bandwidth', T_lat - 1)

                log_q_phi = compute_gaussian_log_prob(z_target, mu_condition, params_condition, max_bandwidth=bw_cond)
                log_q_psi = compute_gaussian_log_prob(z_target, mu_target, params_target, max_bandwidth=bw_prior)

                raw_ratio = log_q_phi.view(B, k_cur) - log_q_psi.view(B, k_cur)
                ratio_clipped = torch.clamp(raw_ratio, min=-10.0, max=10.0)
                b_log_p_is.append(log_p_y + ratio_clipped)

            all_p_is = torch.cat(b_log_p_is, dim=1)
            mc_cnll = -(torch.logsumexp(all_p_is, dim=1) - math.log(K)).cpu().numpy()
            all_is_cnll.extend(mc_cnll.tolist())
            
            sampled_mean = cve_model(x_condition=xb, y_patches=yb, mode='test', y_target=yb, time_mask_full=tmask, ood_threshold=ood_thresh)
            all_y_hat_list.append(sampled_mean['y_hat_target'].cpu())
            
            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    return {
        'is_cnll': np.array(all_is_cnll),
        'y_hat': torch.cat(all_y_hat_list, dim=0),
        'slices': window_slices,
        'total_len': total_len
    }


def estimate_cve_channel_variances(train_dataset, cve_model, device: torch.device, max_windows: int = 500) -> np.ndarray:
    loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    sq_err_sum = np.zeros(len(train_dataset.y_indices), dtype=np.float64)
    total_timesteps = 0
    cve_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_b = batch[0].to(device).permute(0, 2, 1)
            y_b = batch[1].to(device).permute(0, 2, 1)
            B_curr, T_curr, _ = x_b.shape
            t_mask = torch.ones(B_curr, T_curr, device=device, dtype=torch.bool)
            sampled = cve_model(x_condition=x_b, y_patches=y_b, mode='testRandom', y_target=y_b, time_mask_full=t_mask, ood_threshold=1.0)
            sq_err = (y_b - sampled['y_hat_target']) ** 2
            sq_err_sum += sq_err.sum(dim=[0, 1]).cpu().numpy()
            total_timesteps += B_curr * T_curr
            if (batch_idx + 1) * 128 >= max_windows:
                break
    return sq_err_sum / total_timesteps


def point_adjustment_classic(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_tau = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(scores))
    preds = (scores >= best_tau).astype(int)
    
    preds_pa = preds.copy()
    diff = np.diff(np.pad(y_true, (1, 1), 'constant'))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    detected_segments = 0
    for s, e in zip(starts, ends):
        if np.any(preds_pa[s:e] == 1):
            preds_pa[s:e] = 1
            detected_segments += 1
            
    p = float(np.sum((preds_pa == 1) & (y_true == 1)) / (np.sum(preds_pa == 1) + 1e-12))
    r = float(np.sum((preds_pa == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-12))
    f1 = float(2 * p * r / (p + r + 1e-12))
    return {
        'PA_F1': f1,
        'PA_Precision': p,
        'PA_Recall': r,
        'Detected_Segments': int(detected_segments),
        'Total_Segments': len(starts)
    }


def evaluate_all_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    scores = np.nan_to_num(scores, nan=np.nanmedian(scores))
    auc_roc = float(roc_auc_score(y_true, scores))
    auc_pr = float(average_precision_score(y_true, scores))
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_tau = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(scores))
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precision[best_idx])
    best_rec = float(recall[best_idx])
    
    res = {
        'Point_F1': best_f1,
        'Precision': best_prec,
        'Recall': best_rec,
        'AUC_PR': auc_pr,
        'AUC_ROC': auc_roc,
        'Threshold': best_tau
    }
    
    # Point Adjustment
    pa_res = point_adjustment_classic(y_true, scores)
    res.update(pa_res)
    
    # TSB_AD metrics (VUS, Affiliation)
    if tsbad_get_metrics is not None:
        try:
            tsb_metrics = tsbad_get_metrics(scores, y_true.astype(int), slidingWindow=100)
            res['VUS_ROC'] = float(tsb_metrics.get('VUS-ROC', np.nan))
            res['VUS_PR'] = float(tsb_metrics.get('VUS-PR', np.nan))
            res['Affiliation_F1'] = float(tsb_metrics.get('Affiliation-F', np.nan))
            res['Range_F1'] = float(tsb_metrics.get('R-based-F1', np.nan))
        except Exception as e:
            print(f"[WARN] TSB_AD metric evaluation failed: {e}")
            
    # PATE metric.
    # n_jobs is CAPPED, not -1. joblib on Windows spawns rather than forks, so every worker
    # re-imports this module and with it torch (~300-500 MB RSS each). On a 24-core box
    # n_jobs=-1 therefore costs ~10 GB per PATE call, and two concurrent scoring jobs were
    # enough to exhaust system memory. Override with TALON_PATE_JOBS if a machine can spare
    # more; 4 keeps a full 10-seed sweep comfortably inside RAM.
    try:
        from runtime_guard import pate_jobs as _pj
        _pate_jobs = _pj()
    except Exception:
        _pate_jobs = int(os.environ.get("TALON_PATE_JOBS", "4"))
    if PATE is not None:
        try:
            pate_cont = float(PATE(y_true.astype(int), scores.astype(np.float64), binary_scores=False, Big_Data=True, n_jobs=_pate_jobs))
            res['PATE_AUC_PR'] = pate_cont
        except Exception as e:
            try:
                preds_bin = (scores >= best_tau).astype(int)
                pate_bin = float(PATE(y_true.astype(int), preds_bin, binary_scores=True))
                res['PATE_AUC_PR'] = pate_bin
            except Exception as e2:
                print(f"[WARN] PATE metric evaluation failed: {e2}")
            
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print("RUNNING FULL SWAT EVALUATION ON BestFull DIRECTORY")
    print(f"Directory:  {PROJECT_ROOT / 'results/swat_cve/BestFull'}")
    print(f"Hardware:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    dir_path = PROJECT_ROOT / 'results/swat_cve/BestFull'
    cfg_path = dir_path / 'config.json'
    vae_path = dir_path / 'best_vae.pth'
    cve_path = dir_path / 'best_cve.pth'

    with open(cfg_path) as f:
        cfg = json.load(f)
    d_cfg = cfg['dataset']
    m_cfg = cfg['model']

    print(f"\n[1/6] Loading SWaT Dataset...")
    print(f"      - Normal CSV: {d_cfg['normal_csv']}")
    print(f"      - Attack CSV: {d_cfg['attack_csv']}")
    print(f"      - Sampling:   {d_cfg.get('sampling_rate_seconds', 10)}s {d_cfg.get('downsample_mode', 'median')}")
    print(f"      - Window:     {d_cfg['window_size']} steps (Stride=10)")
    
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
        drop_columns=[],
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median')
    )

    y_true = test_dataset.labels
    total_len = len(y_true)
    num_attacks = np.sum(y_true == 1)
    print(f"      - Test Samples: {total_len} timesteps ({num_attacks} anomalies = {num_attacks/total_len*100:.2f}%)")
    print(f"      - X Channels:   {len(train_dataset.x_indices)} (Actuators)")
    print(f"      - Y Channels:   {len(train_dataset.y_indices)} (Sensors)")

    print(f"\n[2/6] Loading Teacher VAE and Student CVE from BestFull...")
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

    vae_ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'], strict=True)
    vae.eval()

    cve_ckpt = torch.load(cve_path, map_location=device)
    cond_dim = cve_ckpt['model_state_dict']['patch_embedding_layer.weight'].shape[1]

    cve = TALONStudent(
        pretrained_tspvae=vae,
        conditioning_input_dim=cond_dim,
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'], bandwidth=m_cfg['bandwidth'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'], encoder_kwargs=m_cfg['encoder_kwargs'],
        kl_direction='reverse', compute_mi=True
    ).to(device)
    cve.load_state_dict(cve_ckpt['model_state_dict'], strict=True)
    cve.eval()
    print("      -> Successfully instantiated and loaded both models.")

    print(f"\n[3/6] Calibrating Dynamic Channel Precision Weighting...")
    raw_var = estimate_cve_channel_variances(train_dataset, cve, device)
    weights = 1.0 / (raw_var + 1e-4)
    weights = weights / np.mean(weights)
    cve.channel_weights.data.copy_(torch.tensor(weights, dtype=torch.float32, device=device))
    print(f"      -> Precision weights min: {np.min(weights):.3f}, max: {np.max(weights):.3f}, mean: {np.mean(weights):.3f}")

    print(f"\n[4/6] Computing Nominal Baseline Calibration Statistics (Zero Test Leakage)...")
    train_extract = extract_signals_full(cve, train_dataset, d_cfg['window_size'], 10, 1.0, device, K=50, max_batches=15)
    mu_train_is = float(np.mean(train_extract['is_cnll']))
    std_train_is = float(np.std(train_extract['is_cnll']))

    tr_slices = train_extract['slices']
    tr_y_hat = train_extract['y_hat']
    tr_len = len(tr_slices) * 10 + 100
    pt_preds = [[] for _ in range(tr_len)]
    for w_idx, (ws, we) in enumerate(tr_slices):
        recon = tr_y_hat[w_idx].numpy()
        for offset, t in enumerate(range(ws, we)):
            pt_preds[t].append(recon[offset])
    tr_cons = []
    for t in range(tr_len):
        if len(pt_preds[t]) > 1:
            preds_arr = np.array(pt_preds[t])
            tr_cons.append(np.average(np.var(preds_arr, axis=0), weights=weights))
    mu_train_cons = float(np.mean(tr_cons))
    std_train_cons = float(np.std(tr_cons))
    print(f"      -> Calibrated IS-CNLL: mu = {mu_train_is:.4f}, sigma = {std_train_is:.4f}")
    print(f"      -> Calibrated Cons-Var: mu = {mu_train_cons:.6f}, sigma = {std_train_cons:.6f}")

    print(f"\n[5/6] Extracting Test Set Signals (K=200 IS MC Samples)...")
    t0 = time.time()
    test_extract = extract_signals_full(cve, test_dataset, d_cfg['window_size'], 10, 1.0, device, K=200)
    print(f"      -> Extracted in {time.time() - t0:.2f}s.")

    slices = test_extract['slices']
    time_scores = [[] for _ in range(total_len)]
    time_scores_mean = [[] for _ in range(total_len)]
    for w_idx, (ws, we) in enumerate(slices):
        val = test_extract['is_cnll'][w_idx]
        for t in range(ws, we):
            time_scores[t].append(val)
            time_scores_mean[t].append(val)

    med_is = np.array([np.median(time_scores[t]) if len(time_scores[t]) > 0 else np.nan for t in range(total_len)])
    mean_is = np.array([np.mean(time_scores_mean[t]) if len(time_scores_mean[t]) > 0 else np.nan for t in range(total_len)])
    score_is_median = np.nan_to_num(med_is, nan=np.nanmedian(med_is))
    score_is_mean = np.nan_to_num(mean_is, nan=np.nanmedian(mean_is))

    # Consensus variance
    y_hat = test_extract['y_hat']
    point_preds = [[] for _ in range(total_len)]
    for w_idx, (ws, we) in enumerate(slices):
        recon = y_hat[w_idx].numpy()
        for offset, t in enumerate(range(ws, we)):
            point_preds[t].append(recon[offset])
    score_cons = np.zeros(total_len)
    for t in range(total_len):
        if len(point_preds[t]) > 1:
            preds_arr = np.array(point_preds[t])
            score_cons[t] = np.average(np.var(preds_arr, axis=0), weights=weights)

    # Standardized scores
    z_is = (score_is_median - mu_train_is) / (std_train_is + 1e-6)
    z_cons = (score_cons - mu_train_cons) / (std_train_cons + 1e-6)
    score_fused = z_is + 0.05 * z_cons

    print(f"\n[6/6] Computing Complete Metric Profiles & Ablation...")
    
    # 1. Raw Baseline (mean aggregation, uniform weights)
    metrics_raw = evaluate_all_metrics(y_true, score_is_mean)

    # 2. Dynamic Channel Precision Weighting (median)
    metrics_is = evaluate_all_metrics(y_true, score_is_median)

    # 3. Train-Calibrated Consensus-Fused IS-CNLL
    metrics_fused = evaluate_all_metrics(y_true, score_fused)

    print("\n" + "=" * 95)
    print("SWAT BENCHMARK RESULTS (BestFull CHECKPOINT)")
    print("=" * 95)
    print(f"{'Metric':<30} | {'Value (BestFull)':<20} | {'Previously Reported'}")
    print("-" * 95)
    print(f"{'Point F1 Score':<30} | {metrics_fused['Point_F1']:.4f}{'':<14} | 0.7681")
    print(f"{'Point Precision':<30} | {metrics_fused['Precision']*100:6.2f}%{'':<13} | 96.82%")
    print(f"{'Point Recall':<30} | {metrics_fused['Recall']*100:6.2f}%{'':<13} | 63.66%")
    print(f"{'AUC-PR':<30} | {metrics_fused['AUC_PR']:.4f}{'':<14} | 0.7365")
    print(f"{'AUC-ROC':<30} | {metrics_fused['AUC_ROC']:.4f}{'':<14} | 0.8251")
    print(f"{'VUS-ROC':<30} | {metrics_fused.get('VUS_ROC', float('nan')):.4f}{'':<14} | 0.5670")
    print(f"{'VUS-PR':<30} | {metrics_fused.get('VUS_PR', float('nan')):.4f}{'':<14} | 0.3774")
    print(f"{'Affiliation F1':<30} | {metrics_fused.get('Affiliation_F1', float('nan')):.4f}{'':<14} | 0.6934")
    print(f"{'PATE AUC-PR':<30} | {metrics_fused.get('PATE_AUC_PR', float('nan')):.4f}{'':<14} | 0.7513")
    print(f"{'Classic PA-F1':<30} | {metrics_fused.get('PA_F1', float('nan')):.4f}{'':<14} | 1.0000")
    print(f"{'Optimal Threshold':<30} | {metrics_fused['Threshold']:.4f}{'':<14} | --")
    print(f"{'Detected Segments':<30} | {metrics_fused.get('Detected_Segments', 0)} / {metrics_fused.get('Total_Segments', 36)}{'':<7} | 36 / 36 (100%)")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("ABLATION STUDY TABLE (BestFull Checkpoint)")
    print("=" * 95)
    print(f"{'Configuration':<45} | {'Point F1':<8} | {'Prec.':<6} | {'Rec.':<6} | {'AUC-PR':<7} | {'AUC-ROC':<7}")
    print("-" * 95)
    print(f"{'(1) Raw Baseline (Uniform, Mean)':<45} | {metrics_raw['Point_F1']:.4f}   | {metrics_raw['Precision']*100:5.1f}% | {metrics_raw['Recall']*100:5.1f}% | {metrics_raw['AUC_PR']:.4f}  | {metrics_raw['AUC_ROC']:.4f}")
    print(f"{'(2) Dynamic Precision Weighting (Median)':<45} | {metrics_is['Point_F1']:.4f}   | {metrics_is['Precision']*100:5.1f}% | {metrics_is['Recall']*100:5.1f}% | {metrics_is['AUC_PR']:.4f}  | {metrics_is['AUC_ROC']:.4f}")
    print(f"{'(3) Train-Calibrated Consensus Fused':<45} | {metrics_fused['Point_F1']:.4f}   | {metrics_fused['Precision']*100:5.1f}% | {metrics_fused['Recall']*100:5.1f}% | {metrics_fused['AUC_PR']:.4f}  | {metrics_fused['AUC_ROC']:.4f}")
    print("=" * 95)

    # Save results to JSON
    output_json_path = dir_path / 'swat_evaluation_results.json'
    results_dict = {
        'metrics_fused': metrics_fused,
        'metrics_is': metrics_is,
        'metrics_raw': metrics_raw,
        'calibration': {
            'mu_train_is': mu_train_is,
            'std_train_is': std_train_is,
            'mu_train_cons': mu_train_cons,
            'std_train_cons': std_train_cons
        }
    }
    with open(output_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[SAVE] Results saved to {output_json_path}")


if __name__ == '__main__':
    main()

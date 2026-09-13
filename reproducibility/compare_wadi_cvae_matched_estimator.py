"""
========================================================================================
WADI: JOINT CVAE vs TALON UNDER A MATCHED SCORE FUNCTION
========================================================================================
WHY THIS EXISTS
    The published WADI Joint CVAE row and the published WADI TALON row are not computed
    with the same score. TALON uses the corrected direct Monte Carlo estimator

        S_TALON = -log[ (1/K) sum_k p_theta(y | z_k) ],    z_k ~ q_psi(z | x)

    while evaluate_wadi_joint_cvae.py's `extract_signals_joint` adds a clamped posterior
    importance weight to every sample before the logsumexp:

        S_CVAE  = -log[ (1/K) sum_k p(y | z_k, x) * exp(clamp(log p_prior(z_k)
                                                              - log q(z_k | x, y), -10, 10)) ]

    That extra term is the superseded clipped importance-sampling estimator this project
    replaced everywhere else, and it is a latent-atypicality signal the TALON score does
    not contain at all. The CVAE posterior also sees Y at test time (q(z|x,y)), so the
    ratio measures how unusual the latent is given the observed target -- a second
    detection channel TALON structurally cannot use.

    Sample budget is NOT the confound: results/wadi_joint_cvae/mc_budget_*/comparison.md
    already showed K=50 -> K=200 moves every CVAE metric by <= 0.004 and flips no winner.
    This script isolates the remaining difference, the estimator itself.

WHAT IS HELD FIXED (everything except the score function)
    ood clamp 10.0, eval stride 10, window from the run config, channel-precision weights
    calibrated on each model's own nominal residuals with the same 0.010 floor, median
    window aggregation, train-only standardization from the same 12 calibration batches,
    linear consensus fusion at lambda = 1, and the same evaluate_all_metrics call.
    Consensus is each model's own decode (TALON: teacher branch per SPEC['WADI']['cons']).

WHAT IS SWEPT
    Three CVAE score variants from ONE extraction pass -- the per-sample log p(y|z) and
    log-ratio matrices are retained [n_windows, K] and combined three ways at the end, so
    the variants differ only in arithmetic, never in sampling:
        no_ratio  : matched to TALON's estimator            <-- the comparable number
        is_clip   : clamp(ratio, +-10), the published row   <-- reproduces the old result
        is_raw    : unclamped ratio
    TALON is recomputed here at the same K and seeds rather than read from
    results/final_table_rows.json, whose WADI entry is stale (it predates the corrected
    WADI pipeline and does not match the published row).
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
from models.JointCVAE import gp_prior_log_prob
from evaluate_swat_bestfull import compute_gaussian_log_prob, evaluate_all_metrics
from rebuild_swat_ablation_bestfull import median_aggregate, build_consensus_variance
from evaluate_wadi_joint_cvae import build_model_from_run, estimate_joint_channel_variances
from final_table_rows import SPEC, extract, chan_weights, load_dataset_and_model
from evaluate_swat_stage2_ablations import build_student

VARIANTS = ['no_ratio', 'is_clip', 'is_raw']
METRICS = ['Point_F1', 'Precision', 'Recall', 'AUC_ROC', 'AUC_PR',
           'VUS_ROC', 'VUS_PR', 'Affiliation_F1', 'PATE_AUC_PR']


def extract_cvae_components(model, dataset, W, stride, ood, device, K, K_sub=50, zero_y=False):
    """`extract_signals_joint`, but retaining the per-sample log p(y|z) and log-ratio
    matrices instead of collapsing them, so all three estimator variants come from one
    set of draws.

    zero_y=True encodes with the Y branch blanked (q(z|x, 0) instead of q(z|x,y)), a
    no-retraining proxy for a conditional prior p(z|x): a cheap way to probe what the
    posterior looks like without Y, without training a separate prior network."""
    total_len = len(dataset.labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    logp_all, ratio_all, yh_all, slices, cur = [], [], [], [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device).permute(0, 2, 1)
            yb = batch[1].to(device).permute(0, 2, 1)
            tmask = torch.ones(xb.size(0), W, device=device, dtype=torch.bool)
            B, C = xb.size(0), yb.size(2)
            nv, var = W * C, 1.0 / C
            const = (nv / 2.0) * math.log(2.0 * math.pi * var)
            for _ in range(B):
                ws, we = cur * stride, cur * stride + W
                if we <= total_len:
                    slices.append((ws, we))
                cur += 1
            lp_acc, rt_acc = [], []
            for p in range(math.ceil(K / K_sub)):
                k = min(K_sub, K - p * K_sub)
                x_rep = xb.repeat_interleave(k, 0)
                y_rep = yb.repeat_interleave(k, 0)
                m_rep = tmask.repeat_interleave(k, 0)
                y_rep_c = torch.clamp(y_rep, min=-ood, max=ood) if ood is not None else y_rep
                y_enc_in = torch.zeros_like(y_rep_c) if zero_y else y_rep_c
                mu, params, x_patches = model.encode(x_rep, y_enc_in)
                z = model.sample_posterior(mu, params)
                y_hat = model.decode(z, x_patches)
                recon = model.compute_scoring_reconstruction_loss(
                    y_rep, y_hat, time_mask=m_rep.float(), ood_threshold=ood)
                mse = recon.view(B, k)
                lp_acc.append(-(mse * nv) / (2.0 * var) - const)
                bw = getattr(model.encoder, 'bandwidth', z.size(-1) - 1)
                log_q = compute_gaussian_log_prob(z, mu, params, max_bandwidth=bw)
                log_pr = gp_prior_log_prob(z, model.gp_prior)
                rt_acc.append((log_pr.view(B, k) - log_q.view(B, k)))
            logp_all.append(torch.cat(lp_acc, 1).cpu())
            ratio_all.append(torch.cat(rt_acc, 1).cpu())
            mu_det, _, xp_det = model.encode(xb, yb)
            yh_all.append(model.apply_discrete_sigmoid(model.decode(mu_det, xp_det)).cpu())
    n = len(slices)
    return {'logp': torch.cat(logp_all, 0)[:n], 'ratio': torch.cat(ratio_all, 0)[:n],
            'y_hat': torch.cat(yh_all, 0)[:n], 'slices': slices, 'total_len': total_len}


def cnll_variant(comp, kind, K):
    """Collapse the retained matrices into one of the three estimators."""
    lp = comp['logp'][:, :K]
    if kind == 'no_ratio':
        tot = lp
    elif kind == 'is_clip':
        tot = lp + torch.clamp(comp['ratio'][:, :K], min=-10.0, max=10.0)
    elif kind == 'is_raw':
        tot = lp + comp['ratio'][:, :K]
    else:
        raise ValueError(kind)
    return -(torch.logsumexp(tot, dim=1) - math.log(K)).numpy()


def summarize(rows):
    out = {}
    for k in METRICS:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = {'mean': float(np.mean(v)),
                      'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    return out


def main():
    _preflight(label='compare_wadi_cvae_matched_estimator')
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1379, 2716, 4053, 5390])
    ap.add_argument('--run_dir', type=str, default=None)
    ap.add_argument('--checkpoint', type=str, default='best_joint_cvae.pth')
    ap.add_argument('--out', type=str, default='results/wadi_cvae_matched_estimator.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K, seeds, stride, ood = args.K, args.seeds, 10, 10.0
    t0 = time.time()

    # ---------------- Joint CVAE ----------------
    base = PROJECT_ROOT / 'results' / 'wadi_joint_cvae'
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        cands = sorted([d for d in base.iterdir() if d.is_dir() and (d / 'config.json').exists()])
        run_dir = cands[-1]
    print(f"[CVAE] run dir {run_dir}", flush=True)
    d_cfg = json.load(open(run_dir / 'config.json', encoding='utf-8'))['dataset']
    W = d_cfg['window_size']

    tr_ds, te_ds = load_csv_dataset(
        data_root=os.path.join(str(PROJECT_ROOT), d_cfg['data_root']),
        normal_csv=d_cfg['normal_csv'], attack_csv=d_cfg['attack_csv'],
        label_column=d_cfg['label_column'], timestamp_columns=d_cfg['timestamp_columns'],
        x_prefixes=d_cfg['x_prefixes'], y_prefixes=d_cfg['y_prefixes'],
        window_size=W, stride=stride,
        x_override=d_cfg.get('x_override'), y_override=d_cfg.get('y_override'),
        drop_columns=d_cfg.get('drop_columns', []),
        scaler_type=d_cfg.get('scaler_type', 'minmax'),
        downsample_rate=d_cfg.get('sampling_rate_seconds', 10),
        downsample_mode=d_cfg.get('downsample_mode', 'median'))
    y_true = te_ds.labels

    model, _ = build_model_from_run(run_dir, len(tr_ds.x_indices), len(tr_ds.y_indices),
                                    tr_ds.y_discrete_mask, device)
    ck = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    print(f"[CVAE] {args.checkpoint} epoch {ck.get('epoch')}", flush=True)

    raw_var = estimate_joint_channel_variances(tr_ds, model, device)
    w_cvae = 1.0 / (raw_var + 0.010)
    w_cvae = w_cvae / np.mean(w_cvae)
    model.channel_weights = torch.tensor(w_cvae, dtype=torch.float32)

    # Train calibration: one pass, per-variant mu/sigma (standardization is per-signal).
    tr_comp = extract_cvae_components(model, tr_ds, W, stride, ood, device, K)
    # 12 calibration batches x 64 windows, matching evaluate_wadi_joint_cvae.py's budget.
    n_cal = min(len(tr_comp['slices']), 12 * 64)
    for key in ('logp', 'ratio', 'y_hat'):
        tr_comp[key] = tr_comp[key][:n_cal]
    tr_comp['slices'] = tr_comp['slices'][:n_cal]
    cal_cvae = {}
    for v in VARIANTS:
        c = cnll_variant(tr_comp, v, K)
        cal_cvae[v] = (float(np.mean(c)), float(np.std(c)))
    tr_len = len(tr_comp['slices']) * stride + W
    tr_cons = build_consensus_variance(tr_comp['y_hat'], tr_comp['slices'], tr_len, w_cvae)
    cnt = np.zeros(tr_len, dtype=int)
    for (a, b) in tr_comp['slices']:
        cnt[a:b] += 1
    mu_c, sd_c = float(np.mean(tr_cons[cnt > 1])), float(np.std(tr_cons[cnt > 1]))
    print(f"[CVAE] calibration done ({time.time()-t0:.0f}s)", flush=True)

    cvae_rows = {v: [] for v in VARIANTS}
    for i, sd in enumerate(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        te = extract_cvae_components(model, te_ds, W, stride, ood, device, K)
        cons = (build_consensus_variance(te['y_hat'], te['slices'], te['total_len'], w_cvae)
                - mu_c) / (sd_c + 1e-6)
        for v in VARIANTS:
            c = cnll_variant(te, v, K)
            mu_v, sd_v = cal_cvae[v]
            z1 = (median_aggregate(c, te['slices'], te['total_len']) - mu_v) / (sd_v + 1e-6)
            m = evaluate_all_metrics(y_true, z1 + 1.0 * cons)
            cvae_rows[v].append(m)
            if v == 'no_ratio':
                # Decompose the fused score into its two additive halves, so a difference
                # against TALON can be attributed to the CNLL term or the consensus term
                # rather than to the sum. Costs no extra forward passes.
                cvae_rows.setdefault('no_ratio_cnll_only', []).append(
                    evaluate_all_metrics(y_true, z1))
                cvae_rows.setdefault('cons_only', []).append(
                    evaluate_all_metrics(y_true, cons))
        print(f"[CVAE] seed {sd} ({i+1}/{len(seeds)}) "
              + ' '.join(f"{v}:ROC={cvae_rows[v][-1]['AUC_ROC']:.4f}" for v in VARIANTS),
              flush=True)

    del model
    torch.cuda.empty_cache()

    # ---------------- TALON, recomputed at the same K/seeds ----------------
    sp = dict(SPEC['WADI'])
    t_tr, t_te, tW, t_y, vae, tck, m_cfg = load_dataset_and_model('WADI', device)
    cve = build_student(tck, vae, m_cfg, device)[0]
    cw = chan_weights(t_tr, cve, device, sp['ood'])
    cve.channel_weights.data.copy_(torch.tensor(cw, dtype=torch.float32, device=device))
    tr = extract(cve, t_tr, tW, device, 50, sp['ood'], max_batches=sp['train_batches'])
    mu_t, sd_t = float(np.mean(tr['cnll'])), float(np.std(tr['cnll']))
    tsl, tlen = tr['slices'], len(tr['slices']) * 10 + tW
    tcnt = np.zeros(tlen, dtype=int)
    for (a, b) in tsl:
        tcnt[a:b] += 1
    trc = build_consensus_variance(tr[f"y_hat_{sp['cons']}"], tsl, tlen, cw)
    mu_tc, sd_tc = float(np.mean(trc[tcnt > 1])), float(np.std(trc[tcnt > 1]))

    talon_rows, talon_cnll_only, talon_cons_only = [], [], []
    for i, sd in enumerate(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        te = extract(cve, t_te, tW, device, K, sp['ood'])
        z1 = (median_aggregate(te['cnll'], te['slices'], te['total_len']) - mu_t) / (sd_t + 1e-6)
        cons = (build_consensus_variance(te[f"y_hat_{sp['cons']}"], te['slices'],
                                         te['total_len'], cw) - mu_tc) / (sd_tc + 1e-6)
        m = evaluate_all_metrics(np.asarray(t_y), z1 + 1.0 * cons)
        talon_rows.append(m)
        talon_cnll_only.append(evaluate_all_metrics(np.asarray(t_y), z1))
        talon_cons_only.append(evaluate_all_metrics(np.asarray(t_y), cons))
        print(f"[TALON] seed {sd} ({i+1}/{len(seeds)}) ROC={m['AUC_ROC']:.4f}", flush=True)

    # ---------------- report ----------------
    res = {v: summarize(cvae_rows[v]) for v in cvae_rows}
    res['TALON'] = summarize(talon_rows)
    res['TALON_cnll_only'] = summarize(talon_cnll_only)
    res['TALON_cons_only'] = summarize(talon_cons_only)
    hdr = ['Point_F1', 'AUC_ROC', 'AUC_PR', 'VUS_ROC', 'VUS_PR', 'Affiliation_F1', 'PATE_AUC_PR']
    print('\n' + '=' * 100)
    print(f"{'configuration':28s}" + ''.join(f"{h[:9]:>10s}" for h in hdr))
    order = ['TALON', 'no_ratio', 'is_clip', 'is_raw',
             'TALON_cnll_only', 'no_ratio_cnll_only', 'TALON_cons_only', 'cons_only']
    for name in order:
        lbl = {'TALON': 'TALON fused',
               'no_ratio': 'CVAE fused, matched est.',
               'is_clip': 'CVAE fused, clipped IS (pub)',
               'is_raw': 'CVAE fused, unclipped IS',
               'TALON_cnll_only': '  TALON CNLL term only',
               'no_ratio_cnll_only': '  CVAE CNLL term only',
               'TALON_cons_only': '  TALON consensus only',
               'cons_only': '  CVAE consensus only'}[name]
        print(f"{lbl:28s}" + ''.join(f"{res[name][h]['mean']:10.4f}" for h in hdr))
    print('=' * 100)

    out = {'benchmark': 'WADI', 'K': K, 'seeds': seeds, 'stride': stride, 'ood': ood,
           'cvae_run_dir': str(run_dir), 'cvae_checkpoint_epoch': ck.get('epoch'),
           'talon_spec': {k: str(v) for k, v in sp.items()},
           'note': 'All post-processing identical; only the score function differs across '
                   'the three CVAE variants. TALON recomputed here, not read from '
                   'final_table_rows.json (whose WADI entry is stale).',
           'results': res,
           'per_seed': {'TALON': talon_rows, 'TALON_cnll_only': talon_cnll_only,
                        'TALON_cons_only': talon_cons_only,
                        **{v: cvae_rows[v] for v in cvae_rows}},
           'wall_clock_sec': time.time() - t0}
    with open(PROJECT_ROOT / args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[SAVED] {PROJECT_ROOT / args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()

"""
========================================================================================
DECISIVE TEST: DOES THE TRAINED STUDENT AFFECT THE REPORTED SWaT METRICS AT ALL?
========================================================================================
An earlier investigation established that the Stage-2 student, under the older clamped
importance-weighted estimator (superseded by the direct Monte Carlo estimator this project now
uses everywhere -- see final_table_rows.py's docstring), reached the SWaT score through
exactly one channel -- the importance log-weight

    w = log q_phi(z|X) - log q_psi(z|Y),   clamped to [-10, +10]

-- and that on the evaluated control checkpoint (epoch 1000) w has mean -468.7, so only
1 sample in 15,360 escapes the clamp. Separately, `extract_signals_full()` reads
`y_hat_target`, which is decoded from `mu_prior` (the FROZEN teacher's latent) under
`no_grad()`, not from the student's `mu_condition`; the student's own decode
(`y_hat_condition`) and reconstruction are never read.

Together those imply the reported metrics may be a function of the frozen teacher alone.
This script tests that implication directly and in the only way that settles it: run the
COMPLETE, UNMODIFIED scoring chain twice on identical windows, changing nothing but the
student's weights.

    trained  -- results/swat_stage2_ablation/control_s42/best_cve.pth (epoch 1000; this is
                the exact checkpoint the published ablation table was computed from)
    random   -- the SAME architecture, freshly initialised, never trained

Interpretation:

  * metrics essentially identical -> the trained student contributes nothing to the
    reported numbers. The conditioning pathway is inert under this scoring pipeline, and
    the finding reaches the headline results, not merely the objective-ablation paragraph.
  * metrics collapse for the random student -> the student matters through a path not
    captured by the log-weight analysis, and only the objective-ablation interpretation
    needs revising.

Nothing here modifies the scoring pipeline; `evaluate_one` is imported and used verbatim so
the comparison cannot drift from the published protocol. Read-only apart from one JSON.
========================================================================================
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

from datasets.LocalTSAD import load_csv_dataset
from models.TALONTeacher import TALONTeacher
from models.TALONStudent import TALONStudent
from evaluate_swat_stage2_ablations import build_student, evaluate_one, METRIC_KEYS


def build_random_student(cve_ckpt, vae, m_cfg, device, seed):
    """Same constructor arguments as build_student, but weights are left at initialisation.

    conditioning_input_dim and compute_mi are read from the trained checkpoint so the two
    students are architecturally identical and differ only in their parameter values.
    """
    sd = cve_ckpt['model_state_dict']
    has_mi = any('mi_estimator' in k for k in sd.keys())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cve = TALONStudent(
        pretrained_tspvae=vae,
        conditioning_input_dim=sd['patch_embedding_layer.weight'].shape[1],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'], bandwidth=m_cfg['bandwidth'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'],
        kl_direction='reverse',
        compute_mi=has_mi,
    ).to(device)
    cve.eval()
    return cve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str,
                    default='results/swat_stage2_ablation/control_s42/best_cve.pth')
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', type=str,
                    default='results/swat_stage2_ablation/random_student_control.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print("DECISIVE TEST: TRAINED STUDENT vs RANDOM STUDENT, IDENTICAL SCORING PIPELINE")
    print(f"Device: {device}")
    print("=" * 95)

    cfg = json.load(open(PROJECT_ROOT / 'results/swat_cve/BestFull/config.json'))
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

    vae = TALONTeacher(
        latent_dim=m_cfg['latent_dim'], input_dim=len(train_dataset.y_indices),
        sequence_length=d_cfg['window_size'], patch_length=d_cfg['patch_length'],
        enc_hidden_dim=m_cfg['enc_hidden_dim'], dec_hidden_dim=m_cfg['dec_hidden_dim'],
        gp_time_kernel=m_cfg['gp_time_kernel'], rank_c=m_cfg['rank_c'],
        gp_jitter=m_cfg['gp_jitter'], bandwidth=m_cfg['bandwidth'],
        posterior_tc_banded=m_cfg['tc_banded_enabled'],
        tc_channel_bandwidth=m_cfg['tc_channel_bandwidth'],
        encoder_kwargs=m_cfg['encoder_kwargs'], decoder_kwargs=m_cfg['decoder_kwargs'],
        discrete_mask=train_dataset.y_discrete_mask).to(device)
    vae.load_state_dict(torch.load(PROJECT_ROOT / 'results/swat_cve/BestFull/best_vae.pth',
                                   map_location=device)['model_state_dict'], strict=True)
    vae.eval()
    print("[*] shared frozen teacher loaded")

    ck = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device)
    print(f"[*] student checkpoint epoch={ck.get('epoch')} from {args.checkpoint}")

    out = {'checkpoint': args.checkpoint, 'checkpoint_epoch': ck.get('epoch'),
           'K': args.K, 'init_seed': args.seed, 'students': {}}

    for label in ('trained', 'random'):
        print()
        print("-" * 95)
        print(f"[>] STUDENT = {label.upper()}")
        print("-" * 95)
        if label == 'trained':
            cve, _ = build_student(ck, vae, m_cfg, device)
        else:
            cve = build_random_student(ck, vae, m_cfg, device, args.seed)

        stages = evaluate_one(cve, train_dataset, test_dataset, d_cfg, y_true, device,
                              K=args.K, verbose=True)
        out['students'][label] = stages

        del cve
        torch.cuda.empty_cache()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)

    # ------------------------------ side-by-side ------------------------------
    tr, rd = out['students']['trained'], out['students']['random']
    print()
    print("=" * 95)
    print("TRAINED vs RANDOM STUDENT -- identical windows, identical pipeline")
    print("=" * 95)
    for stage in tr:
        if stage not in rd:
            continue
        print(f"\n[{stage}]")
        print(f"  {'metric':22s} {'trained':>10s} {'random':>10s} {'delta':>10s}")
        for key, pretty in METRIC_KEYS:
            a, b = tr[stage].get(key), rd[stage].get(key)
            if a is None or b is None:
                continue
            print(f"  {pretty:22s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
    print()
    print("=" * 95)
    print("If the deltas are ~0, the trained student is inert under this scoring pipeline.")
    print(f"[SAVED] {out_path}")


if __name__ == '__main__':
    main()

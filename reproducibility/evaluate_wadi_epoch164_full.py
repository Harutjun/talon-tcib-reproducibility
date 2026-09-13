"""
========================================================================================
WADI EPOCH-164 STUDENT AT FULL PROTOCOL: 10 SEEDS, K=200, BOTH CONSENSUS BRANCHES
========================================================================================
WHY THIS EXISTS
    A SHA256-verified sweep over intermediate checkpoints found nine WADI student
    checkpoints (epoch 61-998) that all beat the published
    epoch-5092 headline on 6 of 7 metrics at a single seed. Ranked by average fractional
    rank across all ten (nine snapshots + published), epoch 164 wins outright: 2.43,
    leading AUC-ROC/AUC-PR/VUS-ROC/VUS-PR/PATE, versus 5092's 8.71 (last of ten, winning
    only Affiliation-F1). This script re-evaluates epoch 164 at the same 10-seed, K=200
    protocol as the published headline, so it can actually replace it rather than stay a
    single-seed hint -- and sweeps both consensus branches, since WADI's teacher-vs-student
    choice (see final_table_rows.SPEC's 'cons' field) was itself only ever checked against
    epoch 5092.

WHAT IS HELD FIXED
    Same teacher as SPEC['WADI'] (results/wadi_vae/WADI/BestFullChannels/best_vae.pth,
    epoch 1572) -- SHA256-verified identical to the one epoch164's own snapshot metadata
    recorded, so this is a clean like-for-like substitution of the student only. Same
    ood=10.0, train_batches=12, K_train=50, K_test=200, eval_stride=10, fusion
    lambda=1, median aggregation, and the same 10-seed list every other WADI/SWaT
    10-seed result in this project uses.

WHAT CHANGES
    The student checkpoint: the SHA256-verified snapshot at
    results/wadi_cve/headline_teacher_20260908_094048/evaluation_20260908_104237/
    checkpoint_snapshot.pth (epoch 164) instead of SPEC['WADI']'s epoch-5092 one.
    Consensus branch: both student and teacher decode are reported, not just the
    currently-published teacher branch.
========================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
from runtime_guard import cap_threads as _cap, preflight as _preflight
_cap()

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = PROJECT_ROOT / 'reproducibility'
for _p in (str(PROJECT_ROOT), str(REPRO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

from final_table_rows import SPEC, COLS, extract, chan_weights, load_dataset_and_model
from evaluate_swat_stage2_ablations import build_student
from evaluate_swat_bestfull import evaluate_all_metrics
from rebuild_swat_ablation_bestfull import median_aggregate, build_consensus_variance

SNAPSHOT = (PROJECT_ROOT / 'results' / 'wadi_cve' / 'headline_teacher_20260908_094048' /
           'evaluation_20260908_104237' / 'checkpoint_snapshot.pth')
SNAPSHOT_SHA256 = '08b32438c17c718e0711dd002b3a4c20de78eeb6b16ebf07813a11417e21eed8'
TEACHER_SHA256 = '0854e7e65295689b013c1d14b018bf96cd706e478216212ceeed63d517064b1c'
SEEDS = [42, 1379, 2716, 4053, 5390, 6727, 8064, 9401, 10738, 12075]


def fingerprint(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuse(kind, z1, cons):
    return z1 if kind == 'none' else z1 + 1.0 * cons


def main():
    _preflight(label='evaluate_wadi_epoch164_full')
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--K_test', type=int, default=200)
    ap.add_argument('--K_train', type=int, default=50)
    ap.add_argument('--out', type=str, default='results/wadi_epoch164_full.json')
    args = ap.parse_args()

    assert fingerprint(SNAPSHOT) == SNAPSHOT_SHA256, "epoch-164 snapshot changed on disk"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sp = dict(SPEC['WADI'])
    teacher_path = PROJECT_ROOT / sp['run'] / sp['vae']
    assert fingerprint(teacher_path) == TEACHER_SHA256, "teacher checkpoint changed on disk"

    train_ds, test_ds, W, y_true, vae, _, m_cfg = load_dataset_and_model('WADI', device)
    ck = torch.load(SNAPSHOT, map_location=device)
    cve = build_student(ck, vae, m_cfg, device)[0]
    print(f"[epoch164] loaded student from snapshot, checkpoint epoch={ck.get('epoch')}",
          flush=True)

    y = np.asarray(y_true)
    cw = chan_weights(train_ds, cve, device, sp['ood'])
    cve.channel_weights.data.copy_(torch.tensor(cw, dtype=torch.float32, device=device))

    tr = extract(cve, train_ds, W, device, args.K_train, sp['ood'],
                 max_batches=sp['train_batches'])
    mu, sd = float(np.mean(tr['cnll'])), float(np.std(tr['cnll']))
    tr_sl, tr_len = tr['slices'], len(tr['slices']) * 10 + W
    cnt = np.zeros(tr_len, dtype=int)
    for (a, b) in tr_sl:
        cnt[a:b] += 1
    cal = {}
    for branch in ('student', 'teacher'):
        trc = build_consensus_variance(tr[f'y_hat_{branch}'], tr_sl, tr_len, cw)
        cal[branch] = (float(np.mean(trc[cnt > 1])), float(np.std(trc[cnt > 1])))
    print(f"[epoch164] calibration: cnll mu={mu:.4f} sd={sd:.4f} | cons {cal}", flush=True)

    cached = []
    for i, s in enumerate(args.seeds):
        torch.manual_seed(s); np.random.seed(s)
        te = extract(cve, test_ds, W, device, args.K_test, sp['ood'])
        z1 = (median_aggregate(te['cnll'], te['slices'], te['total_len']) - mu) / (sd + 1e-6)
        conses = {}
        for branch in ('student', 'teacher'):
            mc, sc = cal[branch]
            conses[branch] = (build_consensus_variance(
                te[f'y_hat_{branch}'], te['slices'], te['total_len'], cw) - mc) / (sc + 1e-6)
        cached.append((z1, conses))
        print(f"[epoch164] extracted seed {s} ({i+1}/{len(args.seeds)})", flush=True)

    res = {}
    print(f"\n  {'configuration':22s}" + "".join(f"{p:>10s}" for _, p in COLS), flush=True)
    for branch in ('student', 'teacher'):
        per_seed = [evaluate_all_metrics(y, fuse('linear', z1, cs[branch])) for (z1, cs) in cached]
        row = {}
        for k, _ in COLS:
            v = [m[k] for m in per_seed if m.get(k) is not None]
            if v:
                row[k], row[k + '_std'] = float(np.mean(v)), float(np.std(v))
        label = f"{branch} consensus"
        res[label] = row
        print(f"  {label:22s}" + "".join(f"{row.get(k, float('nan')):10.4f}" for k, _ in COLS),
              flush=True)

    out = {'dataset': 'WADI', 'checkpoint_epoch': ck.get('epoch'), 'checkpoint_sha256': SNAPSHOT_SHA256,
           'teacher_sha256': TEACHER_SHA256, 'K_test': args.K_test, 'K_train': args.K_train,
           'seeds': args.seeds, 'results': res}
    with open(PROJECT_ROOT / args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {PROJECT_ROOT / args.out}", flush=True)


if __name__ == '__main__':
    main()

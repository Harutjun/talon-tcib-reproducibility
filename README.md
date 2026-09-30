# TALON / TCIB Reproducibility

Code accompanying "Teacher-Aligned Latent-Only Conditioning for Contextual Anomaly Detection
in Paired Input-Output Time Series" (ICASSP 2027 submission). Reproduces every number in
the paper's Table 1 (SWaT/WADI comparison). Trained checkpoints are not included; train from
the released hyperparameters below, then score.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires PyTorch with CUDA for practical training times. `requirements.txt` includes `TSB_AD`
(VUS-ROC/VUS-PR/Affiliation-F1) and `PATE` (PATE-F1); if either fails to resolve from PyPI,
install from source: https://github.com/TheDatumOrg/TSB-AD and
https://github.com/Raminghorbanii/PATE.

## 2. Get the data (not included — see §4)

Place SWaT and WADI under `data/` as described in §4, then everything below runs as shown.

## 3. Train

```bash
# Stage 1: teacher (VAE on Y alone)
python training/train_teacher.py --dataset SWaT --epochs 1000 --batch_size 4096 \
    --learning_rate 1e-4 --weight_decay 0.1 --alpha 1.0 --beta 0.033 --seed 42
python training/train_teacher.py --dataset WADI --epochs 1000 --batch_size 2048 \
    --learning_rate 1e-4 --weight_decay 0.1 --alpha 1.0 --beta 0.033 --seed 42

# Stage 2: student (aligned to the frozen teacher)
python training/train_student.py --dataset SWaT --vae_checkpoint <teacher_ckpt.pth> --seed 42
python training/train_student.py --dataset WADI --vae_checkpoint <teacher_ckpt.pth> --seed 42

# Baselines: Joint VAE and CVAE (per-dataset scripts; same architecture/capacity as TALON)
python training/train_joint_vae.py --dataset SWaT
python training/train_joint_vae.py --dataset WADI
python training/train_cvae_swat.py
python training/train_cvae_wadi.py
```

`--epochs` is a training-time budget, not a target: the script saves `best_vae.pth` whenever
validation loss improves and keeps running for the full budget regardless, so the checkpoint
actually used for the released numbers is from well before the budget's end — training that
far is unnecessary. 1000 epochs comfortably covers where both models saturate; watch
`best_val_loss` in the printed log and raise `--epochs` only if it is still improving when
the run ends.

Full per-run hyperparameters (architecture, window/patch size, optimizer, seed) are recorded
in `results/swat_cve/BestFull/config.json` and `results/wadi_vae/WADI/BestFullChannels/config.json`
— pass any field under `train_args` as the like-named CLI flag to match a released run exactly.
`configs/spatial_benchmark_config.py` holds the architecture/training defaults these scripts
fall back on when a flag isn't given.

| | SWaT | WADI |
|---|---|---|
| window / patch / stride | 100 / 10 / 10 (teacher), 5 (student) | 100 / 10 / 1 |
| latent dim / channel bandwidth | 26 / 26 | 60 / 10 |
| encoder / decoder hidden dim | 64 / 64 | 120 / 120 |
| transformer blocks (enc / dec) | 2 / 2 | 4 / 12 |
| batch size, epoch budget (teacher) | 4096, 1000 | 2048, 1000 |
| learning rate, weight decay | 1e-4, 0.1 | 1e-4, 0.1 |
| alpha, beta (recon / KL weight) | 1.0, 0.033 | 1.0, 0.033 |
| scaler | minmax | minmax |
| seed | 42 | 42 |

## 4. Score every Table 1 row

```bash
# TALON, Teacher, and random-student-control (all 3 in one run)
python reproducibility/final_table_rows.py --datasets SWaT WADI

# Table 1's "Teacher" row specifically (paper's own CNLL estimator, x_condition = 0)
python reproducibility/score_teacher_nullx.py --datasets SWaT WADI

# Joint VAE / CVAE
python reproducibility/evaluate_swat_joint_cvae.py
python reproducibility/evaluate_wadi_joint_cvae.py

# Random Classifier (10-draw uniform-score reference)
python reproducibility/trivial_baselines.py --datasets SWaT WADI

# Self-evaluated baselines (each trains its own model, then scores it)
python reproducibility/run_baseline_selfeval.py --dataset SWaT --model DAGMM
python reproducibility/run_baseline_selfeval.py --dataset SWaT --model OmniAnomaly
python reproducibility/run_baseline_selfeval.py --dataset SWaT --model MAD_GAN
python reproducibility/run_gdn_selfeval.py --dataset SWaT
python reproducibility/run_timesnet_selfeval.py --dataset SWaT
python reproducibility/run_usad_published.py --dataset both
```
(repeat the `--dataset SWaT` rows with `--dataset WADI`)

Scoring hyperparameters (Monte Carlo draws, seeds, OOD clamp, fusion weight) are fixed in
`reproducibility/final_table_rows.py`'s `SPEC` dict: `K_train=50`, `K_test=200`, 10 trained
seeds + 3 random-control seeds (`seed_i = 42 + 1337*i`), OOD clamp 1.0 (SWaT) / 10.0 (WADI),
consensus fusion weight `FUSION_LAMBDA = 1.0` on both datasets. Every script writes a JSON
under `results/` with the full seven-metric row plus the run metadata needed to trace it back.

## Dataset access (SWaT / WADI are not redistributed here)

SWaT and WADI are released by iTrust, Centre for Research in Cyber Security, Singapore
University of Technology and Design, under a data-use agreement that prohibits
redistribution. Request access directly from iTrust (search "iTrust SUTD SWaT WADI dataset
request"); this repository ships no CSVs, and `.gitignore` blocks committing any `data/` path
or `*.csv` file.

Place the files exactly as follows (paths and filenames are load-bearing — the loaders match
on them literally):

```
data/
├── SWaT/
│   ├── SWaT_Dataset_Normal_v1.csv      # SWaT.A1 & A2 (Dec 2015), normal operation
│   └── SWaT_Dataset_Attack_v0.csv      # same release, attack period
└── WaDi/
    ├── WADI.A2_19 Nov 2019/            # used by TALON/Teacher/CVAE/Joint VAE and every
    │   ├── WADI_14days_new.csv         # self-evaluated baseline except USAD's own reproduction
    │   └── WADI_attackdataLABLE.csv
    └── WADI.A1_9 Oct 2017/             # only needed for run_usad_published.py, which
        ├── WADI_14days.csv             # matches USAD's own published paper-config protocol
        ├── WADI_attackdata.csv         # on the WADI release USAD itself was evaluated on
        └── attack_description.xlsx
```

SWaT's CSVs are read with their original headers and an `Attack CSV`'s `Normal/Attack` label
column; WADI's are read with `WADI_14days_new.csv` skipping its 4-row spreadsheet header
(`load_csv_dataset` in `datasets/LocalTSAD.py` documents the exact column and label
conventions if a release ships under a different filename and needs a rename).

## License

This repository's own code is released under the BSD 3-Clause license — see `LICENSE`.

Several baseline model implementations under `reproducibility/baselines/` are vendored from
other projects under their own licenses, not this repository's: `tranad_bundle.py` (DAGMM,
OmniAnomaly, MAD-GAN, TranAD; BSD-3-Clause, © Shreshth Tuli), `gdn.py` (MIT, © d-ailin),
`timesnet.py` (MIT, © THUML @ Tsinghua University), and `usad_official/` (its own bundled
`LICENSE`, BSD, © EURECOM). Full license text for each, and the exact upstream source files
each was vendored from, is in `reproducibility/baselines/THIRD_PARTY_LICENSES.md`; every
vendored file's own header additionally documents what was changed relative to the original
(dead-code removal, dependency inlining, etc.) and why.

SWaT and WADI are third-party datasets under a separate iTrust data-use agreement (§ above);
this repository grants no rights to them and includes no dataset files.

## Citing this work

```bibtex
@inproceedings{magakyan2027talon,
  title     = {Teacher-Aligned Latent-Only Conditioning for Contextual Anomaly Detection
               in Paired Driving-Response Time Series},
  author    = {Magakyan, Harutjun and Shimkin, Nahum},
  booktitle = {ICASSP 2027 (submitted)},
  year      = {2027}
}
```

This is a submitted manuscript; update the venue/year fields above once the paper's
publication status is finalized.

# TALON / TCIB Reproducibility

Code accompanying "Teacher-Aligned Latent-Only Conditioning for Contextual Anomaly Detection
in Paired Driving-Response Time Series" (ICASSP 2027 submission). This repository is scoped to
the paper: it reproduces every number in Table 1 (the SWaT/WADI comparison) and nothing beyond
it — no thesis-only ablations, no exploratory/debug scripts, no unrelated datasets.

The theoretical derivations (TCIB decomposition, the variational bound, and the
Gaussian-process/Kronecker-precision construction) are in a separate companion note, referenced
from the paper and linked here once posted to arXiv.

## Model naming

- `models/TALONTeacher.py` — Stage 1: the response-only teacher, $q_\phi(z\mid y)$ /
  $p_\theta(y\mid z)$, trained with a standard VAE objective on nominal $Y$ alone.
- `models/TALONStudent.py` — Stage 2: the student $q_\psi(z\mid x)$, aligned to the frozen
  teacher's posterior and decoding through the frozen teacher decoder. Together, Teacher +
  Student at inference is TALON.
- `models/JointCVAE.py` — the paper's Joint VAE (models $p(X,Y)$ jointly) and CVAE (models
  $p(Y\mid X)$ with the decoder reading $X$ directly) baselines; both are instances of this
  class under different configurations.
- `models/TSPCVAE.py`, `models/Modules.py` — shared transformer/patch encoder-decoder blocks
  and the Kronecker/GP-prior machinery used by all three models above.

## What's here

- `reproducibility/final_table_rows.py` — the single entry point producing every TALON,
  Teacher, and random-student-control number in Table 1, for both SWaT and WADI, at the full
  seven-metric column coverage and seed count the paper reports.
- `reproducibility/evaluate_swat_joint_cvae.py` / `evaluate_wadi_joint_cvae.py` — Joint VAE's
  reported numbers (`compare_wadi_cvae_matched_estimator.py` documents why WADI's CVAE row uses
  a scoring estimator matched to TALON's rather than the joint model's own importance-weighted
  one).
- `reproducibility/evaluate_swat_bestfull.py` — the shared metric pipeline
  (`evaluate_all_metrics`: Point-F1, Precision, Recall, AUC-ROC, AUC-PR, VUS-ROC, VUS-PR,
  Affiliation-F1, PATE) used by every row in the table, TALON and baselines alike.
- `reproducibility/run_baseline_selfeval.py`, `run_gdn_selfeval.py`, `run_timesnet_selfeval.py`,
  `run_usad_published.py` — self-evaluation of DAGMM, OmniAnomaly, MAD-GAN, GDN, TimesNet, and
  USAD (USAD under its own paper's published training setup) on the identical SWaT/WADI data and
  metric pipeline as TALON.
- `reproducibility/baselines/` — vendored official-architecture implementations (DAGMM,
  OmniAnomaly, MAD-GAN, TranAD, GDN, TimesNet, and the full USAD official repository under
  `usad_official/`), sourced as noted in each file's header. TranAD, Anomaly-Transformer, and
  PatchAD are literature-sourced in the paper (not self-evaluated here), so their baseline code
  is not included.
- `reproducibility/nmc_sweep_pipeline.py`, `evaluate_swat_stage2_ablations.py`,
  `diagnose_random_student_control.py`, `rebuild_swat_ablation_bestfull.py`,
  `evaluate_wadi_epoch164_full.py`, `runtime_guard.py` — supporting pipeline code the scripts
  above depend on (the Monte-Carlo seed-count sensitivity check behind the paper's "sampling
  variance is negligible" claim, the trained/random-student control, shared scoring/aggregation
  helpers, and thread/GPU setup).
- `datasets/LocalTSAD.py` — the SWaT/WADI CSV loading, channel-split, and scaling pipeline.
- `results/swat_cve/BestFull/config.json`, `results/wadi_vae/WADI/BestFullChannels/config.json`
  — the exact preprocessing configuration (scaler, window size, channel split) used for every
  TALON/Teacher/CVAE/Joint VAE number in the paper.
- `results/usad_published/{SWaT,WADI}_seed42/` — USAD's saved scores, training history, and
  protocol record from its own-paper reproduction (metrics, not the trained checkpoint).
- `reproducibility/results/baseline_selfeval_*.json` — the self-evaluation results for
  DAGMM, OmniAnomaly, MAD-GAN, GDN, TimesNet on both datasets.

Trained TALON/CVAE/Joint VAE checkpoints are not included in this release.

**TODO / not yet verified**: this release has only been checked in-place (syntax-compiled and
import-tested against the original working copy's environment) after trimming and renaming. It
has NOT been verified end-to-end from a fresh clone: fresh `pip install -r requirements.txt`
into a clean environment, then actually running `run_baseline_selfeval.py` /
`run_gdn_selfeval.py` / `run_timesnet_selfeval.py` / `run_usad_published.py` against real
SWaT/WADI data to completion. Do this before pointing anyone external at this repo.

## Setup

```
pip install -r requirements.txt
```

Also requires the `TSB_AD` and `PATE` packages (used by `evaluate_all_metrics` for VUS-ROC/
VUS-PR/Affiliation-F1 and PATE respectively) and PyTorch with CUDA if available.

Datasets are not included. Point a script's `DATASETS` dict entries at your own SWaT/WADI CSVs
(same layout as the original dataset releases); `datasets/LocalTSAD.py` documents the expected
column and label conventions.

## Running a baseline reproduction

```
python reproducibility/run_baseline_selfeval.py --dataset SWaT --model OmniAnomaly
python reproducibility/run_usad_published.py --dataset SWaT
python reproducibility/run_gdn_selfeval.py --dataset WADI
```

Each script writes its own `reproducibility/results/baseline_selfeval_<dataset>_<model>.json`
(or `results/usad_published/<dataset>_seed<seed>/`) with the full metric set plus training
metadata (epochs trained, validation loss), so every number is traceable back to a specific run.

Reproducing TALON/CVAE/Joint VAE's own numbers additionally requires their trained checkpoints
(not included) at the paths `final_table_rows.SPEC` expects.

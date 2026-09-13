"""
========================================================================================
RUNTIME GUARD -- keep a scoring run inside this machine's resources
========================================================================================
Scoring runs repeatedly destabilised the development machine. The confirmed cause was the
PATE metric being called with n_jobs=-1: joblib on Windows *spawns* rather than forks, so
every worker re-imports the calling module and with it torch (~300-500 MB resident each).
On a 24-core box that is ~10 GB for a single metric call, and concurrent scoring jobs
multiplied it.

This module centralises the mitigations so no individual script has to remember them:

    pate_jobs()   worker count for PATE, sized from FREE memory rather than core count
    cap_threads() bound the BLAS/OMP thread pools, which otherwise open one thread per
                  core inside every one of those workers
    preflight()   refuse to start when memory is already tight, instead of taking the
                  machine down with it
    free_gpu()    drop cached CUDA blocks between seeds

Everything is overridable by environment variable so a larger machine is not held back:
TALON_PATE_JOBS, TALON_MIN_FREE_GB, TALON_THREADS.
========================================================================================
"""
import os
import sys

_GB = 2 ** 30
# Each spawned joblib worker re-imports torch; measured at roughly this much resident.
_WORKER_GB = 0.6
_DEFAULT_MAX_JOBS = 4


def _free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / _GB
    except Exception:
        return float('inf')


def pate_jobs():
    """Worker count for PATE: explicit override, else sized from free memory, capped at 4."""
    env = os.environ.get('TALON_PATE_JOBS')
    if env:
        return max(1, int(env))
    free = _free_gb()
    if free == float('inf'):
        return 2
    # Leave half of free memory as headroom for the parent process and the OS.
    return max(1, min(_DEFAULT_MAX_JOBS, int((free * 0.5) / _WORKER_GB)))


def cap_threads(n=None):
    """Bound BLAS/OMP pools. Must run before numpy/torch are imported to take full effect."""
    n = int(os.environ.get('TALON_THREADS', n or 4))
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


def preflight(min_free_gb=None, label=''):
    """Abort rather than start a run that the machine cannot finish."""
    need = float(os.environ.get('TALON_MIN_FREE_GB', min_free_gb or 8.0))
    free = _free_gb()
    if free < need:
        sys.stderr.write(
            f"[runtime_guard] REFUSING TO START{' ' + label if label else ''}: "
            f"{free:.1f} GB free, {need:.1f} GB required.\n"
            f"[runtime_guard] Close other work, or lower TALON_MIN_FREE_GB deliberately.\n")
        raise SystemExit(3)
    print(f"[runtime_guard] {free:.1f} GB free | PATE workers {pate_jobs()} | "
          f"threads {os.environ.get('OMP_NUM_THREADS', 'default')}", flush=True)


def free_gpu():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

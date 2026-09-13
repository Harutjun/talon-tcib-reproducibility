"""STATUS: EXPERIMENTAL -- USAD paper-config reproduction, SWaT then WADI.

Uses unmodified upstream training/testing at the commit in usad_official/.
Paper Table 7: median downsample 5, epochs 70, (window, latent)=(12,40)/(10,100).
Notebook supplies batch 7919, chronological 80/20 window split, no shuffle,
train-fitted MinMaxScaler and alpha=beta=.5. Paper does not specify these choices.
Not the notebook demo's 100 epochs / 1200 latent dimensions / native sampling.
Reports paper-style window-OR labels separately from endpoint point labels.
Does not modify paper tables or select checkpoints/hyperparameters on test scores.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TALON_PATE_JOBS', '1')
import sys
import argparse
import hashlib
import json
import time
from pathlib import Path
from runtime_guard import cap_threads, preflight
cap_threads(4)
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, auc

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'reproducibility/baselines/usad_official'
sys.path.insert(0, str(UPSTREAM))
import usad as official


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(name):
    notes = []
    if name == 'SWaT':
        base = ROOT / 'data/SWaT'
        paths = [base/'SWaT_Dataset_Normal_v1.csv', base/'SWaT_Dataset_Attack_v0.csv']
        train, test = [pd.read_csv(p, low_memory=False) for p in paths]
        labels = (test['Normal/Attack'].str.strip() != 'Normal').to_numpy(np.int8)
        train = train.drop(columns=['Timestamp', 'Normal/Attack'])
        test = test.drop(columns=['Timestamp', 'Normal/Attack'])
        expected_rows = 496800
    else:
        base = ROOT / 'data/WaDi/WADI.A1_9 Oct 2017'
        paths = [base/'WADI_14days.csv', base/'WADI_attackdata.csv', base/'attack_description.xlsx']
        train = pd.read_csv(paths[0], skiprows=4, low_memory=False)
        test = pd.read_csv(paths[1], low_memory=False)
        timestamps = pd.to_datetime(test.iloc[:,1].str.strip()+' '+test.iloc[:,2].str.strip(), format='%m/%d/%Y %I:%M:%S.%f %p')
        # Original spreadsheet has year/month typos; chronology and accompanying
        # table_WADI.pdf establish Oct 10 for attack 5, Oct 11 for attacks 11--15.
        # Attack 9 uses Oct 11 from spreadsheet (PDF erroneously repeats Oct 10).
        sheet = pd.read_excel(paths[2])
        intervals = []
        labels = np.zeros(len(test), dtype=np.int8)
        for row in sheet.itertuples(index=False, name=None):
            if not isinstance(row[0], (int, float)) or pd.isna(row[0]) or pd.isna(row[2]) or pd.isna(row[3]):
                continue
            date = pd.Timestamp(row[1]).replace(year=2017, month=10)
            start = date + pd.to_timedelta(str(row[2]).replace('.', ':'))
            end = date + pd.to_timedelta(str(row[3]).replace('.', ':'))
            labels[(timestamps >= start) & (timestamps <= end)] = 1
            intervals.append([str(start), str(end)])
        notes.append({'label_intervals_inclusive': intervals, 'label_source': 'A1 attack_description.xlsx; corrected malformed year/month as documented in source'})
        train.columns = [c.strip().split('\\')[-1] for c in train.columns]
        test.columns = [c.strip().split('\\')[-1] for c in test.columns]
        train = train.drop(columns=['Row','Date','Time'])
        test = test.drop(columns=['Row','Date','Time'])
        expected_rows = 1048571
    for df in (train, test):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].str.replace(',', '.', regex=False), errors='raise')
    dropped = train.columns[train.isna().all()].tolist()
    train = train.drop(columns=dropped)
    test = test[train.columns]
    missing = {'train': int(train.isna().sum().sum()), 'test': int(test.isna().sum().sum())}
    if any(missing.values()):
        notes.append('Missing-value policy unspecified upstream: forward fill within each split, then remaining leading NaNs = 0; all-empty training columns removed.')
    train = train.ffill().fillna(0)
    test = test.ffill().fillna(0)
    metadata = {'files': {str(p): sha256(p) for p in paths}, 'raw_train_rows': len(train), 'raw_test_rows': len(test),
                'paper_train_rows': expected_rows, 'raw_anomaly_fraction': float(labels.mean()),
                'dropped_all_nan_channels': dropped, 'missing_before_fill': missing, 'channels': list(train.columns), 'notes': notes}
    if len(train) != expected_rows:
        notes.append(f'Available training file has {len(train)} rows versus paper {expected_rows}; kept full file, no undocumented truncation.')
    # Fit on original nominal observations before binning, consistent with the
    # notebook's normalization stage; order relative to downsampling unspecified.
    scaler = MinMaxScaler().fit(train)
    x, y = scaler.transform(train), scaler.transform(test)
    def downsample(a):
        return pd.DataFrame(a).groupby(np.arange(len(a))//5, sort=False).median().to_numpy(np.float32)
    x, y = downsample(x), downsample(y)
    lab = pd.Series(labels).groupby(np.arange(len(labels))//5, sort=False).max().to_numpy(np.int8)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x.shape[1] == (51 if name == 'SWaT' else 123), x.shape
    return x, y, lab, scaler, metadata


def windows(a, k):
    # Preserve upstream notebook's N-K windows (it omits the final valid window).
    return np.lib.stride_tricks.sliding_window_view(a, k, axis=0)[:-1].transpose(0,2,1).copy().reshape(len(a)-k, -1)


def basic_metrics(labels, scores):
    p, r, thresholds = precision_recall_curve(labels, scores)
    f = 2*p*r / np.maximum(p+r, 1e-15)
    i = int(np.argmax(f[:-1]))
    return {'F1_best_test_threshold': float(f[i]), 'precision': float(p[i]), 'recall': float(r[i]),
            'threshold': float(thresholds[i]), 'AUC_ROC': float(roc_auc_score(labels, scores)),
            'AUC_PR_average_precision': float(average_precision_score(labels, scores)),
            'AUC_PR_trapezoid': float(auc(r,p))}


def run(name, args):
    preflight(label=f'USAD {name}')
    suffix = f'_epochs{args.epochs}' if args.epochs != 70 else ''
    out = ROOT/'results/usad_published'/f'{name}_seed{args.seed}{suffix}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'checkpoint.pth').exists() and not args.rescore:
        raise FileExistsError(f'{out}: checkpoint exists; use --rescore to evaluate it')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    k, latent = (12,40) if name == 'SWaT' else (10,100)
    start = time.perf_counter()
    x, y, labels, scaler, data_meta = load_data(name)
    xw, yw = windows(x,k), windows(y,k)
    split = int(np.floor(.8*len(xw)))
    device = official.device
    def loader(a):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(a)), batch_size=7919, shuffle=False, num_workers=0)
    train_loader, val_loader, test_loader = loader(xw[:split]), loader(xw[split:]), loader(yw)
    model = official.UsadModel(xw.shape[1], latent).to(device)
    protocol_choices = ['paper table 7 latent m used directly, not notebook hidden_size times window',
              'batch size 7919, alpha=beta=.5, no shuffle, 80/20 split from notebook (not specified for paper runs)',
              'fixed final epoch checkpoint; validation monitoring only, no test-driven checkpoint selection',
              'minmax fit on full nominal file before downsampling and 80/20 window split, no test fitting or clipping',
              'factor-five feature medians; labels max within each bin; last partial bin retained',
              '80/20 split after windowing reproduces notebook overlap across split',
              'unmodified upstream optimizer gradient-clearing semantics preserved',
              'current PyTorch/CUDA differs from paper PyTorch1.3.1/CUDA10; seed42 is a declared local choice']
    if args.epochs != 70:
        protocol_choices.append(f'DEVIATION: epochs={args.epochs}, not Paper Table 7\'s 70 -- a longer-training '
                                 'diagnostic run to check whether the gap to literature-sourced USAD figures '
                                 'narrows with more training, not a claim that this is the published setup')
    meta = {'dataset': name, 'seed': args.seed, 'window': k, 'latent': latent, 'epochs':args.epochs, 'batch_size':7919,
            'optimizer':'two Adam optimizers, upstream defaults lr=0.001', 'alpha':.5, 'beta':.5,
            'upstream_commit':(UPSTREAM/'UPSTREAM_COMMIT.txt').read_text().strip(),
            'torch': torch.__version__, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'parameters':sum(p.numel() for p in model.parameters()), 'downsample':5,
            'train_windows': split, 'val_windows':len(xw)-split, 'test_windows':len(yw), 'data':data_meta,
            'protocol_choices':protocol_choices}
    (out/'protocol.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(out/'scaler.npz', min_=scaler.min_, scale_=scaler.scale_, data_min_=scaler.data_min_, data_max_=scaler.data_max_)
    print(json.dumps({k:v for k,v in meta.items() if k not in ['data','protocol_choices']}), flush=True)
    print('Data notes:',data_meta['notes'], flush=True)
    if args.rescore:
        model.load_state_dict(torch.load(out/'checkpoint.pth', map_location=device, weights_only=True))
    else:
        epoch_time = time.perf_counter()
        history=[]
        def epoch_end(epoch,result):
            nonlocal epoch_time
            if not all(np.isfinite(v) for v in result.values()):
                raise FloatingPointError(f'Nonfinite validation at epoch {epoch+1}')
            now=time.perf_counter()
            history.append({'epoch':epoch+1,'seconds':now-epoch_time,**result})
            epoch_time=now
            (out/'history.json').write_text(json.dumps(history,indent=2))
            print(f'{name} epoch {epoch+1}/{args.epochs}: {result}, {history[-1]["seconds"]:.2f}s',flush=True)
        model.epoch_end=epoch_end
        training_start=time.perf_counter()
        official.training(args.epochs,model,train_loader,val_loader)
        meta['training_seconds']=time.perf_counter()-training_start
        torch.save(model.state_dict(),out/'checkpoint.pth')
    model.eval()
    scores=np.concatenate([r.detach().cpu().numpy() for r in official.testing(model,test_loader,alpha=.5,beta=.5)])
    assert np.isfinite(scores).all()
    window_labels=np.lib.stride_tricks.sliding_window_view(labels,k)[:-1].max(axis=1)
    endpoint_labels=labels[k-1:-1]
    assert len(scores)==len(window_labels)==len(endpoint_labels)
    np.savez_compressed(out/'scores.npz',scores=scores,window_labels=window_labels,endpoint_labels=endpoint_labels,
                        endpoint_downsampled_indices=np.arange(k-1,len(labels)-1))
    metrics={'paper_style_window_OR':basic_metrics(window_labels,scores),
             'supplementary_endpoint_point_labels':basic_metrics(endpoint_labels,scores),
             'paper_reference_without_PA_F1':.7917 if name=='SWaT' else .2328,
             'note':'Window-OR labels match notebook/paper window detection; endpoint labels are additional pointwise evaluation, not the original experiment.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    meta['total_seconds']=time.perf_counter()-start
    (out/'protocol.json').write_text(json.dumps(meta,indent=2))
    print(name,'COMPLETE',json.dumps(metrics),flush=True)
    del model, train_loader, val_loader, test_loader, xw, yw
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['SWaT','WADI','both'],default='both')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=70,help="Paper Table 7's value; override only for a disclosed longer-training diagnostic")
    parser.add_argument('--rescore',action='store_true')
    args=parser.parse_args()
    for name in (['SWaT','WADI'] if args.dataset=='both' else [args.dataset]):
        run(name,args)

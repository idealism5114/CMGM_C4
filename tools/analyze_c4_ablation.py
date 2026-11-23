"""Fixed-checkpoint paired date-block diagnostics; no training or reselection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def macro(error,mask):
    counts=mask.sum(0)
    if (counts==0).any():raise ValueError('No valid labels for a commodity in this segment')
    return float((np.where(mask,error,0.).sum(0)/counts).mean())


def analyze(root):
    root=Path(root)
    if not (root/'COMPLETE.json').exists():raise ValueError('Wait for all models and final frozen evaluation')
    p=read(root/'PAIRED_STATISTICS_PROTOCOL.json');registered=read(root/'PREREGISTERED_PROTOCOL.json')
    rng=np.random.default_rng(p['seed']);summary=[];periods=[];daily=[]
    for split in p['splits']:
        if split=='test' and not (root/'TEST_OPEN_AUTHORIZATION.json').exists():continue
        folder=(root/'test/full_c4') if split=='test' else root/'models/full_c4'
        with np.load(folder/('predictions.npz' if split=='test' else split+'_predictions.npz')) as a:
            dates=a['origin_dates'].copy();y=a['target'][:,1].astype('float64');mask=a['label_valid_mask'][:,1].copy()
            base=np.abs(a['prediction'][:,1].astype('float64')-y)
        n=len(dates);length=p['block_origins'];repeats=p['repeats'];counts=np.zeros((repeats,n),dtype='float64')
        starts=rng.integers(0,n-length+1,size=(repeats,int(np.ceil(n/length))))
        for b,s in enumerate(starts):
            indices=(s[:,None]+np.arange(length)[None]).ravel()[:n]
            counts[b]=np.bincount(indices,minlength=n)
        denominator=counts@mask.astype('float64')
        if (denominator==0).any():raise ValueError('An undefined bootstrap commodity mean')
        baseline=macro(base,mask)
        for key in registered['models']:
            folder=(root/'test'/key) if split=='test' else root/'models'/key
            with np.load(folder/('predictions.npz' if split=='test' else split+'_predictions.npz')) as a:
                np.testing.assert_array_equal(a['origin_dates'],dates)
                np.testing.assert_array_equal(a['label_valid_mask'][:,1],mask)
                np.testing.assert_array_equal(a['target'][:,1],y)
                err=np.abs(a['prediction'][:,1].astype('float64')-y)
            delta=macro(err,mask)-baseline
            sampled=(counts@np.where(mask,err-base,0.)/denominator).mean(1)
            low,high=np.quantile(sampled,[.025,.975])
            summary.append(dict(key=key,split=split,MAE5=macro(err,mask),delta_vs_Full=delta,percentile95_low=float(low),percentile95_high=float(high),
                fraction_resamples_delta_negative=float((sampled<0).mean()),block_origins=length,repeats=repeats))
            for i,idx in enumerate(np.array_split(np.arange(n),4)):
                d=macro(err[idx],mask[idx])-macro(base[idx],mask[idx])
                periods.append(dict(key=key,split=split,period=i+1,first_origin=dates[idx[0]],last_origin=dates[idx[-1]],origins=len(idx),
                    MAE5=macro(err[idx],mask[idx]),delta_vs_Full=d))
            for i,date in enumerate(dates):
                valid=mask[i]
                daily.append(dict(key=key,split=split,origin=date,valid_commodities=int(valid.sum()),
                    mean_absolute_error=float(err[i,valid].mean()) if valid.any() else '',
                    difference_vs_Full=float((err-base)[i,valid].mean()) if valid.any() else ''))
    csvwrite(root/'PAIRED_BOOTSTRAP_5D.csv',summary);csvwrite(root/'PERIOD_STABILITY_5D.csv',periods);csvwrite(root/'PER_ORIGIN_5D.csv',daily)
    note=dict(status='COMPLETE',models_fitted=0,checkpoints_or_selection_changed=False,protocol=p,
        warning='Fixed-checkpoint exploratory date uncertainty; no seed uncertainty, independent confirmation, or statistical significance claim. Segment means may differ from full-population macro weighting.')
    (root/'PAIRED_ANALYSIS_AUDIT.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
    print('Paired diagnostics complete',len(summary),'model/split rows')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);analyze(parser.parse_args().folder)

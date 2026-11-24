"""Independent completed-suite verification and fixed paired statistics. No fits."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cmgm_c4.common import read,save,sha,tensor_hash
from cmgm_c4.factory import build
from cmgm_c4.evaluation.metrics import population


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify(out):
    out=Path(out).resolve()
    if not (out/'COMPLETE.json').exists():raise ValueError('Wait for completed eight-model suite')
    p=read(out/'PREREGISTERED_PROTOCOL.json');final=read(out/'FINAL_AUDIT.json');lock=read(out/'MODELS_FROZEN.json')
    if final['status']!='PASS' or sha(out/'MODELS_FROZEN.json')!=final['models_frozen_sha256']:raise ValueError('Final VAL lock failed')
    if set(lock['models'])!=set(p['models']) or len(p['models'])!=8:raise ValueError('Incomplete or additional baseline models')
    if sha(ROOT/'initial_states/baselines.pt')!=p['initial_archive_sha256']:raise ValueError('Initialization archive changed')
    if sha(out/'run_entry_snapshot.py')!=p['runner_sha256']:raise ValueError('Launch source snapshot changed')
    for rel,h in p['source_hashes'].items():
        snapshot=out/'source_snapshot'/Path(rel).relative_to('src')
        if sha(snapshot)!=h or (rel.startswith('src/cmgm_c4/baselines/') and sha(ROOT/rel)!=h):
            raise ValueError('Source snapshot or baseline architecture changed '+rel)
    if sha(out/'project_config.json')!=p['project_config_sha256']:raise ValueError('Project config snapshot changed')
    for key,digest in p.get('profile_sha256',{}).items():
        if sha(out/'baseline_configs'/(key+'.json'))!=digest:raise ValueError('Baseline config snapshot changed '+key)
    initial=read(out/'INITIALIZATION_AND_GRADIENT_AUDIT.json')['models'];records={};checkpoints_before={}
    for key in p['models']:
        training=p['training_by_model'][key] if 'training_by_model' in p else p['fixed_training']
        steps_per_epoch=math.ceil(2726/training['batch_size'])
        folder=out/'models'/key;history=read(folder/'history.json');meta=read(folder/'models_frozen.json')
        selected=min(history,key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
        fresh=build('baselines/'+key,initial=True)
        if tensor_hash(fresh.state_dict())!=initial[key]['initial_state_sha256']:raise ValueError('Wrong untrained state '+key)
        checks={}
        for name in ('best','latest'):
            f=folder/(name+'.pt');checkpoints_before[str(f)]=sha(f)
            cp=torch.load(f,map_location='cpu',weights_only=False);fresh.load_state_dict(cp['model_state_dict'],strict=True)
            signature=cp['signature']
            if signature['protocol']!=training or signature['source_hashes']!=p['source_hashes']:raise ValueError('Training signature mismatch')
            if signature['initial_state_sha256']!=initial[key]['initial_state_sha256']:raise ValueError('Wrong initialization in checkpoint')
            if signature['dataset_sha256']!=p['dataset_sha256'] or signature['selector']!=p['selector']:raise ValueError('Data/selector mismatch')
            steps=[float(s['step']) for s in cp['optimizer']['state'].values()]
            optimizer_state_count=sum(s['exp_avg'].numel() for s in cp['optimizer']['state'].values())
            if optimizer_state_count!=initial[key]['gradient_connected_parameters']:raise ValueError('Optimizer does not cover every used parameter '+key)
            if not steps or any(step!=cp['epoch']*steps_per_epoch for step in steps):
                raise ValueError('Step count mismatch')
            if name=='best' and (cp['epoch']!=selected['epoch'] or sha(f)!=lock['models'][key]['checkpoint_sha256']):raise ValueError('Wrong best epoch/hash')
            if name=='latest' and cp['epoch']!=history[-1]['epoch']:raise ValueError('Latest epoch mismatch')
            checks[name]=dict(strict_restore=True,epoch=cp['epoch'],optimizer_steps=min(steps),optimizer_used_parameters=optimizer_state_count,checkpoint_sha256=sha(f))
        records[key]=checks
    # Numeric TEST remains unavailable to this tool until the existing suite's locked authorization is verified.
    splits=['train','val']
    if final['TEST_models']:
        auth=read(out/'TEST_OPEN_AUTHORIZATION.json')
        if not p['TEST_authorized'] or auth['models_frozen_sha256']!=sha(out/'MODELS_FROZEN.json'):raise ValueError('TEST authorization mismatch')
        splits.append('test')
    for split in splits:
        base_path=ROOT/p['reference_C4']/(split+'_predictions.npz') if split!='test' else ROOT/Path(p['reference_C4']).parents[1]/'test/full_c4/predictions.npz'
        with np.load(base_path) as reference:
            truth={k:reference[k].copy() for k in ['origin_dates','target','label_valid_mask']}
        for key in p['models']:
            folder=out/'models'/key if split!='test' else out/'test'/key
            arrays_path=folder/(split+'_predictions.npz' if split!='test' else 'predictions.npz')
            score=read(folder/'train_val_metrics.json')[split] if split!='test' else read(folder/'metrics.json')
            with np.load(arrays_path) as a:
                for k,v in truth.items():np.testing.assert_array_equal(a[k],v)
                if population(a['prediction'],a['target'],a['label_valid_mask'])!=score:raise ValueError('Metric recomputation mismatch')
                if not np.isfinite(a['prediction']).all():raise ValueError('Nonfinite prediction')
    if any(sha(Path(f))!=h for f,h in checkpoints_before.items()):raise ValueError('Verification modified checkpoint')
    result=dict(status='PASS',models=8,checks=records,metrics_recomputed_exactly=True,strict_best_latest=True,
        source_snapshots_match=True,untrained_initial_archives_verified=True,same_targets_masks_dates_as_C4=True,
        checkpoint_files_unchanged=True,training_or_selection_performed=False,splits=splits)
    save(out/'INDEPENDENT_VERIFICATION.json',result)
    print('INDEPENDENT VERIFICATION PASS',flush=True)
    return result


def macro(error,mask):
    n=mask.sum(0)
    if (n==0).any():raise ValueError('Empty commodity in population/segment')
    return float((np.where(mask,error,0.).sum(0)/n).mean())


def paired_statistics(out):
    out=Path(out);p=read(out/'PREREGISTERED_PROTOCOL.json');stats=read(out/'PAIRED_STATISTICS_PROTOCOL.json')
    if read(out/'INDEPENDENT_VERIFICATION.json')['status']!='PASS':raise ValueError('Independent verification required')
    paired=[];periods=[];rng=np.random.default_rng(stats['seed'])
    for split in stats['splits']:
        if split=='test' and not read(out/'FINAL_AUDIT.json')['TEST_models']:continue
        refpath=ROOT/p['reference_C4']/(split+'_predictions.npz') if split!='test' else ROOT/Path(p['reference_C4']).parents[1]/'test/full_c4/predictions.npz'
        with np.load(refpath) as a:
            y=a['target'][:,1].astype('float64');mask=a['label_valid_mask'][:,1];dates=a['origin_dates'];base=np.abs(a['prediction'][:,1].astype('float64')-y)
        n=len(dates);length=stats['block_origins'];repeats=stats['repeats']
        counts=np.zeros((repeats,n),dtype='float64')
        starts=rng.integers(0,n-length+1,size=(repeats,int(np.ceil(n/length))))
        for i,s in enumerate(starts):counts[i]=np.bincount((s[:,None]+np.arange(length)[None]).ravel()[:n],minlength=n)
        denom=counts@mask.astype('float64')
        if (denom==0).any():raise ValueError('Undefined bootstrap commodity mean')
        for key in p['models']:
            path=out/'models'/key/(split+'_predictions.npz') if split!='test' else out/'test'/key/'predictions.npz'
            with np.load(path) as a:
                np.testing.assert_array_equal(a['origin_dates'],dates)
                err=np.abs(a['prediction'][:,1].astype('float64')-y)
            sample=(counts@np.where(mask,err-base,0.)/denom).mean(1)
            lo,hi=np.quantile(sample,[.025,.975])
            paired.append(dict(key=key,split=split,MAE5=macro(err,mask),delta_vs_C4=macro(err,mask)-macro(base,mask),
                descriptive95_low=float(lo),descriptive95_high=float(hi),block_origins=length,repeats=repeats))
            for j,indices in enumerate(np.array_split(np.arange(n),4)):
                periods.append(dict(key=key,split=split,period=j+1,first_origin=dates[indices[0]],last_origin=dates[indices[-1]],
                    MAE5=macro(err[indices],mask[indices]),delta_vs_C4=macro(err[indices],mask[indices])-macro(base[indices],mask[indices])))
    write_csv(out/'PAIRED_BOOTSTRAP_5D.csv',paired);write_csv(out/'PERIOD_STABILITY_5D.csv',periods)
    save(out/'PAIRED_ANALYSIS_AUDIT.json',dict(status='COMPLETE',protocol=stats,models_fitted=0,checkpoints_or_selection_changed=False,
        caveat='Exploratory fixed-checkpoint time uncertainty, single seed, reused VAL/TEST, no independent confirmation or multiple-comparison adjustment'))
    print('PAIRED STATISTICS COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    torch.set_num_threads(2)
    verify(args.folder)
    if (args.folder/'PAIRED_STATISTICS_PROTOCOL.json').exists():
        paired_statistics(args.folder)

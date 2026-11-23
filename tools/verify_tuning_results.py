"""Independent post-run recomputation; requires completed VAL lock and run."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cmgm_c4.common import read,save,sha,tensor_hash
from cmgm_c4.evaluation.metrics import population
from cmgm_c4.factory import build


def verify(out):
    out=Path(out).resolve()
    if not (out/'COMPLETE.json').exists():raise ValueError('No completed search; do not open TEST')
    protocol=read(out/'PREREGISTERED_PROTOCOL.json');lock=read(out/'SELECTED_MODEL.json');final=read(out/'FINAL_AUDIT.json')
    if final['status']!='PASS' or sha(out/'SELECTED_MODEL.json')!=final['selection_lock_sha256']:raise ValueError('Search/selection lock failed')
    rows=lock['grid_results'];records=[];reference={};initial_hashes=[]
    for split in ('train','val'):
        data=ROOT/'data/train_val';dates=np.load(data/'reference_dates.npy');origins=np.load(data/(split+'_origin_indices.npy'))
        reference[split]=dict(target=np.load(data/(split+'_targets.npy')),label_valid_mask=np.load(data/(split+'_label_valid_mask.npy')),origin_dates=dates[origins])
    for rel,h in protocol['source_hashes'].items():
        if sha(out/'source_snapshot'/Path(rel).relative_to('src'))!=h:raise ValueError('Source snapshot mismatch')
    for row in rows:
        trial=row['trial'];folder=out/'trials'/trial;history=read(folder/'history.json');scores=read(folder/'train_val_metrics.json')
        checks={}
        for name in ('best','latest'):
            cp=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)
            model=build('c4/full');model.load_state_dict(cp['model_state_dict'],strict=True)
            if sum(p.numel() for p in model.parameters())!=567169:raise ValueError('Architecture changed')
            if cp['signature']['source_hashes']!=protocol['source_hashes']:raise ValueError('Code mismatch')
            if cp['signature']['protocol']['lr']!=row['lr'] or cp['signature']['protocol']['weight_decay']!=row['weight_decay']:raise ValueError('Optimizer settings mismatch')
            initial_hashes.append(cp['signature']['initial_state_sha256'])
            steps=[float(s['step']) for s in cp['optimizer']['state'].values()]
            if not steps or min(steps)!=cp['epoch']*43 or max(steps)!=cp['epoch']*43:raise ValueError('Actual optimizer step coverage failed')
            checks[name]=dict(epoch=cp['epoch'],optimizer_steps=min(steps),state_hash=tensor_hash(cp['model_state_dict']),checkpoint_sha256=sha(folder/(name+'.pt')))
        if checks['best']['epoch']!=min(history,key=lambda x:x['VAL_5D_COMMODITY_MACRO_MAE'])['epoch']:raise ValueError('Best epoch mismatch')
        if checks['latest']['epoch']!=history[-1]['epoch']:raise ValueError('Latest epoch mismatch')
        for split in ('train','val'):
            with np.load(folder/(split+'_predictions.npz')) as arrays:
                recomputed=population(arrays['prediction'],arrays['target'],arrays['label_valid_mask'])
                if recomputed!=scores[split]:raise ValueError('Saved metric does not reproduce exactly')
                if split not in reference:reference[split]={k:arrays[k].copy() for k in ('target','label_valid_mask','origin_dates')}
                else:
                    for k,v in reference[split].items():np.testing.assert_array_equal(v,arrays[k])
        records.append(dict(trial=trial,strict_best_latest=True,checks=checks,metrics_recomputed_exactly=True))
    if len(set(initial_hashes))!=1:raise ValueError('Initial states differ')
    # Reproduce the default-control trajectory and compare actual trained tensors.
    oldcp=torch.load(ROOT/'checkpoints/c4/full/best.pt',map_location='cpu',weights_only=False)
    newcp=torch.load(out/'trials/T00/best.pt',map_location='cpu',weights_only=False)
    h0=read(ROOT/'results/c4/full/history.json');h1=read(out/'trials/T00/history.json')
    control=dict(old_best_epoch=oldcp['epoch'],new_best_epoch=newcp['epoch'],
        trained_state_identical=tensor_hash(oldcp['model_state_dict'])==tensor_hash(newcp['model_state_dict']),
        old_epochs=len(h0),new_epochs=len(h1),
        paired_history_max_VAL5_difference=max(abs(a['VAL_5D_COMMODITY_MACRO_MAE']-b['VAL_5D_COMMODITY_MACRO_MAE']) for a,b in zip(h0,h1)))
    test_records={}
    if final['TEST_models']:
        auth=read(out/'TEST_OPEN_AUTHORIZATION.json')
        if auth['selection_lock_sha256']!=sha(out/'SELECTED_MODEL.json'):raise ValueError('TEST selection lock failed')
        data=ROOT/'data/test_release';dates=np.load(data/'reference_dates.npy');origins=np.load(data/'test_origin_indices.npy')
        test_reference=dict(target=np.load(data/'test_targets.npy'),label_valid_mask=np.load(data/'test_label_valid_mask.npy'),origin_dates=dates[origins])
        for trial in final['TEST_models']:
            with np.load(out/'test'/trial/'predictions.npz') as arrays:
                scores=population(arrays['prediction'],arrays['target'],arrays['label_valid_mask'])
                if scores!=read(out/'test'/trial/'metrics.json'):raise ValueError('TEST metric mismatch')
                if test_reference is None:test_reference={k:arrays[k].copy() for k in ('target','label_valid_mask','origin_dates')}
                else:
                    for k,v in test_reference.items():np.testing.assert_array_equal(v,arrays[k])
                test_records[trial]=dict(primary=scores['primary'],origins=len(arrays['prediction']),finite_prediction=bool(np.isfinite(arrays['prediction']).all()),metrics_recomputed_exactly=True)
        if set(test_records)!={lock['winner'],'T00'}:raise ValueError('Unexpected TEST candidates')
        with np.load(out/'test/T00/predictions.npz') as current,np.load(ROOT/'results/c4/full/test_predictions.npz') as old_test:
            same=all(np.array_equal(current[k],old_test[k],equal_nan=True) if np.issubdtype(current[k].dtype,np.number) else np.array_equal(current[k],old_test[k]) for k in ('target','label_valid_mask','origin_dates'))
            control['historical_TEST_population_identical']=same
            control['historical_TEST_prediction_max_abs_difference']=float(np.max(np.abs(current['prediction']-old_test['prediction']))) if same else None
    result=dict(status='PASS',trials=len(rows),source_snapshots_match=True,common_initial_state_hash=initial_hashes[0],
        records=records,default_control_reproduction=control,TEST=test_records,TEST_after_VAL_lock=True,
        fitting_performed=False,selection_changed=False)
    save(out/'INDEPENDENT_VERIFICATION.json',result)
    print('INDEPENDENT VERIFICATION PASS',control,test_records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);verify(parser.parse_args().folder)

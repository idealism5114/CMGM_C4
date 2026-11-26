"""Independent read-only audit of a completed tuned C4 ablation suite."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cmgm_c4.common import read,save,sha,tensor_hash
from cmgm_c4.factory import build
from cmgm_c4.evaluation.metrics import population


def verify(out):
    out=Path(out).resolve()
    if (out/'PROTOCOL.json').is_file() and read(out/'PROTOCOL.json').get('id') in ('C4_PROGRESSIVE_V1','C4_BRANCH_V2','C4_COMPONENT_V2'):
        from cmgm_c4.training.progressive import verify as verify_progressive
        return verify_progressive(out)
    if not (out/'COMPLETE.json').exists():raise ValueError('Wait for all13 completion before reading results')
    p=read(out/'PREREGISTERED_PROTOCOL.json');frozen=read(out/'MODELS_FROZEN.json');final=read(out/'FINAL_AUDIT.json');init=read(out/'INITIALIZATION_AUDIT.json')
    if final['status']!='PASS' or sha(out/'MODELS_FROZEN.json')!=final['models_frozen_sha256']:raise ValueError('Freeze audit failed')
    for rel,h in p['source_hashes'].items():
        if sha(out/'source_snapshot'/Path(rel).relative_to('src'))!=h:raise ValueError('Source snapshot changed')
    reference={}
    splits=['train','val']+(['test'] if final['TEST_models'] else [])
    if 'test' in splits:
        if read(out/'TEST_OPEN_AUTHORIZATION.json')['models_frozen_sha256']!=sha(out/'MODELS_FROZEN.json'):raise ValueError('TEST lock invalid')
    for split in splits:
        folder=ROOT/('data/test_release' if split=='test' else 'data/train_val')
        dates=np.load(folder/'reference_dates.npy');origins=np.load(folder/(split+'_origin_indices.npy'))
        reference[split]=dict(target=np.load(folder/(split+'_targets.npy')),label_valid_mask=np.load(folder/(split+'_label_valid_mask.npy')),origin_dates=dates[origins])
    records={};full_predictions={};full_scores={};differences={}
    for key in p['models']:
        folder=out/'models'/key;history=read(folder/'history.json');row=frozen['models'][key];scores=read(folder/'train_val_metrics.json');checks={}
        if [r['epoch'] for r in history]!=list(range(1,len(history)+1)):raise ValueError('Missing/repeated epoch')
        if any(r['optimizer_steps']!=43 or r['train_origins_used']!=2726 for r in history):raise ValueError('Coverage mismatch')
        for which in ('best','latest'):
            path=folder/(which+'.pt');cp=torch.load(path,map_location='cpu',weights_only=False)
            m=build('ablations/'+key);m.load_state_dict(cp['model_state_dict'],strict=True)
            if cp['signature']['protocol']!=p['fixed_training'] or cp['signature']['source_hashes']!=p['source_hashes']:raise ValueError('Checkpoint protocol/source mismatch')
            if cp['signature']['initial_state_sha256']!=init['records'][key]['initial_state_sha256']:raise ValueError('Initialization mismatch')
            steps=[float(v['step']) for v in cp['optimizer']['state'].values()]
            if not steps or min(steps)!=cp['epoch']*43 or max(steps)!=cp['epoch']*43:raise ValueError('Optimizer step count mismatch')
            grouped={i for g in cp['optimizer']['param_groups'] for i in g['params']}
            if set(cp['optimizer']['state'])!=grouped:raise ValueError('Optimizer contains tensors with no update state')
            if which=='best' and (cp['epoch']!=min(history,key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])['epoch'] or sha(path)!=row['checkpoint_sha256']):raise ValueError('Wrong best checkpoint')
            if which=='latest' and cp['epoch']!=history[-1]['epoch']:raise ValueError('Latest epoch mismatch')
            checks[which]=dict(epoch=cp['epoch'],optimizer_steps=min(steps),strict_restore=True,checkpoint_sha256=sha(path))
        for split in splits:
            path=(out/'test'/key/'predictions.npz') if split=='test' else folder/(split+'_predictions.npz')
            expected=read(out/'test'/key/'metrics.json') if split=='test' else scores[split]
            with np.load(path) as a:
                for name,values in reference[split].items():np.testing.assert_array_equal(a[name],values)
                measured=population(a['prediction'],a['target'],a['label_valid_mask'])
                if measured!=expected:raise ValueError('Metric reproduction failed '+key+' '+split)
                if not np.isfinite(a['prediction']).all():raise ValueError('Nonfinite predictions')
                if key=='full_c4':full_predictions[split]=a['prediction'].copy();full_scores[split]=measured['primary']
                elif split in ('val','test'):
                    truth=a['target'][:,1].astype('float64')
                    d=np.abs(a['prediction'][:,1].astype('float64')-truth)-np.abs(full_predictions[split][:,1].astype('float64')-truth)
                    valid=a['label_valid_mask'][:,1]
                    per=[float(d[:,c][valid[:,c]].mean()) for c in range(d.shape[1])]
                    if abs(float(np.mean(per))-(measured['primary']-full_scores[split]))>1e-14:raise ValueError('Commodity macro difference weighting failed')
                    differences.setdefault(key,{})[split]=dict(commodities_better_than_Full=sum(v<0 for v in per),commodities_worse_than_Full=sum(v>0 for v in per),macro_delta=float(np.mean(per)))
        records[key]=dict(strict_checkpoints=checks,metrics_exactly_recomputed=True,dates_targets_masks_match_frozen_data=True)
    old_root=ROOT/p['reference'];oldcp=torch.load(old_root/p['reference_best_checkpoint'],map_location='cpu',weights_only=False);newcp=torch.load(out/'models/full_c4/best.pt',map_location='cpu',weights_only=False)
    reproduction=dict(weights_identical=tensor_hash(oldcp['model_state_dict'])==tensor_hash(newcp['model_state_dict']),old_best_epoch=oldcp['epoch'],new_best_epoch=newcp['epoch'])
    for split in ('train','val','test'):
        if split not in full_predictions:continue
        old=(old_root/'test/T08/predictions.npz') if split=='test' else old_root/'trials/T08'/(split+'_predictions.npz')
        with np.load(old) as a:
            np.testing.assert_allclose(a['prediction'],full_predictions[split],atol=1e-6,rtol=1e-5)
            reproduction[split+'_prediction_max_difference']=float(np.abs(a['prediction']-full_predictions[split]).max())
    result=dict(status='PASS',models=13,checks=records,Full_reproduction=reproduction,per_commodity_changes=differences,commodity_difference_dtype='float64',
        source_snapshots_match=True,selection_unchanged=True,TEST_only_after_all_VAL_checkpoints_frozen=True,model_training_or_updates=False)
    save(out/'INDEPENDENT_VERIFICATION.json',result);print('INDEPENDENT ABLATION VERIFICATION PASS',reproduction)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);verify(p.parse_args().folder)

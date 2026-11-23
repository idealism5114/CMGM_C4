"""Sep17 D0B mechanisms + imported TriWeave: preflight, fixed suite, frozen TEST.

Default never trains. --run is explicitly authorized for this user's new suite.
No reuse of Legacy-data, Candidate MoE or changed TriWeave NoNorm checkpoints.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cmgm.models.formal_d0b_main_ablation import NAMES, KEY, DEFINITIONS, HISTORICAL_COMMIT, MainInnovationAblation

ROOT = Path(__file__).resolve().parents[2]
IMPORT = ROOT/'Data/triweave_imports/20261022_151203_692701'
DATA_REL = Path('data/derived/triweave_23_after_close_research_v1')
INPUT_KEYS = ('X', 'input_valid_mask', 'feature_valid_mask', 'feature_applicable_mask')
PROTOCOL = dict(id='CMGM_SEP17_TRIWEAVE23_MASKED_V1', historical_commit=HISTORICAL_COMMIT,
    full='switching_latent_balanced_readout', configurations=list(NAMES), seed=42,
    input_nodes=311, features=26, targets=23, window=20, horizons=[1,5,10,20],
    target_unit='decimal_return', label_formula='P(t+h)/P(t)-1; original Research23 mask, no label redefinition',
    train=dict(optimizer='Adam', lr=1e-4, weight_decay=1e-5, batch_size=64, shuffle=False,
        drop_last=False, epochs=200, patience=10, scheduler_factor=.5, scheduler_patience=5,
        prediction='SUM_h valid origin*commodity Huber means; delta=.02; fail empty horizon',
        Switch_KL='native beta_max=.0005/warmup20 once; no temporal and uniform routing zero'),
    selection='prediction-only SUM_h full VAL valid-label Huber totals/counts; same for scheduler/best/earlystop',
    differences_from_20260917='New economic-type 26 features, 311/23 population, availability masks; TRAIN tail retained and VAL global-valid-label aggregation repair Phase A issues. Model mechanisms from 2026-09-17; NOT numerical reproduction of old data/protocol',
    epoch0='Not a checkpoint candidate: retain historical epochs1..200 selection',
    test='Evaluate ALL fixed 12 best VAL-selected checkpoints once, after full-suite freeze; no TEST selection',
    primary_test='pooled TEST MAE@5d; also commodity macro and valid counts explicitly named',
    status='EXPLORATORY; historical TEST has been viewed; no historical PIT/roll certification')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    t.replace(p)


def checkpoint(p,x):
    t=p.with_name(p.name+'.tmp');torch.save(x,t);t.replace(p)


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def state_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])


def export_history(destination):
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',HISTORICAL_COMMIT,'cmgm'],cwd=ROOT,text=True).splitlines()
    records={}
    for name in files:
        if not name.endswith('.py'):continue
        original=subprocess.check_output(['git','show',HISTORICAL_COMMIT+':'+name],cwd=ROOT)
        text=original.decode()
        # Namespace-only rewriting. Mathematical source remains otherwise exact.
        text=re.sub(r'(?m)(from |import )cmgm(?=[.\s])',r'\1cmgm_sep17_frozen',text)
        dest=destination/'cmgm_sep17_frozen'/Path(name).relative_to('cmgm')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        records[name]=dict(original_sha256=hashlib.sha256(original).hexdigest(),namespaced_sha256=sha(dest))
    save(destination/'source_manifest.json',records)
    return records


class PreparedDataset(Dataset):
    def __init__(self,path,split):
        if split not in ('train','val','test'):raise ValueError(split)
        self.path=Path(path);self.split=split
        self.a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in self.path.glob('*.npy')}
        if split=='test' and not json.loads((self.path/'dataset_manifest.json').read_text()).get('audit_pass'):
            raise ValueError('TEST release gate failed')
        if f'{split}_origin_indices' not in self.a:raise ValueError('Requested split is absent')

    def __len__(self):return len(self.a[self.split+'_origin_indices'])

    def __getitem__(self,k):
        a=self.a;s=self.split;idx=a[s+'_window_indices'][k];pad=idx<0
        def window(name):
            value=np.array(a[name][np.maximum(idx,0)],copy=True);value[pad]=0;return value
        origin=int(a[s+'_origin_indices'][k])
        return dict(X=window('features'),input_valid_mask=window('input_valid_mask'),
            feature_valid_mask=window('feature_valid_mask'),feature_applicable_mask=np.array(a['feature_applicable_mask'],copy=True),
            Y=np.array(a[s+'_targets'][k],copy=True),label_valid_mask=np.array(a[s+'_label_valid_mask'][k],copy=True),
            origin_date=str(a['reference_dates'][origin]),origin_index=origin)


def loaders(path,split):
    return DataLoader(PreparedDataset(path,split),batch_size=64,shuffle=False,drop_last=False,num_workers=0)


def moved(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def prediction_loss(p,y,mask):
    losses=[]
    for h in range(4):
        valid=mask[:,h]
        if not valid.any():raise ValueError('Empty horizon in batch; no silent zero objective')
        losses.append(torch.nn.functional.huber_loss(p[:,h][valid],y[:,h][valid],delta=.02))
    return sum(losses)


def metrics(p,y,m,targets):
    result={};per=[]
    for hi,h in enumerate((1,5,10,20)):
        def calc(a,b):
            e=a.astype(np.float64)-b.astype(np.float64)
            if not len(e):return dict(status='NOT_EVALUABLE',valid_labels=0,MAE=None,MSE=None,RMSE=None,Hit=None,Huber=None)
            ab=np.abs(e);mse=float(np.mean(e**2))
            return dict(status='EVALUABLE',valid_labels=len(e),MAE=float(ab.mean()),MSE=mse,RMSE=float(np.sqrt(mse)),
                Hit=float((np.sign(a)==np.sign(b)).mean()),Huber=float(np.where(ab<=.02,.5*e**2,.02*(ab-.01)).mean()))
        result[str(h)]=calc(p[:,hi][m[:,hi]],y[:,hi][m[:,hi]])
        vals=[]
        for c,target in enumerate(targets):
            valid=m[:,hi,c];v=calc(p[:,hi,c][valid],y[:,hi,c][valid]);per.append(dict(horizon=h,commodity=target,**v))
            if v['valid_labels']:vals.append(v['MAE'])
        result[str(h)]['commodity_macro_MAE']=float(np.mean(vals)) if vals else None
        result[str(h)]['evaluable_commodities']=len(vals)
    return dict(horizons=result,per_commodity=per,selection=sum(result[str(h)]['Huber'] for h in (1,5,10,20)),
                Hit_unit='fraction; display x100 percent',error_unit='decimal cumulative return')


@torch.inference_mode()
def evaluate(model,loader,device,targets,return_predictions=False):
    model.eval();predictions=[];ys=[];masks=[];dates=[];regime=[];long=[];micro=[];disagreement=[]
    for batch in loader:
        b=moved(batch,device);p=model(**{k:b[k] for k in INPUT_KEYS})
        if p.shape!=b['Y'].shape or not torch.isfinite(p).all():raise ValueError('Evaluation shape/finite error')
        predictions.append(p.cpu().numpy());ys.append(batch['Y'].numpy());masks.append(batch['label_valid_mask'].numpy());dates+=batch['origin_date']
        d=model.diagnostics
        if 'regime_p' in d:
            support=d['temporal_valid'];regime.append(d['regime_p'][support].cpu().numpy())
            long.append(d['long_norm'].cpu().numpy());micro.append(d['micro_norm'].cpu().numpy())
            c=d['candidates'];disagreement.append(torch.stack([(c[:,:,i]-c[:,:,j]).abs().mean(-1)[support] for i,j in ((0,1),(0,2),(1,2))],-1).cpu().numpy())
    p,y,m=np.concatenate(predictions),np.concatenate(ys),np.concatenate(masks)
    if len(p)!=len(loader.dataset) or len(set(dates))!=len(dates):raise ValueError('Origin coverage/order failure')
    r=metrics(p,y,m,targets);r.update(origins=len(p),first_origin=dates[0],last_origin=dates[-1],drop_last=False)
    if regime:
        q=np.concatenate(regime).astype(float);hard=np.bincount(q.argmax(-1),minlength=3)/len(q)
        r['mechanisms']=dict(mean_p=q.mean(0).tolist(),entropy=float(-(q*np.log(q+1e-8)).sum(-1).mean()),
            hard_occupancy=hard.tolist(),long_norm=float(np.concatenate(long).mean()),micro_norm=float(np.concatenate(micro).mean()),
            candidate_pairwise_disagreement=float(np.concatenate(disagreement).mean()),
            uniform_occupancy_note='Uniform argmax [1,0,0] is deterministic tie breaking, not learned collapse')
    return (r,dict(prediction=p,target=y,label_valid_mask=m,origin_dates=np.array(dates),commodity_order=np.array(targets))) if return_predictions else r


def model_for(name,out,device):
    seed_all();return MainInnovationAblation(name,out/'historical_source').to(device)


def preflight(out,data_path):
    from cmgm.models.formal_d0b_main_ablation import native_modules
    native_modules(out/'historical_source')
    ref_module=importlib.import_module('cmgm_sep17_frozen.models.formal_d0b_main_ablation')
    fake=dict(n_nodes=28,market_indices=dict(stock=(0,2),bond=(2,4),commodity=(4,28)))
    torch.set_num_threads(2);rows={}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(912);x=torch.randn(2,20,28,21)
    valid=torch.ones(x.shape[:3],dtype=torch.bool);fm=torch.ones_like(x,dtype=torch.bool);fa=torch.ones(28,21,dtype=torch.bool)
    for name in NAMES:
        seed_all();reference=ref_module.MainInnovationAblation(name,fake).eval()
        seed_all();model=MainInnovationAblation(name,out/'historical_source',nodes=28,stocks=2,bonds=2,targets=24,features=21).eval()
        shared=[]
        for k,v in reference.state_dict().items():
            key=k if k.startswith('ordinary_mixhop') else 'native.'+k
            if key in model.state_dict():
                if not torch.equal(v,model.state_dict()[key]):raise ValueError('Shared init mismatch '+name+' '+key)
                shared.append(k)
        with torch.no_grad():
            old=reference(x);new=model(x,valid,fm,fa)
            diff=float((old-new).abs().max());torch.testing.assert_close(new,old,atol=2e-6,rtol=2e-5)
            permutation=model(x.flip(0),valid,fm,fa).flip(0)
            torch.testing.assert_close(new,permutation,atol=2e-6,rtol=2e-5)
            dirty=x.clone();iv=valid.clone();fv=fm.clone();iv[:,:,0]=False;fv[:,:,0]=False
            a=model(dirty,iv,fv,fa);dirty[:,:,0]=float('nan');b=model(dirty,iv,fv,fa)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            empty=model(dirty,torch.zeros_like(iv),torch.zeros_like(fv),fa)
            if not torch.isfinite(empty).all():raise ValueError('Empty source NaN')
        model.train();model.set_epoch(20);p=model(x,valid,fm,fa)
        loss=(p-.03).square().mean()+model.switch_loss();loss.backward()
        if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):raise ValueError('Nonfinite synthetic gradients')
        active=sum(v.numel() for v in model.parameters() if v.grad is not None)
        rows[name]=dict(all_valid_native_max_abs_diff=diff,shared_tensors=len(shared),shared_initial_max_abs_diff=0.,mismatch_count=0,
            synthetic_shape=list(p.shape),finite_gradients=True,unavailable_source_invariance=True,empty_support_finite=True,synthetic_active_parameters=active,PASS=True)
        print('[Sep17 native parity]',name,'diff',diff,'PASS',flush=True)
    # Actual data shapes/one TRAIN forward only, never choose from TEST here.
    data=json.loads((data_path/'dataset_manifest.json').read_text())
    for n,e in data['files'].items():
        if sha(data_path/n)!=e['sha256']:raise ValueError('Imported data hash mismatch '+n)
    t=next(iter(loaders(data_path,'train')))
    t={k:v[:1] if torch.is_tensor(v) else v[:1] for k,v in t.items()}
    actual_initial={};seed_all();full=MainInnovationAblation('Full D0B',out/'historical_source')
    full_state=full.native.state_dict()
    for name in NAMES:
        m=model_for(name,out,'cpu').eval();diffs=[k for k,v in m.native.state_dict().items() if not torch.equal(v,full_state[k])]
        if diffs:raise ValueError('Actual shared init mismatch '+repr(diffs))
        with torch.no_grad():
            p=m(**{k:t[k] for k in INPUT_KEYS});loss=prediction_loss(p,t['Y'],t['label_valid_mask'])
        actual_initial[name]=dict(shared_parameter_initial_max_abs_diff=0.,mismatch_count=0,
            shared_parameter_count=sum(v.numel() for v in m.native.parameters()),parameters=sum(v.numel() for v in m.parameters()),
            actual_shape=list(p.shape),finite_loss=bool(torch.isfinite(loss)),PASS=True)
    result=dict(PASS=True,native_mathematical_parity=rows,actual_initialization=actual_initial,
        dataset_protocol=data['protocol_id'],TRIVAL=data['split_summary'],TEST_access=False)
    save(out/'preflight.json',result);return result


def prepare(args):
    out=ROOT/'experiments/sep17_d0b_triweave'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    history=export_history(out/'historical_source')
    data_path=args.import_root/DATA_REL
    config=dict(protocol=PROTOCOL,import_root=str(args.import_root),data_path=str(data_path),created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        source_hashes={p:sha(ROOT/p) for p in ('cmgm/models/formal_d0b_main_ablation.py','cmgm/scripts/formal_main_innovation_ablation.py','tests/test_sep17_triweave_ablation.py')},
        historical_export_manifest_sha256=sha(out/'historical_source/source_manifest.json'),dataset_manifest_sha256=sha(data_path/'dataset_manifest.json'),
        target_order=[x['asset_id'] for x in json.loads((data_path/'target_manifest.json').read_text())['targets']],
        test_authorization='User explicitly authorized complete Sep17 suite training and TEST results on imported TriWeave data, 2026-10-22',
        checkpoint_dir=str(ROOT/'checkpoints/sep17_d0b_triweave'/out.name))
    save(out/'config.json',config);save(out/'status.json',dict(status='PREFLIGHT',completed=0,total=12))
    preflight(out,data_path)
    save(out/'status.json',dict(status='PREFLIGHT_PASSED',completed=0,total=12))
    print('PREPARED',str(out),flush=True)
    return out


def train_one(name,out,config,device):
    key=KEY[name];run=out/'runs'/key;run.mkdir(parents=True,exist_ok=True)
    cpdir=Path(config['checkpoint_dir'])/key;cpdir.mkdir(parents=True,exist_ok=True)
    best_path=cpdir/'best.pt';latest_path=cpdir/'latest.pt'
    if (run/'training_complete.json').exists():
        done=json.loads((run/'training_complete.json').read_text())
        if sha(best_path)!=done['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return done
    model=model_for(name,out,device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
    train_loader=loaders(Path(config['data_path']),'train');val_loader=loaders(Path(config['data_path']),'val')
    history=[];best=float('inf');best_epoch=None;stale=0;start_epoch=1;seconds=0.
    if latest_path.exists():
        cp=torch.load(latest_path,map_location='cpu',weights_only=False)
        if cp['config_sha256']!=sha(out/'config.json') or cp['name']!=name:raise ValueError('Resume identity mismatch')
        model.load_state_dict(cp['model_state_dict'],strict=True);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        model.set_epoch(cp['epoch']);restore_rng(cp['rng']);history=cp['history'];best=cp['best_val'];best_epoch=cp['best_epoch']
        stale=cp['stale'];start_epoch=cp['epoch']+1;seconds=cp['seconds']
    for epoch in range(start_epoch,201):
        if stale>=10:break
        tic=time.perf_counter();model.train();model.set_epoch(epoch);loss_sum=0.;kl_sum=0.;steps=0;origins=[]
        for batch in train_loader:
            b=moved(batch,device);optimizer.zero_grad(set_to_none=True)
            p=model(**{k:b[k] for k in INPUT_KEYS});pred=prediction_loss(p,b['Y'],b['label_valid_mask']);kl=model.switch_loss();loss=pred+kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if not torch.stack([torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None]).all():raise ValueError('Nonfinite gradient')
            optimizer.step();steps+=1;loss_sum+=pred.item();kl_sum+=kl.item();origins+=batch['origin_date']
        if len(origins)!=len(train_loader.dataset) or len(set(origins))!=len(origins):raise ValueError('TRAIN coverage failure')
        val=evaluate(model,val_loader,device,config['target_order']);score=val['selection'];scheduler.step(score)
        improved=score<best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        seconds+=time.perf_counter()-tic
        row=dict(epoch=epoch,train_prediction_batch_mean=loss_sum/steps,train_KL_batch_mean=kl_sum/steps,
            VAL_prediction_global_valid_label_Huber=score,VAL_MAE={h:r['MAE'] for h,r in val['horizons'].items()},
            optimizer_steps=steps,train_origins_used=len(origins),coverage=1.,LR=optimizer.param_groups[0]['lr'],
            active_gradient_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
            switch_beta=model.switching_latent_transformer.regime_filter.current_beta,seconds=time.perf_counter()-tic)
        history.append(row);save(run/'training_history.json',history)
        state=dict(name=name,epoch=epoch,best_epoch=best_epoch,best_val=best,stale=stale,seconds=seconds,
            model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=state_rng(),
            history=history,config_sha256=sha(out/'config.json'))
        if improved:checkpoint(best_path,state)
        checkpoint(latest_path,state)
        print(f'[TRAIN] {key} epoch={epoch} VAL_Huber={score:.10g} VAL5_MAE={val["horizons"]["5"]["MAE"]:.10g} best={best_epoch} steps={steps} seconds={row["seconds"]:.1f}',flush=True)
    cp=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True);model.set_epoch(cp['epoch'])
    val,arrays=evaluate(model,val_loader,device,config['target_order'],True)
    if not np.isclose(val['selection'],best,atol=1e-10,rtol=1e-6):raise ValueError('Best VAL reproduction mismatch')
    np.savez_compressed(run/'val_predictions.npz',**arrays)
    train=evaluate(model,train_loader,device,config['target_order'])
    save(run/'train_val_metrics.json',dict(train=train,val=val))
    done=dict(name=name,key=key,status='COMPLETE',best_epoch=cp['epoch'],best_val_Huber=best,parameters=sum(p.numel() for p in model.parameters()),
        active_parameters_observed=max(r['active_gradient_parameters'] for r in history),
        training_seconds=seconds,checkpoint=str(best_path),checkpoint_sha256=sha(best_path),history=str(run/'training_history.json'),
        train_val_metrics=str(run/'train_val_metrics.json'),strict_restore=True,test_evaluated=False)
    save(run/'training_complete.json',done)
    del model,optimizer,scheduler,cp
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return done


def release_test(out,config,freeze):
    # Use copied, frozen TriWeave preprocessing, never refit scaler or alter source data.
    root=out/'data_workspace';root.mkdir(exist_ok=True)
    if not (root/'data').exists():(root/'data').symlink_to(Path(config['import_root'])/'data',target_is_directory=True)
    if not (root/'configs').exists():(root/'configs').symlink_to(Path(config['import_root'])/'source_reference/configs',target_is_directory=True)
    phase=root/'experiments/phase_e_23_research';phase.mkdir(parents=True,exist_ok=True)
    source=Path(config['data_path'])
    for original,name in ((source/'protocol.json','frozen_protocol.json'),(source/'asset_manifest.json','asset_manifest.json')):
        if not (phase/name).exists():shutil.copyfile(original,phase/name)
    # Preserve exact archived source-index order in a private data view only.
    source_records=json.loads((source/'source_manifest.json').read_text())
    old_provenance=root/'experiments/phase_d_continuous/provenance.json'
    if not old_provenance.exists():
        save(old_provenance,dict(source_files=source_records,
            reconstructed_from=str(source/'source_manifest.json'),source_manifest_sha256=sha(source/'source_manifest.json'),
            scope='Exact archived source identities/order; no new values or selection'))
    pkg=out/'pipeline_snapshot'
    if not pkg.exists():
        # The source-reference snapshot contains every top-level preprocessing module.
        shutil.copytree(Path(config['import_root'])/'source_reference/src/triweave',pkg/'triweave')
    sys.path.insert(0,str(pkg));tr=importlib.import_module('triweave.test_release')
    dest=out/'test_dataset'
    if (dest/'dataset_manifest.json').exists():
        return tr.verify_dataset(dest,sha(out/'models_frozen.json'))
    if dest.exists():raise ValueError('Incomplete TEST release exists: inspect before any rerun')
    dest.mkdir()
    protocol,assets=tr.r23.definitions(root)
    frames,sources,logs,replay=tr.load_sources(root,assets,tr.ac.SPLITS['TEST'][1],authorized=True)
    calendar=tr.pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    aligned,quality=tr.align_sources(frames,assets,calendar,protocol,tr.ac.SPLITS['TEST'][1])
    rules=tr.read_json(source/'source_quarantine_manifest.json');aligned,accepted=tr.r23.apply_quarantine(aligned,assets,rules)
    raw_x,raw_mask,app=tr.ac.feature_panel(aligned,assets)
    params=tr.read_json(source/'scaler_params.json');z,mask=tr.ac.transform(raw_x,raw_mask,assets,params)
    arrays=dict(features=z,feature_valid_mask=mask,feature_applicable_mask=app,input_valid_mask=aligned['valid'],
        reference_dates=aligned['dates'],decision_time=aligned['decision_time'],source_observation_acceptance_mask=accepted)
    for k in ('observation_day','available_at','source_index','age_days','new_observation','trading_status'):arrays[k]=aligned[k]
    n=len(np.load(source/'reference_dates.npy',mmap_mode='r'));prefix={}
    for k,v in arrays.items():
        prefix[k]=bool(np.array_equal(v if k=='feature_applicable_mask' else v[:n],np.load(source/(k+'.npy'),mmap_mode='r')))
    prefix['unscaled_features']=bool(np.array_equal(raw_x[:n],np.load(source/'unscaled_features.npy',mmap_mode='r')))
    if not all(prefix.values()):save(dest/'FAILED_PREFIX.json',prefix);raise ValueError('TEST extension changed TRAIN/VAL inputs')
    y,m,ends,origins=tr.labels(aligned,assets,'TEST')
    offset=int(np.flatnonzero(aligned['dates']>=tr.ac.SPLITS['TEST'][0])[0])-19
    for k,v in arrays.items():np.save(dest/(k+'.npy'),v if k=='feature_applicable_mask' else v[offset:],allow_pickle=False)
    for k,v in dict(test_targets=y[origins],test_label_valid_mask=m[origins],test_origin_indices=origins-offset,
        test_window_indices=origins[:,None]+np.arange(-19,1)-offset).items():np.save(dest/(k+'.npy'),v,allow_pickle=False)
    save(dest/'source_manifest.json',sources);save(dest/'audit.json',dict(prefix=prefix,scaler_refitted=False,read_logs=logs,
        source_quality=quality,replay=replay,authorization=config['test_authorization'],frozen_sha256=sha(out/'models_frozen.json')))
    manifest=dict(release_id=tr.RELEASE,audit_pass=True,models_frozen_sha256=sha(out/'models_frozen.json'),
        origin_count=len(origins),first_origin=str(aligned['dates'][origins[0]]),last_origin=str(aligned['dates'][origins[-1]]),
        label_counts={str(h):int(m[origins,i].sum()) for i,h in enumerate((1,5,10,20))},
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},scaler_refitted=False)
    save(dest/'dataset_manifest.json',manifest);return manifest


def report(out,config,done,test):
    import csv
    rows=[];full=test['Full D0B']['horizons']['5']['MAE'] if 'Full D0B' in test else None
    for name in NAMES:
        if name not in test:continue
        r=test[name]['horizons']['5'];d=done[name]
        rows.append(dict(Variant=name,TEST_MAE_5d=r['MAE'],TEST_MSE_5d=r['MSE'],TEST_RMSE_5d=r['RMSE'],TEST_Hit_percent=100*r['Hit'],
            TEST_commodity_macro_MAE_5d=r['commodity_macro_MAE'],Delta_MAE=r['MAE']-full if full is not None else None,
            Relative_Delta_percent=100*(r['MAE']-full)/full if full else None,best_epoch=d['best_epoch'],parameters=d['parameters'],
            training_seconds=d['training_seconds'],checkpoint=d['checkpoint']))
    if rows:
        with (out/'main_TEST5_table.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    text=['# TEST：Sep17 D0B主创新消融 × TriWeave23','',
        '9eab8eeb原始门控D0B机制；11消融+Full。新数据311节点/26特征/23目标，保留经济类型数据和availability/feature/label masks。',
        '本实验不是旧24商品数据或其TEST MAE的数值复现；不是后来MoE/NoNorm/修订机制。训练budget/Huber/KL沿用原定义；drop_last=False、VAL按全体有效标签选模修复Phase A聚合问题。',
        '全部checkpoint在TEST前冻结。无TEST调参/选epoch/选seed。连续期货换月与历史PIT未经证实，结果属探索性供应商序列预测。','',
        '| Variant | TEST MAE@5d | ΔMAE | TEST RMSE@5d | Hit% | Best epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        delta=f"{r['Delta_MAE']:+.10f}" if r['Delta_MAE'] is not None else 'pending Full TEST'
        text.append(f"| {r['Variant']} | {r['TEST_MAE_5d']:.10f} | {delta} | {r['TEST_RMSE_5d']:.10f} | {r['TEST_Hit_percent']:.3f} | {r['best_epoch']} |")
    text += ['',f'已完成TEST {len(test)}/12。四周期、有效标签数和逐商品指标见test_metrics.json；模型/数据/源码清单见config、preflight及models_frozen。',
        '没有有效标签的商品显示NOT_EVALUABLE，pooled与commodity-macro明确区分。原始预测/标签/mask/origin/商品顺序保存为各run的test_predictions.npz。',
        '单seed不构成统计显著性或跨时期泛化证据。消融差值=消融−Full，负值如实保留。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')


def run(out,device):
    config=json.loads((out/'config.json').read_text())
    if config['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
    if not json.loads((out/'preflight.json').read_text())['PASS']:raise ValueError('Preflight absent')
    for p,h in config['source_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Source changed since preparation: '+p)
    if sha(Path(config['data_path'])/'dataset_manifest.json')!=config['dataset_manifest_sha256']:raise ValueError('Data manifest changed')
    if not torch.cuda.is_available():raise RuntimeError('Formal suite requires authorized CUDA; no CPU fallback')
    torch.set_num_threads(4)
    done={}
    # Full first is a stable reference; remaining order fixed by historical table.
    order=('Full D0B',)+tuple(n for n in NAMES if n!='Full D0B')
    for name in order:
        save(out/'status.json',dict(status='TRAINING',current=name,completed=len(done),total=12))
        done[name]=train_one(name,out,config,device);save(out/'training_index.json',done)
    freeze=dict(protocol=PROTOCOL,config_sha256=sha(out/'config.json'),checkpoints=done,models=list(NAMES),
        dataset_manifest_sha256=config['dataset_manifest_sha256'],test_authorized=True,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    if (out/'models_frozen.json').exists():
        previous=json.loads((out/'models_frozen.json').read_text())
        if previous['checkpoints']!=done:raise ValueError('Frozen checkpoints changed')
    else:save(out/'models_frozen.json',freeze)
    save(out/'status.json',dict(status='PREPARING_TEST',completed=12,total=12))
    release_test(out,config,freeze)
    test=json.loads((out/'test_metrics.json').read_text()) if (out/'test_metrics.json').exists() else {}
    for name in NAMES:
        if name in test:continue
        path=Path(done[name]['checkpoint'])
        if sha(path)!=done[name]['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
        cp=torch.load(path,map_location='cpu',weights_only=False);m=model_for(name,out,device)
        m.load_state_dict(cp['model_state_dict'],strict=True);m.set_epoch(cp['epoch']);m.eval()
        result,arrays=evaluate(m,loaders(out/'test_dataset','test'),device,config['target_order'],True)
        np.savez_compressed(out/'runs'/KEY[name]/'test_predictions.npz',**arrays)
        test[name]=dict(result,checkpoint=str(path),checkpoint_sha256=sha(path),best_epoch=cp['epoch'],status='COMPLETE',test_evaluated=True)
        save(out/'test_metrics.json',test);report(out,config,done,test)
        print('[TEST]',name,'MAE@5d',result['horizons']['5']['MAE'],flush=True)
        del m,cp;torch.cuda.empty_cache()
    save(out/'status.json',dict(status='COMPLETE',training_complete=12,test_complete=12,total=12,default_model_changed=False))
    save(out/'file_index.json',{str(p.relative_to(out)):dict(path=str(p),sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name not in ('file_index.json','runner.log')})
    print('SUITE COMPLETE',out,flush=True)


def recover_evaluation(out,device):
    """Finish a report-interrupted frozen TEST suite; cannot call train_one.

    Keep original config/source hashes and checkpoint selection unchanged.
    Permit only report/main edits and additive recovery code, proven by AST.
    """
    import ast
    config=json.loads((out/'config.json').read_text())
    freeze=json.loads((out/'models_frozen.json').read_text())
    done=json.loads((out/'training_index.json').read_text())
    if config['protocol']!=PROTOCOL or freeze['protocol']!=PROTOCOL or set(done)!=set(NAMES):
        raise ValueError('Frozen suite/protocol mismatch')
    if freeze['config_sha256']!=sha(out/'config.json') or freeze['checkpoints']!=done:
        raise ValueError('Frozen configuration/checkpoints changed')
    old_source=out/'source_at_training'
    for relative,digest in config['source_hashes'].items():
        old=old_source/relative;now=ROOT/relative
        if sha(old)!=digest:raise ValueError('Original source snapshot invalid: '+relative)
        if sha(now)==digest:continue
        if relative not in ('cmgm/scripts/formal_main_innovation_ablation.py','tests/test_sep17_triweave_ablation.py'):
            raise ValueError('Model source changed; evaluation recovery forbidden')
        before=ast.parse(old.read_text());after=ast.parse(now.read_text())
        old_funcs={n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new_funcs={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        allowed={'report','main'} if relative.startswith('cmgm/') else set()
        if any(new_funcs.get(k)!=v for k,v in old_funcs.items() if k not in allowed):
            raise ValueError('Training/model/metric/evaluator code changed')
        if relative.startswith('cmgm/'):
            if set(new_funcs)-set(old_funcs)!={'recover_evaluation'}:raise ValueError('Unexpected recovery code')
            old_globals=[ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))]
            new_globals=[ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))]
            if old_globals!=new_globals:raise ValueError('Global protocol/imports changed')
    manifest=json.loads((out/'test_dataset/dataset_manifest.json').read_text())
    if not manifest['audit_pass'] or manifest['models_frozen_sha256']!=sha(out/'models_frozen.json'):
        raise ValueError('Different TEST release')
    for name,digest in manifest['files'].items():
        if sha(out/'test_dataset'/name)!=digest:raise ValueError('TEST artifact changed: '+name)
    historical=json.loads((out/'historical_source/source_manifest.json').read_text())
    for name,entry in historical.items():
        path=out/'historical_source/cmgm_sep17_frozen'/Path(name).relative_to('cmgm')
        if sha(path)!=entry['namespaced_sha256']:raise ValueError('Native historical source changed')
    protected={str(out/'config.json'):sha(out/'config.json'),str(out/'models_frozen.json'):sha(out/'models_frozen.json')}
    for name,row in done.items():
        cp=Path(row['checkpoint'])
        if row['status']!='COMPLETE' or sha(cp)!=row['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint invalid: '+name)
        history=json.loads(Path(row['history']).read_text())
        best=min(history,key=lambda x:x['VAL_prediction_global_valid_label_Huber'])
        if best['epoch']!=row['best_epoch'] or best['VAL_prediction_global_valid_label_Huber']!=row['best_val_Huber']:
            raise ValueError('VAL selection/history mismatch: '+name)
        for p in (cp,cp.with_name('latest.pt'),Path(row['history']),Path(row['train_val_metrics'])):
            protected[str(p)]=sha(p)
    test=json.loads((out/'test_metrics.json').read_text()) if (out/'test_metrics.json').exists() else {}
    # Preserve the original failed state; do not erase the cause or first result.
    recovery=out/('evaluation_recovery_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
    recovery.mkdir()
    for name in ('status.json','failure.json','runner.log','test_metrics.json','main_TEST5_table.csv'):
        if (out/name).exists():shutil.copyfile(out/name,recovery/name)
    audit=dict(status='RUNNING',reason='Report formatted None delta before Full TEST existed',
        training_restarted=False,training_or_metric_functions_unchanged_AST=True,
        original_source_hashes=config['source_hashes'],current_runner_sha256=sha(Path(__file__)),
        frozen_checkpoint_count=len(done),already_evaluated=list(test),protected_before=protected)
    save(recovery/'audit.json',audit)
    torch.set_num_threads(4)
    report(out,config,done,test)
    for name in NAMES:
        if name in test:
            arrays=np.load(out/'runs'/KEY[name]/'test_predictions.npz',allow_pickle=False)
            recalculated=metrics(arrays['prediction'],arrays['target'],arrays['label_valid_mask'],config['target_order'])
            if recalculated['horizons']!=test[name]['horizons']:raise ValueError('Saved TEST statistics differ')
            if test[name]['checkpoint_sha256']!=done[name]['checkpoint_sha256']:raise ValueError('Wrong saved TEST checkpoint')
            continue
        save(out/'status.json',dict(status='EVALUATING_TEST',training_complete=12,test_complete=len(test),current=name,total=12))
        cp=torch.load(done[name]['checkpoint'],map_location='cpu',weights_only=False)
        if cp['name']!=name or cp['epoch']!=done[name]['best_epoch']:raise ValueError('Checkpoint identity mismatch')
        m=model_for(name,out,device);m.load_state_dict(cp['model_state_dict'],strict=True);m.set_epoch(cp['epoch']);m.eval()
        result,arrays=evaluate(m,loaders(out/'test_dataset','test'),device,config['target_order'],True)
        np.savez_compressed(out/'runs'/KEY[name]/'test_predictions.npz',**arrays)
        test[name]=dict(result,checkpoint=done[name]['checkpoint'],checkpoint_sha256=sha(done[name]['checkpoint']),
            best_epoch=cp['epoch'],status='COMPLETE',test_evaluated=True)
        save(out/'test_metrics.json',test);report(out,config,done,test)
        print('[RECOVERED TEST]',name,'MAE@5d',result['horizons']['5']['MAE'],flush=True)
        del m,cp
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if any(sha(Path(p))!=h for p,h in protected.items()):raise ValueError('Frozen training artifact changed during inference')
    audit.update(status='COMPLETE',test_complete=len(test),protected_after_identical=True)
    save(recovery/'audit.json',audit)
    save(out/'status.json',dict(status='COMPLETE',training_complete=12,test_complete=12,total=12,
        default_model_changed=False,report_error_recovered=True,recovery_audit=str(recovery/'audit.json')))
    save(out/'file_index.json',{str(p.relative_to(out)):dict(path=str(p),sha256=sha(p)) for p in out.rglob('*')
        if p.is_file() and p.name not in ('file_index.json','runner.log','evaluation_recovery.log')})
    print('FROZEN EVALUATION COMPLETE',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',type=Path);parser.add_argument('--import-root',type=Path,default=IMPORT);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--recover-evaluation',action='store_true',help='Report-only audited recovery; never retrain')
    args=parser.parse_args()
    if args.recover_evaluation and (not args.resume or args.run):raise ValueError('Use --resume --recover-evaluation without --run')
    out=args.resume.resolve() if args.resume else prepare(args)
    if args.run or args.recover_evaluation:
        with (out/'runner.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if args.recover_evaluation:recover_evaluation(out,args.device)
                else:run(out,args.device)
            except Exception as e:
                save(out/'failure.json',dict(error=type(e).__name__,message=str(e),time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()));raise


if __name__=='__main__':main()

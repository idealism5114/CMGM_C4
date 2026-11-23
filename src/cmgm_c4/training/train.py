"""Explicit stable training entry. Imported for availability, never auto-executed."""
from pathlib import Path
import copy
import math
import time
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,seed_all,checkpoint,rng_state,restore_rng,prediction_loss,sha,runtime_hashes,audit_train_val_files,tensor_hash
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import forward,evaluate
from cmgm_c4.factory import build


def effective_protocol(defaults,lr=None,weight_decay=None):
    """Override only optimizer hyperparameters, preserving the default protocol."""
    result=copy.deepcopy(defaults)
    for name,value in (('lr',lr),('weight_decay',weight_decay)):
        if value is None:continue
        value=float(value)
        if not math.isfinite(value) or value<0 or (name=='lr' and value==0):
            raise ValueError('Invalid optimizer parameter '+name)
        result[name]=value
    return result


def train(identifier,out,device='cuda:0',resume=False,*,lr=None,weight_decay=None):
    root=project_root();cfg=read(root/'configs/project.json');out=Path(out).resolve()
    protocol=effective_protocol(cfg['train'],lr,weight_decay)
    if out.is_relative_to(root) and not out.is_relative_to(root/'runs'):
        raise ValueError('Use runs/ for new work; all migrated project files are immutable')
    if out.exists() and not resume:raise ValueError('New training requires a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);neutral=identifier.startswith('baselines/')
    m=build(identifier,initial=True,device=device);seed_all(42)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=protocol['scheduler_factor'],patience=protocol['scheduler_patience'])
    batches,val=loader('train',neutral),loader('val',neutral)
    history=[];best=float('inf');best_epoch=None;stale=0;start=1;seconds=0.
    signature=dict(identifier=identifier,protocol=protocol,selector=cfg['selector'],dataset_sha256=cfg['dataset_manifest_sha256'],
        source_hashes=runtime_hashes(),data_audit=audit_train_val_files(),initial_state_sha256=tensor_hash(m.state_dict()))
    if resume:
        cp=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if cp['signature']!=signature:raise ValueError('Resume protocol mismatch')
        m.load_state_dict(cp['model_state_dict'],strict=True);opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);restore_rng(cp['rng'])
        history,best,best_epoch,stale,start,seconds=cp['history'],cp['best'],cp['best_epoch'],cp['stale'],cp['epoch']+1,cp['seconds']
    save(out/'config.json',signature)
    for epoch in range(start,protocol['max_epochs']+1):
        if stale>=protocol['patience']:break
        tic=time.perf_counter();m.train()
        if hasattr(m,'set_epoch'):m.set_epoch(epoch)
        logs=[];used=[]
        for b in batches:
            opt.zero_grad(set_to_none=True);p,_=forward(m,b,device,neutral)
            pred=prediction_loss(p,b['Y'].to(device),b['label_valid_mask'].to(device))
            kl=m.switch_loss() if hasattr(m,'switch_loss') else pred.new_zeros(())
            loss=pred+kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('Nonfinite parameter gradient')
            opt.step();logs.append((float(pred.detach()),float(kl.detach())));used+=b['origin_date']
        if len(used)!=2726 or len(set(used))!=2726 or len(logs)!=43:raise ValueError('TRAIN coverage mismatch')
        metrics,_=evaluate(m,val,device,neutral);score=metrics['primary'];scheduler.step(score)
        improved=score<best
        if improved:best,best_epoch,stale=score,epoch,0
        else:stale+=1
        elapsed=time.perf_counter()-tic;seconds+=elapsed
        history.append(dict(epoch=epoch,VAL_5D_COMMODITY_MACRO_MAE=score,VAL=metrics,
            train_Huber_batch_mean=float(np.mean(logs,0)[0]),train_KL_batch_mean=float(np.mean(logs,0)[1]),
            optimizer_steps=43,train_origins_used=2726,lr=opt.param_groups[0]['lr'],seconds=elapsed))
        cp=dict(identifier=identifier,signature=signature,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,seconds=seconds,
            history=history,model_state_dict=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),selection=cfg['selector'])
        if improved:checkpoint(out/'best.pt',cp)
        checkpoint(out/'latest.pt',cp);save(out/'history.json',history)
        print(identifier,epoch,score,'best',best_epoch,flush=True)
    save(out/'models_frozen.json',dict(identifier=identifier,best_epoch=best_epoch,checkpoint='best.pt',checkpoint_sha256=sha(out/'best.pt'),selection=cfg['selector'],
        parameters=sum(v.numel() for v in m.parameters()),trainable_parameters=sum(v.numel() for v in m.parameters() if v.requires_grad),training_seconds=seconds))
    frozen=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(frozen['model_state_dict'],strict=True)
    if hasattr(m,'set_epoch'):m.set_epoch(best_epoch)
    scores={}
    for split in ('train','val'):
        scores[split],arrays=evaluate(m,loader(split,neutral),device,neutral,True)
        np.savez_compressed(out/(split+'_predictions.npz'),**arrays)
    save(out/'train_val_metrics.json',scores)
    save(out/'COMPLETE.json',dict(status='COMPLETE',signature=signature,best_epoch=best_epoch,TEST_evaluated=False))


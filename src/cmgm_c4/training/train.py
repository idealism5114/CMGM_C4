"""Explicit stable training entry. Imported for availability, never auto-executed."""
from pathlib import Path
import copy
import math
import time
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,seed_all,checkpoint,rng_state,restore_rng,prediction_loss,sha,runtime_hashes,audit_train_val_files,tensor_hash
from cmgm_c4.data.dataset import FrozenDataset, loader
from cmgm_c4.evaluation.evaluate import forward,evaluate
from cmgm_c4.factory import build
from cmgm_c4.models.progressive import ROWS
from cmgm_c4.models.horizon import BRANCHES


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


def training_protocol(identifier, defaults, lr=None, weight_decay=None, *, root=None):
    if not identifier.startswith('baselines/'):
        return effective_protocol(defaults, lr, weight_decay)
    root = project_root() if root is None else root
    profile = read(root / 'configs' / (identifier + '.json'))
    settings = {name: defaults[name] for name in
                ('seed', 'shuffle', 'drop_last', 'Huber_delta', 'epoch0_candidate')}
    settings.update(profile['train'], training_source=profile['training_source'])
    return effective_protocol(settings, lr, weight_decay)


def make_optimizer(model, protocol):
    if protocol['optimizer'] != 'Adam':
        raise ValueError(f"Unsupported optimizer {protocol['optimizer']}")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=protocol['lr'], weight_decay=protocol['weight_decay'],
        betas=tuple(protocol.get('betas', (0.9, 0.999))), eps=protocol.get('eps', 1e-8),
    )
    schedule = protocol.get('scheduler', 'plateau')
    if schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=protocol['scheduler_factor'], patience=protocol['scheduler_patience'])
    elif schedule == 'type1':
        # The official scripts set the LR after epoch e to initial_lr * 0.5**(e-1).
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epochs: 0.5 ** max(0, epochs - 1))
    elif schedule == 'transformer_warmup':
        warmup = protocol['warmup_steps']

        def warmup_factor(completed_steps):
            step = completed_steps + 1
            return min(step / warmup, math.sqrt(warmup / step))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)
    elif schedule == 'constant':
        scheduler = None
    else:
        raise ValueError(f'Unknown LR schedule {schedule}')
    return optimizer, scheduler


def baseline_train_loader(dataset, batch_size, horizons):
    """Keep chronological batches and move the last boundary only if the tail lacks labels."""
    valid = dataset.a['train_label_valid_mask'].any(axis=2)
    batches = [range(i, min(i + batch_size, len(dataset))) for i in range(0, len(dataset), batch_size)]
    if len(batches) >= 2:
        previous, tail = batches[-2:]
        if not valid[tail.start:tail.stop].any(axis=0).all():
            # Borrow the fewest adjacent origins needed, respecting the configured batch size.
            earliest = max(previous.start + 1, tail.stop - batch_size)
            for boundary in range(tail.start - 1, earliest - 1, -1):
                if not valid[previous.start:boundary].any(axis=0).all():
                    break
                if valid[boundary:tail.stop].any(axis=0).all():
                    batches[-2:] = [range(previous.start, boundary), range(boundary, tail.stop)]
                    print(f'Adjusted final TRAIN batches from {len(previous)}+{len(tail)} '
                          f'to {boundary - previous.start}+{tail.stop - boundary}', flush=True)
                    break
    for batch in batches:
        covered = valid[batch.start:batch.stop].any(axis=0)
        if not covered.all():
            missing = [h for h, ok in zip(horizons, covered) if not ok]
            raise ValueError(f'Baseline TRAIN batch {dataset.dates[batch.start]} to '
                             f'{dataset.dates[batch.stop - 1]} has no valid labels for horizons {missing}')
    return torch.utils.data.DataLoader(dataset, batch_sampler=batches)


def train(identifier,out,device='cuda:0',resume=False,*,lr=None,weight_decay=None,seed=None):
    root=project_root();cfg=read(root/'configs/project.json');out=Path(out).resolve()
    variants = {'progressive/'+row for row in ROWS} | {'c4v2/'+row for row in BRANCHES}
    if identifier not in read(root/'configs/registry.json') and identifier not in variants:
        raise ValueError('Unknown stable experiment: '+identifier)
    protocol=training_protocol(identifier,cfg['train'],lr,weight_decay,root=root)
    if seed is not None:
        if identifier not in variants:
            raise ValueError('Custom seeds are supported for paired ablations')
        protocol['seed']=seed
    if out.is_relative_to(root) and not out.is_relative_to(root/'runs'):
        raise ValueError('Use runs/ for new work; all migrated project files are immutable')
    if out.exists() and not resume:raise ValueError('New training requires a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);neutral=identifier.startswith('baselines/')
    m=build(identifier,initial=True,device=device,seed=protocol['seed']);seed_all(protocol['seed'])
    opt,scheduler=make_optimizer(m,protocol)
    schedule=protocol.get('scheduler','plateau')
    if neutral:
        batches=baseline_train_loader(FrozenDataset('train', neutral=True), protocol['batch_size'], cfg['horizons'])
    else:
        batches=loader('train',batch_size=protocol['batch_size'])
    val=loader('val',neutral)
    history=[];best=float('inf');best_epoch=None;stale=0;start=1;seconds=0.
    signature=dict(identifier=identifier,protocol=protocol,selector=cfg['selector'],dataset_sha256=cfg['dataset_manifest_sha256'],
        source_hashes=runtime_hashes(),data_audit=audit_train_val_files(),initial_state_sha256=tensor_hash(m.state_dict()))
    if resume and (out/'latest.pt').exists():
        cp=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if cp['signature']!=signature:raise ValueError('Resume protocol mismatch')
        m.load_state_dict(cp['model_state_dict'],strict=True);opt.load_state_dict(cp['optimizer'])
        if scheduler is not None:scheduler.load_state_dict(cp['scheduler'])
        restore_rng(cp['rng'])
        history,best,best_epoch,stale,start,seconds=cp['history'],cp['best'],cp['best_epoch'],cp['stale'],cp['epoch']+1,cp['seconds']
    elif resume and any(out.iterdir()):
        if not (out/'config.json').exists() or read(out/'config.json')!=signature:
            raise ValueError('Resume protocol mismatch')
    save(out/'config.json',signature)
    for epoch in range(start,protocol['max_epochs']+1):
        if protocol['patience'] is not None and stale>=protocol['patience']:break
        tic=time.perf_counter();m.train()
        if hasattr(m,'set_epoch'):m.set_epoch(epoch)
        logs=[];used=[];epoch_lr=opt.param_groups[0]['lr']
        for b in batches:
            opt.zero_grad(set_to_none=True);p,_=forward(m,b,device,neutral)
            pred=prediction_loss(p,b['Y'].to(device),b['label_valid_mask'].to(device))
            kl=m.switch_loss() if hasattr(m,'switch_loss') else pred.new_zeros(())
            loss=pred+kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('Nonfinite parameter gradient')
            if protocol.get('grad_clip_norm') is not None:
                torch.nn.utils.clip_grad_norm_(m.parameters(), protocol['grad_clip_norm'], error_if_nonfinite=True)
            epoch_lr=opt.param_groups[0]['lr']
            opt.step()
            if schedule=='transformer_warmup':scheduler.step()
            logs.append((float(pred.detach()),float(kl.detach())));used+=b['origin_date']
        if len(used)!=2726 or used!=list(batches.dataset.dates) or len(logs)!=len(batches):
            raise ValueError('TRAIN coverage mismatch')
        metrics,_=evaluate(m,val,device,neutral);score=metrics['primary']
        if schedule=='plateau':
            scheduler.step(score)
            epoch_lr=opt.param_groups[0]['lr']
        elif schedule=='type1':
            scheduler.step()
        improved=score<best
        if improved:best,best_epoch,stale=score,epoch,0
        else:stale+=1
        elapsed=time.perf_counter()-tic;seconds+=elapsed
        history.append(dict(epoch=epoch,VAL_5D_COMMODITY_MACRO_MAE=score,VAL=metrics,
            train_Huber_batch_mean=float(np.mean(logs,0)[0]),train_KL_batch_mean=float(np.mean(logs,0)[1]),
            optimizer_steps=len(logs),train_origins_used=len(used),lr=epoch_lr,seconds=elapsed))
        cp=dict(identifier=identifier,signature=signature,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,seconds=seconds,
            history=history,model_state_dict=m.state_dict(),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,rng=rng_state(),selection=cfg['selector'])
        if improved:checkpoint(out/'best.pt',cp)
        checkpoint(out/'latest.pt',cp);save(out/'history.json',history)
        print(identifier,epoch,score,'best',best_epoch,flush=True)
    save(out/'history.json',history)
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


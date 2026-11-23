"""Frozen split evaluation; does not fit, train, or choose a checkpoint."""
import numpy as np
import torch
from cmgm_c4.common import tensor_hash
from .metrics import population


def forward(model,batch,device,neutral=False,capture=False):
    if neutral:return model(batch['X'].to(device)),{}
    keys=('X','input_valid_mask','feature_valid_mask','feature_applicable_mask')
    return model(**{k:batch[k].to(device) for k in keys},return_aux=True)


@torch.inference_mode()
def evaluate(model,batches,device='cpu',neutral=False,capture=False):
    model.eval();before=tensor_hash(model.state_dict());parts={k:[] for k in ('prediction','target','label_valid_mask')};dates=[];auxes={}
    for b in batches:
        p,a=forward(model,b,device,neutral,capture)
        if p.shape!=b['Y'].shape or not torch.isfinite(p).all():raise ValueError('Bad model output')
        parts['prediction'].append(p.cpu().numpy());parts['target'].append(b['Y'].numpy());parts['label_valid_mask'].append(b['label_valid_mask'].numpy());dates+=b['origin_date']
        if capture and a:
            for k in ('routing','expert_prediction'):auxes.setdefault(k,[]).append(a[k].cpu().numpy())
    if dates!=list(batches.dataset.dates) or len(dates)!=len(batches.dataset):raise ValueError('Coverage/order mismatch')
    if before!=tensor_hash(model.state_dict()):raise ValueError('Inference mutated parameters')
    a={k:np.concatenate(v) for k,v in parts.items()};a.update({k:np.concatenate(v) for k,v in auxes.items()});a['origin_dates']=np.array(dates)
    return population(a['prediction'],a['target'],a['label_valid_mask']),a


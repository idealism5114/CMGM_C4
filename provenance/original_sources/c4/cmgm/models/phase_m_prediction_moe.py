"""Phase M: fixed four-expert prediction mixtures, on the original D0B only.

Backbone modules and the shared nonlinear head are reused, not copied or frozen.
The C0 path delegates to the original forward. No trained checkpoint is used by
the matched initialization factory. Temporal sequences are live return values.
"""
import copy
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch import nn
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.four_task_moe import (
    RecentChangeExpert, LongMemoryExpert, SpatialStructureExpert, LowRankInteractionExpert,
)
from cmgm.models.horizon_tensor_fusion import predict_by_horizon
from cmgm.models.frozen_d0b_task_experts import tensor_state_hash

HORIZONS = (1, 5, 10, 20)
EXPERTS = ('recent_change', 'long_memory', 'spatial_structure', 'st_interaction')
MODEL_IDS = (
    'moe_phase_m/d0b_original',
    'moe_phase_m/pred_global_static',
    'moe_phase_m/pred_short_long_static',
    'moe_phase_m/pred_horizon_static',
    'moe_phase_m/pred_global_dynamic',
    'moe_phase_m/pred_horizon_dynamic',
    'moe_phase_m/rep_horizon_dynamic',
)
KEYS = tuple(f'C{i}' for i in range(7))
MODES = dict(zip(KEYS, MODEL_IDS))


class MeanAugmentedRecentExpert(RecentChangeExpert):
    """Reuse the prior recent-expert structure, explicitly include its mean too.

    Compared with the old 128->64 class this is 192->64: [last, mean5, last-mean5].
    The extra input is linearly redundant but adds 4096 parameters. All six MoEs
    use this same fixed definition; old results are not matched controls.
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.projection = nn.Linear(192, 64)

    def forward(self, latent):
        if latent.ndim != 3 or latent.shape[1:] != (20, 64):
            raise ValueError('Expected live Z: B,20,64')
        last, mean = latent[:, -1], latent[:, -5:].mean(1)
        return torch.relu(self.projection(torch.cat((last, mean, last-mean), -1)))


class PureLowRankInteractionExpert(LowRankInteractionExpert):
    """Reuse the old rank16 matrices, omit its old temporal residual by design."""
    def forward(self, spatial, temporal):
        return self.output(self.spatial_rank(spatial)*self.temporal_rank(temporal))


class ExpertBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.recent_change = MeanAugmentedRecentExpert()
        self.long_memory = LongMemoryExpert()
        self.spatial_structure = SpatialStructureExpert()
        self.st_interaction = PureLowRankInteractionExpert()

    def forward(self, s, t, h, z):
        return torch.stack((self.recent_change(z), self.long_memory(h),
                            self.spatial_structure(s), self.st_interaction(s,t)), 1)


class PhaseMRouter(nn.Module):
    """Temperature1; all four candidates always accessible; no auxiliary loss."""
    def __init__(self, control):
        super().__init__()
        self.control = control
        if control in ('C1','C2','C3'):
            self.logits = nn.Parameter(torch.zeros({'C1':1,'C2':2,'C3':4}[control],4))
        elif control in ('C4','C5','C6'):
            self.long_projection = nn.Linear(128,64)
            self.norm_s = nn.LayerNorm(64)
            self.norm_t = nn.LayerNorm(64)
            self.norm_change = nn.LayerNorm(64)
            self.norm_long = nn.LayerNorm(64)
            if control != 'C4':
                self.horizon_embedding = nn.Embedding(4,8)
                nn.init.normal_(self.horizon_embedding.weight,mean=0.,std=.02)
            self.network = nn.Sequential(nn.Linear(256 if control=='C4' else 264,32),
                                         nn.GELU(),nn.Linear(32,4))
            nn.init.normal_(self.network[-1].weight,mean=0.,std=1e-3)
            nn.init.zeros_(self.network[-1].bias)
        else:raise ValueError(control)

    def context(self,s,t,h,z):
        change=z[:,-1]-z[:,-5:].mean(1)
        return torch.cat((self.norm_s(s),self.norm_t(t),self.norm_change(change),
                          self.norm_long(self.long_projection(h.mean(1)))), -1)

    def forward(self,s,t,h,z):
        if self.control in ('C1','C2','C3'):
            weights=self.logits.softmax(-1)
            indices={'C1':[0,0,0,0],'C2':[0,0,1,1],'C3':[0,1,2,3]}[self.control]
            return weights[indices][None].expand(len(s),-1,-1)
        context=self.context(s,t,h,z)
        if self.control=='C4':
            return self.network(context).softmax(-1)[:,None].expand(-1,4,-1)
        embedding=self.horizon_embedding.weight[None].expand(len(s),-1,-1)
        inputs=torch.cat((context[:,None].expand(-1,4,-1),embedding),-1)
        return self.network(inputs).softmax(-1)


def expert_forecasts(head,experts,commodities):
    """One shared head-body call; candidate predictions B,H,K,C.

    Body dropout is indexed by expert here; C6's body is indexed by horizon.
    K=H=4, so both call the same body once on B,4,64, consuming the same count
    of random draws, but masks act on different mathematical representations.
    Eval comparisons have no dropout. No separate expert or horizon heads.
    """
    if experts.ndim!=3 or experts.shape[1:]!=(4,64):raise ValueError('Expected B,4,64 experts')
    body=head[:3](experts)
    final=head[3]
    if final.in_features!=64 or final.out_features!=4*commodities:
        raise ValueError('Shared horizon-major head shape changed')
    return torch.einsum('bkd,hcd->bhkc',body,final.weight.view(4,commodities,64))+final.bias.view(4,commodities)[None,:,None]


def checked_weights(weights,batch,device,dtype):
    w=torch.as_tensor(weights,device=device,dtype=dtype)
    if w.shape==(4,4):w=w[None].expand(batch,-1,-1)
    if (w.shape!=(batch,4,4) or not torch.isfinite(w).all() or (w<0).any()
            or not torch.allclose(w.sum(-1),torch.ones(batch,4,device=device,dtype=dtype),atol=1e-6,rtol=1e-6)):
        raise ValueError('Routing override must be normalized H,K or B,H,K')
    return w


class PhaseMModel(nn.Module):
    def __init__(self,control,model_kwargs):
        super().__init__()
        if control not in KEYS:raise ValueError(control)
        self.control=control;self.variant=MODES[control]
        self.base=HeteroMixHopCMGM(variant='switching_latent_balanced_readout',**model_kwargs)
        self.n_commodities=self.base.n_commodities
        if self.base.feat_dim!=21 or self.base.n_horizons!=4 or self.n_commodities!=24:
            raise ValueError('Phase M requires original CMGM 21-feature/24-target protocol, not TriWeave')
        if control!='C0':
            del self.base.gate_fc  # No inactive gate remains registered.
            self.experts=ExpertBank()
            self.router=PhaseMRouter(control)

    @property
    def switching_latent_transformer(self):return self.base.switching_latent_transformer

    @property
    def graph_learner(self):return self.base.graph_learner

    def set_epoch(self,epoch):return self.switching_latent_transformer.set_epoch(epoch)

    def forward(self,x,*,debug=False,return_aux=False,routing_override=None,with_candidates=False):
        if self.control=='C0':
            if routing_override is not None:raise ValueError('C0 has no router')
            p=self.base(x,debug=debug)
            return (p,{}) if return_aux else p
        hs=self.base._temp_weighted_spatial(x)
        ht,h,z=self.switching_latent_transformer(x,return_live_sequences=True)
        s,t=self.base.gcn_proj(hs),self.base.lstm_proj(ht)
        expert=self.experts(s,t,h,z)
        native_pi=self.router(s,t,h,z)
        pi=native_pi if routing_override is None else checked_weights(routing_override,len(x),s.device,s.dtype)
        candidates=None
        if self.control=='C6':
            states=torch.einsum('bhk,bkd->bhd',pi,expert)
            prediction=predict_by_horizon(self.base.head,states,self.n_commodities)
            if with_candidates:
                if self.training:raise ValueError('Extra C6 candidate forecasts are eval-only; no extra training dropout')
                candidates=expert_forecasts(self.base.head,expert,self.n_commodities)
        else:
            candidates=expert_forecasts(self.base.head,expert,self.n_commodities)
            prediction=torch.einsum('bhk,bhkc->bhc',pi,candidates)
        if return_aux:
            return prediction,dict(H=h,Z=z,s=s,t=t,experts=expert,routing=pi,native_routing=native_pi,
                expert_prediction=candidates)
        return prediction


@contextmanager
def preserved_rng():
    py,npstate=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:yield
        finally:random.setstate(py);np.random.set_state(npstate)


def copy_shared(source,target,keys):
    src,dst=source.state_dict(),target.state_dict()
    for k in keys:
        if k not in dst or dst[k].shape!=src[k].shape:raise ValueError('Shared key differs: '+k)
        dst[k]=src[k].clone()
    target.load_state_dict(dst,strict=True)
    diff={k:float((src[k]-target.state_dict()[k]).abs().max()) for k in keys
          if not torch.equal(src[k],target.state_dict()[k])}
    return dict(copied_keys=list(keys),shared_parameter_count=sum(src[k].numel() for k in keys),
        shared_parameter_initial_max_abs_diff=max(diff.values(),default=0.),mismatch_count=len(diff),mismatches=diff,PASS=not diff)


def matched_models(kwargs,seed=42):
    """Fresh original D0B state; explicit copying, never trained checkpoint state."""
    with preserved_rng():
        torch.manual_seed(seed)
        base=PhaseMModel('C0',kwargs)
        models={'C0':base};audit={'shared':{},'experts':{},'routers':{}}
        for control in KEYS[1:]:
            torch.manual_seed(seed)
            m=PhaseMModel(control,kwargs)
            keys=[k for k in base.state_dict() if not k.startswith('base.gate_fc.')]
            audit['shared'][control]=copy_shared(base,m,keys)
            models[control]=m
        for control in KEYS[2:]:
            keys=[k for k in models['C1'].state_dict() if k.startswith('experts.')]
            audit['experts'][control]=copy_shared(models['C1'],models[control],keys)
        audit['routers']['C6_from_C5']=copy_shared(models['C5'],models['C6'],list(models['C5'].state_dict()))
        # Same dynamic-context transforms, last projection, and context columns.
        a,b=models['C5'].router,models['C4'].router
        keys=[k for k in a.state_dict() if not k.startswith(('network.0.','horizon_embedding.'))]
        copy_shared(a,b,keys)
        with torch.no_grad():
            b.network[0].weight.copy_(a.network[0].weight[:,:256])
            b.network[0].bias.copy_(a.network[0].bias)
        audit['routers']['C4_context_from_C5']=dict(context_keys=keys,context_columns_identical=bool(torch.equal(b.network[0].weight,a.network[0].weight[:,:256])),PASS=True)
        audit['initial_state_sha256']={k:tensor_state_hash(m.state_dict()) for k,m in models.items()}
        audit['expert_state_sha256']={k:tensor_state_hash(m.experts.state_dict()) for k,m in models.items() if k!='C0'}
        audit['C5_C6_entire_initial_state_identical']=audit['initial_state_sha256']['C5']==audit['initial_state_sha256']['C6']
        audit['PASS']=all(v['PASS'] for section in ('shared','experts','routers') for v in audit[section].values()) and audit['C5_C6_entire_initial_state_identical']
        if not audit['PASS']:raise ValueError('Matched initialization failed')
    return models,audit


def parameter_counts(model):
    total=sum(p.numel() for p in model.parameters())
    return dict(total=total,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        backbone_and_head=sum(p.numel() for p in model.base.parameters()),
        experts={n:sum(p.numel() for p in getattr(model.experts,n).parameters()) for n in EXPERTS} if model.control!='C0' else {},
        router=sum(p.numel() for p in model.router.parameters()) if model.control!='C0' else 0)

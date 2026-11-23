"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .temporal import PhaseMLiveBackbone
from .experts import ExpertBank,expert_forecasts,checked_weights
from .router import C4Router
KEYS=('C0','C4')
MODES={'C0':'moe_phase_m/d0b_original','C4':'moe_phase_m/pred_global_dynamic'}


class NewDataPhaseMModel(nn.Module):
    def __init__(self, control="C4", history_root=None, **shape):
        super().__init__()
        if control not in KEYS:raise ValueError(control)
        self.control=control;self.variant=MODES[control]
        self.encoder=PhaseMLiveBackbone('Full D0B',history_root,**shape)
        self.n_commodities=self.encoder.targets
        if control!='C0':
            del self.base.gate_fc
            self.experts=ExpertBank()
            self.router=C4Router()

    @property
    def base(self):return self.encoder.native

    @property
    def switching_latent_transformer(self):return self.encoder.switching_latent_transformer

    def set_epoch(self,epoch):return self.encoder.set_epoch(epoch)

    def switch_loss(self):return self.encoder.switch_loss()

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,
                *,return_aux=False,routing_override=None,with_candidates=False):
        if self.control=='C0':
            if routing_override is not None:raise ValueError('C0 has no router')
            p=self.encoder(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
            return (p,{}) if return_aux else p
        app=feature_applicable_mask
        if app.ndim==2:app=app[None,None]
        elif app.ndim==3:app=app[:,None]
        fm=feature_valid_mask & app & input_valid_mask[...,None]
        x=torch.where(fm,X,0.)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite valid input')
        valid=input_valid_mask & fm.any(-1)
        self.encoder.diagnostics={}
        hs=self.encoder.spatial(x,valid)
        ht,h,z=self.encoder.temporal(x,valid,return_live_sequences=True)
        s,t=self.base.gcn_proj(hs),self.base.lstm_proj(ht)
        bank=self.experts(s,t,h,z)
        native_pi=self.router(s,t,h,z)
        pi=native_pi if routing_override is None else checked_weights(routing_override,len(x),s.device,s.dtype)
        candidates=expert_forecasts(self.base.head,bank,self.n_commodities)
        prediction=torch.einsum('bhk,bhkc->bhc',pi,candidates)
        aux=dict(H=h,Z=z,s=s,t=t,experts=bank,routing=pi,native_routing=native_pi,expert_prediction=candidates)
        return (prediction,aux) if return_aux else prediction


class C4Model(NewDataPhaseMModel):
    def __init__(self,**shape):
        super().__init__("C4",**shape)

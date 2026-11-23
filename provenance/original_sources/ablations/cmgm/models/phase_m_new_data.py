"""Phase M on the explicitly chosen frozen311-node/26-feature/23-target data.

The sole temporal override adds live returns to an otherwise AST-identical
method. Existing mask adapter/default D0B and historical sources are untouched.
"""
import math
import copy
import torch
from torch import nn
from cmgm.models.formal_d0b_main_ablation import MainInnovationAblation, masked_softmax, zero
from cmgm.models.phase_m_prediction_moe import (
    ExpertBank, PhaseMRouter, expert_forecasts, checked_weights, preserved_rng,
    copy_shared, tensor_state_hash, KEYS, MODES, EXPERTS,
)
from cmgm.models.horizon_tensor_fusion import predict_by_horizon


class PhaseMLiveBackbone(MainInnovationAblation):
    def temporal(self, x, valid, return_live_sequences=False):
        b = self.switching_latent_transformer
        hidden, tv = self.market_tokens(x, valid)
        bias = b.long_memory.relative_bias(hidden)
        if self.main_name == 'w/o Relative Position Encoding':
            bias = torch.zeros_like(bias)
        length = hidden.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        support = tv[:, None, :, None] & tv[:, None, None, :] & causal
        for layer in b.long_memory.layers:
            attn = layer.attention
            batch, steps, dim = hidden.shape
            q, k, v = [p(hidden).view(batch, steps, attn.n_heads, attn.head_dim).transpose(1, 2)
                        for p in (attn.q, attn.k, attn.v)]
            alpha = masked_softmax(q@k.transpose(-2, -1)/math.sqrt(attn.head_dim)+bias, support, -1)
            output = attn.out((alpha@v).transpose(1, 2).reshape(batch, steps, dim))
            hidden = zero(layer.norm1(hidden+layer.dropout(output)), tv)
            hidden = zero(layer.norm2(hidden+layer.dropout(layer.ffn(hidden))), tv)
        filt = b.regime_filter
        p = x.new_full((x.size(0), b.K), 1/b.K)
        z = x.new_zeros(x.size(0), b.z_dim)
        transition = filt.transition_matrix()
        probabilities, priors, states, candidates_all, losses = [], [], [], [], []
        for i in range(length):
            if self.main_name == 'w/o Adaptive Regime Routing':
                prior = proposed = x.new_full(p.shape, 1/b.K)
                kl = hidden[:, i].sum(-1)*0
            else:
                prior, _, proposed, kl = filt.step(hidden[:, i], p, transition)
            p = torch.where(tv[:, i, None], proposed, p)
            if self.main_name == 'w/o Regime-Specific Transitions':
                shared = b.latent_transition.generators[0](torch.cat([hidden[:, i], z], -1))
                candidates = shared[:, None].expand(-1, b.K, -1)
                proposal = torch.einsum('bk,bkd->bd', p, candidates)
            else:
                proposal, candidates = b.latent_transition(hidden[:, i], z, p)
            z = torch.where(tv[:, i, None], proposal, z)
            probabilities.append(p); priors.append(prior); states.append(z); candidates_all.append(candidates)
            losses.append(torch.where(tv[:, i], kl, 0.))
        filt._last_switch_loss = torch.stack(losses, 1).sum()/tv.sum().clamp_min(1)
        positions = torch.arange(length, device=x.device)[None].expand_as(tv)
        last = positions.masked_fill(~tv, -1).max(1).values
        h_last = hidden[torch.arange(len(x), device=x.device), last.clamp_min(0)]
        result = b.readout(h_last, z, zero_component='Z' if self.main_name == 'w/o Microstate' else None)
        result = zero(result, tv.any(1))
        self.diagnostics.update(regime_p=torch.stack(probabilities, 1).detach(),
            regime_prior=torch.stack(priors, 1).detach(), latent_states=torch.stack(states, 1).detach(),
            long_memory=hidden.detach(),
            candidates=torch.stack(candidates_all, 1).detach(), temporal_valid=tv.detach(),
            long_norm=b.last_h_long.norm(dim=-1).detach(), micro_norm=b.last_h_micro.norm(dim=-1).detach())
        return (result, hidden, torch.stack(states, 1)) if return_live_sequences else result


class NewDataPhaseMModel(nn.Module):
    def __init__(self, control, history_root, **shape):
        super().__init__()
        if control not in KEYS:raise ValueError(control)
        self.control=control;self.variant=MODES[control]
        self.encoder=PhaseMLiveBackbone('Full D0B',history_root,**shape)
        self.n_commodities=self.encoder.targets
        if control!='C0':
            del self.base.gate_fc
            self.experts=ExpertBank()
            self.router=PhaseMRouter(control)

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
        if self.control=='C6':
            fused=torch.einsum('bhk,bkd->bhd',pi,bank)
            prediction=predict_by_horizon(self.base.head,fused,self.n_commodities)
            candidates=expert_forecasts(self.base.head,bank,self.n_commodities) if with_candidates else None
            if with_candidates and self.training:raise ValueError('C6 candidate diagnosis is eval-only')
        else:
            candidates=expert_forecasts(self.base.head,bank,self.n_commodities)
            prediction=torch.einsum('bhk,bhkc->bhc',pi,candidates)
        aux=dict(H=h,Z=z,s=s,t=t,experts=bank,routing=pi,native_routing=native_pi,expert_prediction=candidates)
        return (prediction,aux) if return_aux else prediction


def matched_new_models(history_root,shape,seed=42):
    with preserved_rng():
        torch.manual_seed(seed)
        models={'C0':NewDataPhaseMModel('C0',history_root,**shape)}
        audit={'common':{},'experts':{},'routers':{}}
        for key in KEYS[1:]:
            torch.manual_seed(seed)
            model=NewDataPhaseMModel(key,history_root,**shape)
            keys=[k for k in models['C0'].state_dict() if not k.startswith('encoder.native.gate_fc.')]
            audit['common'][key]=copy_shared(models['C0'],model,keys)
            models[key]=model
        for key in KEYS[2:]:
            keys=[k for k in models['C1'].state_dict() if k.startswith('experts.')]
            audit['experts'][key]=copy_shared(models['C1'],models[key],keys)
        audit['routers']['C6_from_C5']=copy_shared(models['C5'],models['C6'],list(models['C5'].state_dict()))
        a,b=models['C5'].router,models['C4'].router
        keys=[k for k in a.state_dict() if not k.startswith(('network.0.','horizon_embedding.'))]
        audit['routers']['C4_shared_context']=copy_shared(a,b,keys)
        with torch.no_grad():
            b.network[0].weight.copy_(a.network[0].weight[:,:256]);b.network[0].bias.copy_(a.network[0].bias)
        audit['expert_state_hashes']={k:tensor_state_hash(m.experts.state_dict()) for k,m in models.items() if k!='C0'}
        audit['initial_state_hashes']={k:tensor_state_hash(m.state_dict()) for k,m in models.items()}
        audit['C5_C6_identical']=audit['initial_state_hashes']['C5']==audit['initial_state_hashes']['C6']
        audit['PASS']=all(v['PASS'] for group in ('common','experts','routers') for v in audit[group].values()) and audit['C5_C6_identical']
        if not audit['PASS']:raise ValueError('Shared initialization audit failed')
    return models,audit

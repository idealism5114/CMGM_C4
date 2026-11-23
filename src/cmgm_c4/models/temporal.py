"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import MainInnovationAblation,masked_softmax,zero


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

"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from types import SimpleNamespace
from ._native.native import NativeSLBR
from ._native.mixhop import MixHopPropagation

def native_modules(history_root=None):
    return SimpleNamespace(HeteroMixHopCMGM=lambda nodes,targets,n_stock,n_bond,variant,feat_dim: NativeSLBR(nodes,targets,n_stock,n_bond,feat_dim),MixHopPropagation=MixHopPropagation)


NAMES = ('w/o Spatial Branch', 'w/o Temporal Branch', 'w/o Spatial Temporal Attention',
         'w/o Adaptive Graph', 'w/o EdgeAttnMixHop', 'w/o Relative Position Encoding',
         'w/o Adaptive Regime Routing', 'w/o Regime-Specific Transitions',
         'w/o Microstate', 'w/o Balanced Readout', 'w/o Adaptive Fusion', 'Full D0B')


KEY = dict(zip(NAMES, ('no_spatial', 'no_temporal', 'no_spatial_temporal_attention',
    'no_adaptive_graph', 'no_edge_attn_mixhop', 'no_rpe', 'no_adaptive_regime_routing',
    'no_regime_specific_transition', 'no_microstate', 'no_balanced_readout', 'no_adaptive_fusion', 'full_d0b')))


DEFINITIONS = dict(zip(NAMES, (
    'Temporal projection only; original head; native Switch KL retained',
    'Spatial projection only; original head; temporal path and KL inactive',
    'Uniform mean over available historical tokens per node; graph unchanged',
    'A=ones(N,N); native edge attention and both EdgeAttnMixHop blocks retained',
    'Learned A retained; two ordinary MixHopPropagation(64,64,K=2,beta=.05), exactly the Sep17 definition',
    'BaseRPE bias zero; complete causal Transformer retained',
    'p/prior uniform; three distinct generators retained; KL zero',
    'Adaptive Markov p/native KL retained; generator[0] shared across candidates',
    'Post-normalization micro readout zero; recurrence/filter/KL retained',
    'Bypass only long/micro readout LayerNorm; both projections retained',
    'Half temporal projection + half spatial projection; original head',
    'Original native switching_latent_balanced_readout with availability support only')))


def masked_softmax(logits, valid, dim):
    support = torch.broadcast_to(valid, logits.shape)
    scores = logits.masked_fill(~support, torch.finfo(logits.dtype).min)
    p = torch.softmax(scores, dim) * support.to(logits.dtype)
    return p / p.sum(dim, keepdim=True).clamp_min(1e-12)


def zero(x, valid):
    return torch.where(valid[..., None], x, 0.)


class MainInnovationAblation(nn.Module):
    def __init__(self, name, history_root=None, *, nodes=311, stocks=263, bonds=25, targets=23, features=26):
        super().__init__()
        if name not in NAMES:
            raise ValueError(name)
        self.main_name = name
        native = native_modules(history_root)
        # Construct every original shared module before interventions.
        self.native = native.HeteroMixHopCMGM(nodes, targets, n_stock=stocks, n_bond=bonds,
            variant='switching_latent_balanced_readout', feat_dim=features)
        self.variant = 'switching_latent_balanced_readout'
        self.stocks, self.bonds, self.targets, self.features = stocks, bonds, targets, features
        self.disable_switch_kl = name == 'w/o Temporal Branch'
        self.native.switching_latent_transformer.formal_bypass_balance = name == 'w/o Balanced Readout'
        if name == 'w/o EdgeAttnMixHop':
            self.ordinary_mixhop1 = native.MixHopPropagation(64, 64, K=2, beta=.05)
            self.ordinary_mixhop2 = native.MixHopPropagation(64, 64, K=2, beta=.05)
        self.diagnostics = {}

    @property
    def switching_latent_transformer(self):
        return self.native.switching_latent_transformer

    def set_epoch(self, epoch):
        return self.switching_latent_transformer.set_epoch(epoch)

    def switch_loss(self):
        if self.disable_switch_kl:
            return next(self.parameters()).new_zeros(())
        return self.switching_latent_transformer.switch_loss()

    @staticmethod
    def edge_block(block, x, adjacency, valid):
        b, n, _ = x.shape
        hidden = x = zero(x, valid)
        out = zero(block.Ws[0](x), valid)
        prior = torch.log(adjacency.clamp_min(0)+1e-6)
        support = valid[:, :, None, None] & valid[:, None, :, None]
        for k in range(1, block.K+1):
            q, key, value = [layer(hidden).view(b, n, block.n_heads, block.head_dim)
                             for layer in (block.q, block.k, block.v)]
            logits = torch.einsum('bnhd,bmhd->bnmh', q, key)/math.sqrt(block.head_dim)
            alpha = masked_softmax(logits + block.prior_scale*prior[None, :, :, None], support, 2)
            message = torch.einsum('bnmh,bmhd->bnhd', block.attn_drop(alpha), value).reshape(b, n, -1)
            message = zero(block.out_proj(message), valid)
            hidden = zero(block.beta*x+(1-block.beta)*message, valid)
            out = out + zero(block.Ws[k](hidden), valid)
        return out

    @staticmethod
    def ordinary_block(block, x, adjacency, valid):
        n = adjacency.shape[-1]
        support = valid[:, :, None] & valid[:, None, :]
        augmented = adjacency + torch.eye(n, device=x.device, dtype=x.dtype)
        norm = torch.where(support, augmented[None], 0.)
        norm = norm/norm.sum(-1, keepdim=True).clamp_min(1e-8)
        hidden = x = zero(x, valid)
        out = zero(block.Ws[0](x), valid)
        for k in range(1, block.K+1):
            hidden = zero(block.beta*x+(1-block.beta)*(norm@hidden), valid)
            out = out + zero(block.Ws[k](hidden), valid)
        return out

    def spatial(self, x, valid):
        m = self.native
        seq = zero(torch.stack([m.type_proj(x[:, i], self.stocks, self.bonds) for i in range(x.size(1))], 1), valid)
        if self.main_name == 'w/o Spatial Temporal Attention':
            alpha = valid.to(x.dtype)/valid.sum(1, keepdim=True).clamp_min(1)
        else:
            alpha = masked_softmax(m.temporal_score(seq).squeeze(-1), valid, 1)
        pre = (seq*alpha[..., None]).sum(1)
        node_valid = valid.any(1)
        adjacency = pre.new_ones(pre.size(1), pre.size(1)) if self.main_name == 'w/o Adaptive Graph' else m.graph_learner()
        if self.main_name == 'w/o EdgeAttnMixHop':
            h = self.ordinary_block(self.ordinary_mixhop1, pre, adjacency, node_valid)
            h = self.ordinary_block(self.ordinary_mixhop2, F.relu(h), adjacency, node_valid)
        else:
            h = self.edge_block(m.attn_mixhop1, pre, adjacency, node_valid)
            h = self.edge_block(m.attn_mixhop2, F.relu(h), adjacency, node_valid)
        h = zero(m.gcn_norm(h), node_valid)
        pools = []
        for start, end in ((0, self.stocks), (self.stocks, self.stocks+self.bonds), (self.stocks+self.bonds, h.size(1))):
            pools.append(h[:, start:end].sum(1)/node_valid[:, start:end].sum(1, keepdim=True).clamp_min(1))
        pooled = zero(m.type_pool.type_agg(torch.cat(pools, -1)), node_valid.any(-1))
        self.diagnostics['spatial_alpha'] = alpha.detach()
        self.diagnostics['graph_node_valid'] = node_valid.detach()
        self.diagnostics['adjacency'] = adjacency.detach()
        return pooled

    def market_tokens(self, x, valid):
        encoder = self.switching_latent_transformer.market_encoder
        pieces = encoder._split_markets(x)
        pieces_valid = torch.split(valid, (self.stocks, self.bonds, valid.size(2)-self.stocks-self.bonds), 2)
        reps = []
        for (name, (node_encoder, score)), mask in zip(encoder._market_modules().items(), pieces_valid):
            h = zero(node_encoder(pieces[name]), mask)
            alpha = masked_softmax(score(h), mask[..., None], 2)
            level = (h*alpha).sum(2)
            count = mask.sum(2, keepdim=True).clamp_min(1)
            mean = h.sum(2)/count
            variance = torch.where(mask[..., None], (h-mean[:, :, None]).square(), 0.).sum(2)/count
            # Exact sqrt on positive support; finite derivative at zero variance.
            positive = variance > 0
            dispersion = torch.sqrt(torch.where(positive, variance, 1.))*positive
            reps.append(zero(torch.cat([level, dispersion], -1), mask.any(2)))
        temporal_valid = valid.any(2)
        tokens = zero(encoder.daily_projection(torch.cat(reps, -1)), temporal_valid)
        return tokens, temporal_valid

    def temporal(self, x, valid):
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
        return result

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask):
        if X.ndim != 4 or X.shape[-1] != self.features:
            raise ValueError('B,T,N,F required')
        if feature_applicable_mask.ndim == 2:
            feature_applicable_mask = feature_applicable_mask[None, None]
        elif feature_applicable_mask.ndim == 3:
            feature_applicable_mask = feature_applicable_mask[:, None]
        fm = feature_valid_mask & feature_applicable_mask & input_valid_mask[..., None]
        x = torch.where(fm, X, 0.)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite valid features')
        valid = input_valid_mask & fm.any(-1)
        self.diagnostics = {}
        if self.main_name != 'w/o Spatial Branch':
            hs = self.spatial(x, valid)
        if self.main_name != 'w/o Temporal Branch':
            ht = self.temporal(x, valid)
        if self.main_name == 'w/o Spatial Branch':
            fused = self.native.lstm_proj(ht)
        elif self.main_name == 'w/o Temporal Branch':
            fused = self.native.gcn_proj(hs)
        else:
            gate = hs.new_full(hs.shape, .5) if self.main_name == 'w/o Adaptive Fusion' else torch.sigmoid(self.native.gate_fc(torch.cat([hs, ht], -1)))
            fused = gate*self.native.lstm_proj(ht)+(1-gate)*self.native.gcn_proj(hs)
        return self.native.head(fused).reshape(len(X), 4, self.targets)

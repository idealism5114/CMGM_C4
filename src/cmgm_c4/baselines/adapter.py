"""Parameter-free, validity-aware Deep V3 input view for imported TriWeave data.

The original V3 view is retained: stock mean/std, bond mean/std, commodity
tokens. This is a compressed representation, unlike D0B's full-node encoder.
No future filling or fitted transformation is performed here. Invalid feature
cells become neutral zero only *after* exclusion from market statistics.
The V3 cores have no missing-token attention interface; their subsequent latent
states are not claimed to implement D0B's graph availability masking.
"""
import torch
from torch import nn
from .models import make_model


class MaskedNeutralAdapter(nn.Module):
    def __init__(self, market_indices, features):
        super().__init__()
        self.market_indices = market_indices
        self.features = features

    def forward(self, x, input_valid_mask, feature_valid_mask, feature_applicable_mask):
        if x.ndim != 4 or x.shape[-1] != self.features:
            raise ValueError('Expected B,T,N,F input')
        app = feature_applicable_mask
        if app.ndim == 2:
            app = app[None, None]
        elif app.ndim == 3:
            app = app[:, None]
        valid = input_valid_mask[..., None] & feature_valid_mask & app
        values = torch.where(valid, x, 0.)
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite valid input')
        tokens = []
        for market in ('stock', 'bond'):
            lo, hi = self.market_indices[market]
            v, m = values[:, :, lo:hi], valid[:, :, lo:hi]
            count = m.sum(2).clamp_min(1)
            mean = v.sum(2) / count
            variance = torch.where(m, (v-mean[:, :, None]).square(), 0.).sum(2) / count
            # A zero dispersion has zero derivative, including empty support.
            sd = torch.where(variance > 0, variance.clamp_min(1e-30).sqrt(), 0.)
            tokens.extend((mean, sd))
        lo, hi = self.market_indices['commodity']
        return torch.cat([torch.stack(tokens, 2), values[:, :, lo:hi]], 2)


class NeutralIdentity(nn.Module):
    def forward(self, x):
        return x

    def sequence(self, x):
        return x.flatten(2)


def prepared_model(name, commodities=23, features=26):
    """Reuse the actual current V3 core, without summarizing cached tokens twice."""
    model = make_model(name, dict(stock=(0, 2), bond=(2, 4),
                                commodity=(4, 4+commodities)), features=features)
    model.adapter = NeutralIdentity()
    return model

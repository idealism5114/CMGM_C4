"""C4 with a shared expert prior and horizon-specific conditional logits."""
import torch
from torch import nn

from .ablations import C4Ablation
from .experts import expert_forecasts
from .router import C4Router


BRANCHES = {
    'full': 'Full C4（分周期路由）',
    'no_temporal': '去时间分支',
    'no_spatial': '去空间分支',
    'no_moe': '去 MoE（双分支等权融合）',
}


class HorizonRouter(C4Router):
    def __init__(self, original):
        super().__init__()
        self.load_state_dict(original.state_dict())
        self.global_logits = nn.Parameter(torch.zeros(4))
        self.network[-1] = nn.Linear(32, 16)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, spatial, temporal, hidden, latent):
        temporal_only = spatial is None
        if temporal_only:
            spatial = torch.zeros_like(temporal)
        context = self.context(spatial, temporal, hidden, latent)
        if temporal_only:
            context = torch.cat((torch.zeros_like(spatial), context[:, spatial.size(-1):]), dim=-1)
        logits = self.global_logits + self.network(context).reshape(-1, 4, 4)
        if temporal_only:
            logits = logits.masked_fill(logits.new_tensor([False, False, True, True], dtype=torch.bool), -torch.inf)
        return logits.softmax(dim=-1)


class HorizonC4(C4Ablation):
    def __init__(self, row, shape, inactive, initial_state=None):
        if row not in BRANCHES:
            raise ValueError(row)
        super().__init__('full_c4' if row in ('full', 'no_moe') else row,
                         initial_state=initial_state, **shape)
        self.variant = 'c4v2/' + row
        if row in ('no_temporal', 'no_spatial'):
            self.freeze_inactive(inactive[row])
        if row == 'no_moe':
            # The backbone's equal-fusion path feeds the same head directly.
            self.control = 'C0'
            self.encoder.main_name = 'w/o Adaptive Fusion'
            del self.experts, self.router
        elif row == 'no_temporal':
            del self.router
        else:
            self.router = HorizonRouter(self.router)
            if row == 'no_spatial':
                self.router.norm_s.requires_grad_(False)

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask,
                *, return_aux=False, routing_override=None, with_candidates=False):
        if self.ablation != 'no_spatial':
            return super().forward(X, input_valid_mask, feature_valid_mask, feature_applicable_mask,
                                   return_aux=return_aux, routing_override=routing_override,
                                   with_candidates=with_candidates)
        if routing_override is not None:
            raise ValueError('Temporal-only routing requires the two temporal experts')
        app = feature_applicable_mask
        if app.ndim == 3:
            app = app[:, None]
        mask = feature_valid_mask & app & input_valid_mask[..., None]
        x = torch.where(mask, X, 0.)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite valid input')
        self.encoder.diagnostics = {}
        ht, h, z = self.encoder.temporal(x, mask.any(-1), return_live_sequences=True)
        t = self.base.lstm_proj(ht)
        weights = self.router(None, t, h, z)
        recent = self.experts.recent_change(z)
        empty = torch.zeros_like(recent)
        bank = torch.stack((recent, self.experts.long_memory(h), empty, empty), dim=1)
        candidates = expert_forecasts(self.base.head, bank, self.n_commodities)
        prediction = torch.einsum('bhk,bhkc->bhc', weights, candidates)
        aux = dict(H=h, Z=z, t=t, s=None, experts=bank, expert_prediction=candidates,
                   routing=weights, native_routing=weights)
        return (prediction, aux) if return_aux else prediction

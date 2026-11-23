"""Sep17 mechanism interventions transplanted onto the frozen Phase M C4 design.

Only ablation implementations: no changes to original D0B, C4 or their runners.
Whole-branch removals disable experts whose required branch is absent. The old
microstate ablation nulls only the balanced final readout, not live Z experts.
"""
import torch

from .c4 import NewDataPhaseMModel
from .backbone import NAMES, KEY, DEFINITIONS, native_modules
from .experts import expert_forecasts

BONE = tuple(KEY[n] for n in NAMES[:-2])
KEYS = ('full_c4',) + BONE + ('no_dynamic_routing', 'no_moe')
LABELS = dict(zip(BONE, NAMES[:-2]))
LABELS.update(full_c4='Full C4 Global Dynamic Prediction MoE',
    no_dynamic_routing='w/o Dynamic Routing (uniform experts)', no_moe='w/o MoE (original D0B gate)')
DEFS = {KEY[n]: DEFINITIONS[n] for n in NAMES[:-2]}
DEFS.update(full_c4='Exact Phase M C4, reused seed42 best epoch35; prediction-level four-expert mixture',
    no_spatial='Remove graph spatial encoder/projection; disable spatial and ST experts; recent/long experts remain with masked two-candidate dynamic routing',
    no_temporal='Remove switching temporal encoder, KL and temporal-dependent experts/router; spatial expert through the same shared head',
    no_microstate='Exact Sep17 post-normalization microstate null in final balanced readout; live Z recurrence AND recent-Z expert retained',
    no_dynamic_routing='All four experts retained; fixed weights 1/4; dynamic context/router inactive (frozen and excluded from optimizer)',
    no_moe='Exact original D0B channel gate and shared head; no experts/router; reuse Phase M C0 seed42 best epoch46')


class C4Ablation(NewDataPhaseMModel):
    def __init__(self, ablation, history_root=None, *, initial_state=None, **shape):
        if ablation not in KEYS:
            raise ValueError(ablation)
        super().__init__('C0' if ablation == 'no_moe' else 'C4', history_root, **shape)
        # Shared random initial state comes from the completed Phase M INITIAL
        # states, never from a trained checkpoint. New MixHop modules come later.
        if initial_state is not None:
            self.load_state_dict(initial_state, strict=True)
        self.ablation = ablation
        self.variant = 'moe_phase_m/c4_ablation/'+ablation
        if ablation in BONE:
            self.encoder.main_name = LABELS[ablation]
        self.encoder.disable_switch_kl = ablation == 'no_temporal'
        self.switching_latent_transformer.formal_bypass_balance = ablation == 'no_balanced_readout'
        if ablation == 'no_edge_attn_mixhop':
            native = native_modules(history_root)
            self.encoder.ordinary_mixhop1 = native.MixHopPropagation(64, 64, K=2, beta=.05)
            self.encoder.ordinary_mixhop2 = native.MixHopPropagation(64, 64, K=2, beta=.05)

    def freeze_inactive(self, names):
        params = dict(self.named_parameters())
        for name in names:
            params[name].requires_grad_(False)

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask,
                *, return_aux=False, routing_override=None, with_candidates=False):
        if self.ablation not in ('no_spatial', 'no_temporal'):
            if self.ablation == 'no_dynamic_routing' and routing_override is None:
                routing_override = X.new_full((len(X), 4, 4), .25)
            return super().forward(X, input_valid_mask, feature_valid_mask, feature_applicable_mask,
                return_aux=return_aux, routing_override=routing_override, with_candidates=with_candidates)
        if routing_override is not None:
            raise ValueError('Branch-removal routing is explicitly masked; no external override')
        app = feature_applicable_mask
        if app.ndim == 2:
            app = app[None, None]
        elif app.ndim == 3:
            app = app[:, None]
        mask = feature_valid_mask & app & input_valid_mask[..., None]
        x = torch.where(mask, X, 0.)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite valid input')
        valid = input_valid_mask & mask.any(-1)
        self.encoder.diagnostics = {}
        if self.ablation == 'no_spatial':
            ht, h, z = self.encoder.temporal(x, valid, return_live_sequences=True)
            t = self.base.lstm_proj(ht)
            s = torch.zeros_like(t)
            e1, e2 = self.experts.recent_change(z), self.experts.long_memory(h)
            bank = torch.stack((e1, e2, torch.zeros_like(e1), torch.zeros_like(e1)), 1)
            context = self.router.context(s, t, h, z)
            # No spatial content/bias from its normalization may enter routing.
            context = torch.cat((torch.zeros_like(context[:, :64]), context[:, 64:]), -1)
            logits = self.router.network(context).masked_fill(
                torch.tensor([False, False, True, True], device=x.device), float('-inf'))
            pi = logits.softmax(-1)[:, None].expand(-1, 4, -1)
        else:
            hs = self.encoder.spatial(x, valid)
            s = self.base.gcn_proj(hs)
            e = self.experts.spatial_structure(s)
            bank = torch.stack((torch.zeros_like(e), torch.zeros_like(e), e, torch.zeros_like(e)), 1)
            pi = x.new_zeros(len(x), 4, 4)
            pi[:, :, 2] = 1
            h = z = t = None
        candidates = expert_forecasts(self.base.head, bank, self.n_commodities)
        prediction = torch.einsum('bhk,bhkc->bhc', pi, candidates)
        aux = dict(H=h, Z=z, s=s, t=t, experts=bank, routing=pi,
                   native_routing=pi, expert_prediction=candidates)
        return (prediction, aux) if return_aux else prediction

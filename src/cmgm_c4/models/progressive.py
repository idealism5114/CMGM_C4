"""Matched spatial, temporal and expert-routing ablations of C4."""
import torch
from torch import nn

ROWS = {
    'A': '仅空间专家',
    'B': '空间、近期、长期三专家等权',
    'C': '四专家等权',
    'D': '四专家全局可学习权重',
    'E': 'Full C4 动态权重',
}


class SharedWeights(nn.Module):
    def __init__(self, row):
        super().__init__()
        if row == 'D':
            self.logits = nn.Parameter(torch.zeros(4))
        else:
            weights = [1/3, 1/3, 1/3, 0.] if row == 'B' else [0.25]*4
            self.register_buffer('weights', torch.tensor(weights))

    def forward(self, spatial, temporal, hidden, latent):
        weights = self.logits.softmax(-1) if hasattr(self, 'logits') else self.weights
        return weights.expand(len(spatial), 4, 4)


class OmittedInteraction(nn.Module):
    def forward(self, spatial, temporal):
        return torch.zeros_like(spatial)


def configure(model, row, inactive):
    if row == 'A':
        model.freeze_inactive(inactive['no_temporal'])
    if row in ('B', 'C', 'D'):
        model.router = SharedWeights(row)
    if row == 'B':
        model.experts.st_interaction = OmittedInteraction()
        # B reads live H and Z directly. Only the final temporal projection
        # and balanced readout lose their paths to the prediction loss.
        model.base.lstm_proj.requires_grad_(False)
        temporal = model.switching_latent_transformer
        for name in ('long_memory_readout', 'micro_state_readout',
                     'long_memory_norm', 'micro_state_norm', 'state_readout'):
            getattr(temporal, name).requires_grad_(False)
    return model

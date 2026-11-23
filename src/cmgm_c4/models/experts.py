"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LongMemoryExpert(nn.Module):
    """ReLU(W[H_last, mean(H_all_20)]), from live H (B,T,128)."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 64)

    def forward(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (20, 128):
            raise ValueError("Long expert requires live (B,20,128) long memory")
        return F.relu(self.projection(torch.cat((memory[:, -1], memory.mean(dim=1)), dim=-1)))


class SpatialStructureExpert(nn.Module):
    """s + ReLU(W_s s), where s is the original projected spatial vector."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Linear(64, 64)

    def forward(self, spatial):
        return spatial + F.relu(self.refine(spatial))


class LowRankInteractionExpert(nn.Module):
    """t + W_o[(U_s s) ⊙ (U_t t)], with rank 16 and no extra gate."""

    def __init__(self):
        super().__init__()
        self.spatial_rank = nn.Linear(64, 16, bias=False)
        self.temporal_rank = nn.Linear(64, 16, bias=False)
        self.output = nn.Linear(16, 64, bias=False)

    def forward(self, spatial, temporal):
        return temporal + self.output(self.spatial_rank(spatial) * self.temporal_rank(temporal))


class MeanAugmentedRecentExpert(nn.Module):
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

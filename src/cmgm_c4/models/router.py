"""The original C4 global dynamic router only; four horizons share its weights."""
import torch
from torch import nn

class C4Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.long_projection=nn.Linear(128,64)
        self.norm_s=nn.LayerNorm(64);self.norm_t=nn.LayerNorm(64)
        self.norm_change=nn.LayerNorm(64);self.norm_long=nn.LayerNorm(64)
        self.network=nn.Sequential(nn.Linear(256,32),nn.GELU(),nn.Linear(32,4))
        nn.init.normal_(self.network[-1].weight,mean=0.,std=1e-3);nn.init.zeros_(self.network[-1].bias)

    def context(self,s,t,h,z):
            change=z[:,-1]-z[:,-5:].mean(1)
            return torch.cat((self.norm_s(s),self.norm_t(t),self.norm_change(change),
                              self.norm_long(self.long_projection(h.mean(1)))), -1)

    def forward(self,s,t,h,z):
        return self.network(self.context(s,t,h,z)).softmax(-1)[:,None].expand(-1,4,-1)

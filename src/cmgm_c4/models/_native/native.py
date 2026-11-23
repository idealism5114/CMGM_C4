"""Original D0B module inventory, no experimental variant registry."""
from torch import nn
from .spatial import _TypeInputProjection,EdgeAttnMixHop,_TypeMeanPool
from .graph import AdaptiveGraphLearner
from .temporal import SwitchingLatentTransformerBranch

class NativeSLBR(nn.Module):
    def __init__(self,num_nodes,n_commodities,n_stock=263,n_bond=25,feat_dim=26):
        super().__init__()
        self.num_nodes=num_nodes;self.n_commodities=n_commodities;self.n_stock=n_stock;self.n_bond=n_bond
        self.feat_dim=feat_dim;self.n_horizons=4;self.variant='switching_latent_balanced_readout'
        self.type_proj=_TypeInputProjection(feat_dim,64)
        self.graph_learner=AdaptiveGraphLearner(num_nodes,embed_dim=10,alpha=.5,top_k=10)
        self.attn_mixhop1=EdgeAttnMixHop(64,64,K=2,beta=.05,n_heads=8,dropout=.1,hard_mask=False,prior_scale=.5,self_heads=0,cross_mask=None,graph_prior_heads=None,qk_norm=False)
        self.attn_mixhop2=EdgeAttnMixHop(64,64,K=2,beta=.05,n_heads=8,dropout=.1,hard_mask=False,prior_scale=.5,self_heads=0,cross_mask=None,graph_prior_heads=None,qk_norm=False)
        self.gcn_norm=nn.LayerNorm(64)
        self.type_pool=_TypeMeanPool(64,n_stock,n_bond)
        # Gate is constructed at the original point to preserve random draws;
        # C4 deletes it after constructing the entire encoder, as before.
        self.gate_fc=nn.Linear(128,64)
        self.gcn_proj=nn.Linear(64,64);self.lstm_proj=nn.Linear(64,64)
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,4*n_commodities))
        self.temporal_score=nn.Linear(64,1)
        self.switching_latent_transformer=SwitchingLatentTransformerBranch(
            feat_dim=feat_dim,n_stock=n_stock,n_bond=n_bond,n_commodity=num_nodes-n_stock-n_bond,
            node_dim=32,d_model=128,n_heads=4,n_layers=2,ffn_dim=256,dropout=.3,max_len=20,
            K=3,z_dim=64,sticky_alpha=.5,tau=1.,beta_max=5e-4,warmup_epochs=20,
            output_dim=64,balanced_readout=True,forecast_horizons=(1,5,10,20))

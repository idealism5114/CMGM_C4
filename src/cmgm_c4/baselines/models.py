"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .graph import AdaptiveGraphLearner
from .mixhop import MixHopPropagation


HORIZONS=(1,5,10,20)


TRAINABLE=('RNN','GRU','LSTM','VanillaTransformer','GraphWaveNet','MTGNN','MSGNet','CrossGNN')


ORDER=TRAINABLE


class BaselineMultimodalAdapter(nn.Module):
    """Population std (correction=0); token-major then feature-major flatten."""
    def __init__(self,n_stock=248,n_bond=12,n_commodities=24,features=21):
        super().__init__()
        self.n_stock=n_stock;self.n_bond=n_bond;self.n_commodities=n_commodities;self.features=features
        if min(n_stock,n_bond,n_commodities,features)<1:raise ValueError('Expected nonempty markets, commodity and feature axes')
    def forward(self,x):
        if x.ndim!=4 or x.shape[2:]!=(self.n_stock+self.n_bond+self.n_commodities,self.features):raise ValueError('Expected (B,T,N,21) with declared market ordering')
        stock=x[:,:,:self.n_stock];bond=x[:,:,self.n_stock:self.n_stock+self.n_bond];comm=x[:,:,self.n_stock+self.n_bond:]
        summaries=torch.stack([stock.mean(2),stock.std(2,unbiased=False),bond.mean(2),bond.std(2,unbiased=False)],dim=2)
        return torch.cat([summaries,comm],dim=2)
    def sequence(self,x):return self(x).flatten(2)


class Base(nn.Module):
    def __init__(self,**market):
        super().__init__();self.adapter=BaselineMultimodalAdapter(**market)
        self.n_commodities=self.adapter.n_commodities;self.features=self.adapter.features
        self.neutral_nodes=4+self.n_commodities;self.sequence_dim=self.neutral_nodes*self.features
    def check(self,p,b):
        if p.shape!=(b,4,self.n_commodities):raise ValueError(f'Wrong baseline output shape {p.shape}')
        return p


class GRUBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.gru=nn.GRU(self.sequence_dim,128,num_layers=2,dropout=.1,batch_first=True);self.head=nn.Linear(128,4*self.n_commodities)
    def temporal_states(self,x):return self.gru(self.adapter.sequence(x))[0]
    def forward(self,x):
        _,hidden=self.gru(self.adapter.sequence(x))
        return self.check(self.head(hidden[-1]).view(len(x),4,self.n_commodities),len(x))


class VanillaTransformerBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.input_projection=nn.Linear(self.sequence_dim,128)
        position=torch.arange(20,dtype=torch.float32)[:,None];frequency=torch.exp(torch.arange(0,128,2)*(-math.log(10000.)/128))
        pe=torch.zeros(20,128);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position_encoding',pe)
        layer=nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(128,4*self.n_commodities)
    def temporal_states(self,x):
        h=self.input_projection(self.adapter.sequence(x))+self.position_encoding[:x.shape[1]]
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        return self.encoder(h,mask=mask)
    def forward(self,x):return self.check(self.head(self.temporal_states(x)[:,-1]).view(len(x),4,self.n_commodities),len(x))


class GraphTemporalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.left_padding=2*dilation
        self.filter=nn.Conv2d(64,64,(1,3),dilation=(1,dilation))
        self.gate=nn.Conv2d(64,64,(1,3),dilation=(1,dilation))
        self.forward_graph=MixHopPropagation(64,64,K=2,beta=.05)
        self.reverse_graph=MixHopPropagation(64,64,K=2,beta=.05)
    def forward(self,x,A):
        # x=(B,T,N,C); padding touches time only, never batch/node.
        padded=F.pad(x.permute(0,3,2,1),(self.left_padding,0,0,0))
        temporal=(torch.tanh(self.filter(padded))*torch.sigmoid(self.gate(padded))).permute(0,3,2,1)
        return F.relu(x+self.forward_graph(temporal,A)+self.reverse_graph(temporal,A.T))


class MTGNNBaseline(Base):
    """Simplified MTGNN-style baseline with reused independent graph primitives."""
    def __init__(self,**market):
        super().__init__(**market);self.input_projection=nn.Linear(self.features,64)
        self.graph=AdaptiveGraphLearner(self.neutral_nodes,embed_dim=10,alpha=.5,top_k=10)
        self.blocks=nn.ModuleList([GraphTemporalBlock(1),GraphTemporalBlock(2)]);self.head=nn.Linear(64,4)
    def temporal_states(self,x):
        h=self.input_projection(self.adapter(x));A=self.graph()
        for block in self.blocks:h=block(h,A)
        return h
    def forward(self,x):return self.check(self.head(self.temporal_states(x)[:,-1,4:]).permute(0,2,1),len(x))


class RNNBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.rnn=nn.RNN(self.sequence_dim,128,num_layers=2,nonlinearity='tanh',dropout=.1,batch_first=True);self.head=nn.Linear(128,4*self.n_commodities)
    def temporal_states(self,x):return self.rnn(self.adapter.sequence(x))[0]
    def forward(self,x):
        _,h=self.rnn(self.adapter.sequence(x))
        return self.check(self.head(h[-1]).view(len(x),4,self.n_commodities),len(x))


class LSTMBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.lstm=nn.LSTM(self.sequence_dim,128,num_layers=2,dropout=.1,batch_first=True);self.head=nn.Linear(128,4*self.n_commodities)
    def temporal_states(self,x):return self.lstm(self.adapter.sequence(x))[0]
    def forward(self,x):
        _,(h,_)=self.lstm(self.adapter.sequence(x))
        return self.check(self.head(h[-1]).view(len(x),4,self.n_commodities),len(x))


class GraphWaveNetBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market)
        from .graph_wavenet import GraphWaveNetCore
        self.core=GraphWaveNetCore(num_nodes=self.neutral_nodes,input_dim=self.features)
    def temporal_states(self,x):return self.core(self.adapter(x).permute(0,3,2,1)).permute(0,3,2,1)
    def forward(self,x):return self.check(self.temporal_states(x)[:,-1,4:].transpose(1,2),len(x))
    def mechanism_sanity(self):
        a=self.core.adjacency().detach()
        return dict(adjacency_shape=list(a.shape),adjacency_finite=bool(torch.isfinite(a).all()),
                    row_sum_max_error=float((a.sum(-1)-1).abs().max()),
                    PASS=a.shape==(self.neutral_nodes,self.neutral_nodes) and bool(torch.isfinite(a).all()) and float((a.sum(-1)-1).abs().max())<1e-6)


class MSGNetBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market)
        from .msgnet import MSGNetCore
        self.node_value_projection=nn.Linear(self.features,1);self.core=MSGNetCore(num_nodes=self.neutral_nodes)
    def scalar_input(self,x):return self.node_value_projection(self.adapter(x)).squeeze(-1)
    def forward(self,x):
        scalar=self.scalar_input(x);self.last_scalar_shape=list(scalar.shape)
        return self.check(self.core(scalar)[:,:,4:],len(x))
    def mechanism_sanity(self):
        c=self.core.last;weights=c['scale_weights'];freq=c['frequencies']
        finite=all(bool(torch.isfinite(v).all()) for v in c.values())
        graphs=all(bool(torch.isfinite(g.last_adjacency).all()) for g in self.core.graphs)
        return dict(scalar_shape=self.last_scalar_shape,frequencies=freq.cpu().tolist(),periods=c['periods'].cpu().tolist(),
                    fft_finite=finite,graph_adjacencies_finite=graphs,scale_weight_sum_max_error=float((weights.sum(-1)-1).abs().max()),
                    PASS=finite and graphs and freq.shape[1]==3 and bool((freq>0).all()) and float((weights.sum(-1)-1).abs().max())<1e-6)


class CrossGNNBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market)
        from .crossgnn import CrossGNNCore
        self.node_value_projection=nn.Linear(self.features,1);self.core=CrossGNNCore(num_nodes=self.neutral_nodes)
    def scalar_input(self,x):return self.node_value_projection(self.adapter(x)).squeeze(-1)
    def forward(self,x):
        scalar=self.scalar_input(x);self.last_scalar_shape=list(scalar.shape)
        return self.check(self.core(scalar)[:,:,4:],len(x))
    def mechanism_sanity(self):
        c=self.core.last;finite=all(bool(torch.isfinite(v).all()) for v in c.values())
        return dict(scalar_shape=self.last_scalar_shape,periods=c['periods'].cpu().tolist(),
                    multi_scale_shape=list(c['multi_scale'].shape),time_adjacency_shape=list(c['time_adjacency'].shape),
                    variable_adjacency_shape=list(c['variable_adjacency'].shape),all_graphs_and_states_finite=finite,
                    anti_ood=self.core.anti_ood,PASS=finite and not self.core.anti_ood and bool((c['periods']>=1).all()))


MODELS=dict(RNN=RNNBaseline,GRU=GRUBaseline,LSTM=LSTMBaseline,VanillaTransformer=VanillaTransformerBaseline,
            GraphWaveNet=GraphWaveNetBaseline,MTGNN=MTGNNBaseline,MSGNet=MSGNetBaseline,CrossGNN=CrossGNNBaseline)


def make_model(name,market_indices,features=21):
    if name not in MODELS:raise ValueError('Only the eight prespecified deep V3 baselines are supported')
    return MODELS[name](n_stock=market_indices['stock'][1]-market_indices['stock'][0],n_bond=market_indices['bond'][1]-market_indices['bond'][0],n_commodities=market_indices['commodity'][1]-market_indices['commodity'][0],features=features)

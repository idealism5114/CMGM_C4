"""Graph WaveNet core adaptation; see ADAPTATION_NOTES.md for causal normalization choices."""
import torch
from torch import nn
from torch.nn import functional as F


class DiffusionGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Conv2d(3*32,32,1)
        self.dropout=nn.Dropout(.1)

    def forward(self,x,a):
        first=torch.einsum('bcnt,nm->bcmt',x,a)
        second=torch.einsum('bcnt,nm->bcmt',first,a)
        return self.dropout(self.projection(torch.cat([x,first,second],1)))


class WaveLayer(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.left_padding=dilation
        self.filter=nn.Conv2d(32,32,(1,2),dilation=(1,dilation))
        self.gate=nn.Conv2d(32,32,(1,2),dilation=(1,dilation))
        self.skip=nn.Conv2d(32,256,1);self.graph=DiffusionGraph()

    def forward(self,x,a):
        padded=F.pad(x,(self.left_padding,0,0,0))
        z=self.filter(padded).tanh()*self.gate(padded).sigmoid()
        return x+self.graph(z,a),self.skip(z)


class GraphWaveNetCore(nn.Module):
    """B,F,N,T -> B,4,N,T, length-preserving left-causal blocks."""
    def __init__(self,num_nodes=28,input_dim=21):
        super().__init__();self.num_nodes=num_nodes;self.input_dim=input_dim;self.receptive_field=13
        self.nodevec1=nn.Parameter(torch.randn(num_nodes,10));self.nodevec2=nn.Parameter(torch.randn(10,num_nodes))
        self.start=nn.Conv2d(input_dim,32,1)
        self.layers=nn.ModuleList([WaveLayer(d) for _ in range(4) for d in (1,2)])
        self.end=nn.Sequential(nn.ReLU(),nn.Conv2d(256,512,1),nn.ReLU(),nn.Conv2d(512,4,1))

    def adjacency(self):return F.relu(self.nodevec1@self.nodevec2).softmax(-1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1:3]!=(self.input_dim,self.num_nodes):raise ValueError('Graph WaveNet expects B,21,28,T')
        a=self.adjacency();h=self.start(x);skip=None
        for layer in self.layers:
            h,s=layer(h,a);skip=s if skip is None else skip+s
        self.last_adjacency=a.detach()
        return self.end(skip)

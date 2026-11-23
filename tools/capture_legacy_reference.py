"""One-time read-only synthetic reference; never imported by the new runtime."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'Commedities'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from cmgm_c4.common import seed_all,tensor_hash,sha,save
sys.path.insert(0,str(OLD))
from cmgm.models.phase_m_new_data import NewDataPhaseMModel

torch.set_num_threads(4)
HIST=OLD/'experiments/sep17_d0b_triweave/20261022_153322_689200/historical_source'
shape=dict(nodes=311,stocks=263,bonds=25,targets=23,features=26)
seed_all(42)
model=NewDataPhaseMModel('C4',HIST,**shape)
constructor_hash=tensor_hash(model.state_dict())
initial=torch.load(ROOT/'initial_states/c4.pt',map_location='cpu',weights_only=True)['C4']
model.load_state_dict(initial,strict=True)
seed_all(11012)
x=torch.randn(2,20,311,26)
iv=torch.ones(2,20,311,dtype=torch.bool);iv[:,:6,:3]=False
fv=torch.ones_like(x,dtype=torch.bool);fv[:,:4,:,2]=False
batch=dict(X=x,input_valid_mask=iv,feature_valid_mask=fv,feature_applicable_mask=torch.ones(311,26,dtype=torch.bool))
target=torch.randn(2,4,23)*.02
outputs={}
for mode in ('eval','train'):
    model.train(mode=='train');model.set_epoch(20);model.zero_grad(set_to_none=True);seed_all(11013)
    p,a=model(**batch,return_aux=True)
    pred=sum(torch.nn.functional.huber_loss(p[:,h],target[:,h],delta=.02) for h in range(4))
    kl=model.switch_loss();(pred+kl).backward()
    outputs[mode]=dict(prediction=p.detach(),routing=a['routing'].detach(),expert_prediction=a['expert_prediction'].detach(),
        H=a['H'].detach(),Z=a['Z'].detach(),prediction_loss=pred.detach(),KL=kl.detach(),rng=torch.get_rng_state(),
        gradients={n:v.grad.detach().clone() if v.grad is not None else None for n,v in model.named_parameters()})
path=ROOT/'provenance/legacy_synthetic_reference.pt'
if path.exists():raise ValueError('Do not overwrite the original reference')
torch.save(dict(constructor_hash=constructor_hash,initial_hash=tensor_hash(initial),batch=batch,target=target,outputs=outputs),path)
save(ROOT/'provenance/legacy_synthetic_reference.json',dict(source=str(OLD),sha256=sha(path),
    constructor_hash=constructor_hash,initial_hash=tensor_hash(initial),optimizer_created=False,TEST_numeric_reads=0,
    description='Synthetic CPU eval/train forward/backward, original code; no parameter updates'))
print('Saved read-only original reference',path)

import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from cmgm.models.phase_m_new_data import NewDataPhaseMModel,matched_new_models
from cmgm.models.phase_m_prediction_moe import KEYS,expert_forecasts,tensor_state_hash
from cmgm.models.formal_d0b_main_ablation import MainInnovationAblation
from cmgm.scripts.phase_m_new_data import temporal_parity,TrainValDataset
from cmgm.scripts.formal_main_innovation_ablation import prediction_loss

HISTORY=Path(__file__).resolve().parents[1]/'experiments/sep17_d0b_triweave/20261022_153322_689200/historical_source'
SHAPE=dict(nodes=29,stocks=4,bonds=2,targets=23,features=26)

def batch(n=3):
 x=torch.randn(n,20,29,26)
 return dict(X=x,input_valid_mask=torch.ones(n,20,29,dtype=torch.bool),feature_valid_mask=torch.ones_like(x,dtype=torch.bool),feature_applicable_mask=torch.ones(29,26,dtype=torch.bool))

@pytest.fixture
def models():return matched_new_models(HISTORY,SHAPE)[0]

def test_temporal_return_only_and_c0_identical_to_frozen_model(models):
 assert temporal_parity()
 m=models['C0'].eval();old=MainInnovationAblation('Full D0B',HISTORY,**SHAPE).eval()
 old.load_state_dict(m.encoder.state_dict(),strict=True)
 b=batch();torch.testing.assert_close(m(**b),old(**b),atol=0,rtol=0)
 b['input_valid_mask'][:,:8,:3]=False
 torch.testing.assert_close(m(**b),old(**b),atol=0,rtol=0)

def test_shared_initialization_rng_and_no_unused_gate():
 torch.manual_seed(42);rng=torch.get_rng_state().clone()
 ms,a=matched_new_models(HISTORY,SHAPE)
 assert torch.equal(rng,torch.get_rng_state()) and a['PASS'] and a['C5_C6_identical']
 assert len(set(a['expert_state_hashes'].values()))==1
 for key,m in ms.items():assert hasattr(m.base,'gate_fc')==(key=='C0')
 assert len({next(ms[k].experts.parameters()).data_ptr() for k in KEYS[1:]})==6

@pytest.mark.parametrize('key',KEYS)
def test_all_controls_shapes_live_gradients_counts_and_roundtrip(key,models,tmp_path):
 m=models[key].eval();b=batch();y=torch.randn(3,4,23)*.02
 calls={'spatial':0,'temporal':0,'body':0}
 spatial=m.encoder.spatial;temporal=m.encoder.temporal
 def s(*a,**k):calls['spatial']+=1;return spatial(*a,**k)
 def t(*a,**k):calls['temporal']+=1;return temporal(*a,**k)
 hook=m.base.head[0].register_forward_hook(lambda *a:calls.__setitem__('body',calls['body']+1))
 with patch.object(m.encoder,'spatial',side_effect=s),patch.object(m.encoder,'temporal',side_effect=t):p,aux=m(**b,return_aux=True)
 hook.remove();assert calls==dict(spatial=1,temporal=1,body=1)
 assert p.shape==(3,4,23) and torch.isfinite(p).all()
 if key!='C0':
  assert aux['H'].shape==(3,20,128) and aux['Z'].shape==(3,20,64)
  assert aux['H'].requires_grad and aux['Z'].requires_grad
  aux['H'].retain_grad();aux['Z'].retain_grad()
  assert aux['routing'].shape==(3,4,4);torch.testing.assert_close(aux['routing'].sum(-1),torch.ones(3,4))
  if key!='C6':assert aux['expert_prediction'].shape==(3,4,4,23)
 loss=prediction_loss(p,y,torch.ones_like(y,dtype=torch.bool));loss.backward()
 assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
 if key!='C0':
  assert aux['H'].grad.abs().sum()>0 and aux['Z'].grad.abs().sum()>0
  assert any(v.grad.abs().sum()>0 for v in m.router.parameters())
  for module in m.experts.children():assert any(v.grad.abs().sum()>0 for v in module.parameters())
 perm=torch.tensor([2,0,1]);bp={k:(v[perm] if v.ndim==4 or k=='input_valid_mask' else v) for k,v in b.items()}
 torch.testing.assert_close(m(**bp),p[perm],atol=1e-6,rtol=1e-6)
 path=tmp_path/'cp.pt';torch.save(m.state_dict(),path)
 other=NewDataPhaseMModel(key,HISTORY,**SHAPE).eval();other.load_state_dict(torch.load(path,weights_only=True),strict=True)
 torch.testing.assert_close(other(**b),p,atol=0,rtol=0)

def test_prediction_and_representation_formulas_expert_input_effects(models):
 a=models['C5'].eval();b=models['C6'].eval();x=batch()
 p,aux=a(**x,return_aux=True);q=b(**x)
 candidates=torch.stack([a.base.head(aux['experts'][:,k]).view(3,4,23) for k in range(4)],2)
 torch.testing.assert_close(p,(candidates*aux['routing'][...,None]).sum(2),atol=1e-7,rtol=1e-6)
 fused=torch.einsum('bhk,bkd->bhd',aux['routing'],aux['experts'])
 ref=torch.stack([b.base.head(fused[:,h])[:,h*23:(h+1)*23] for h in range(4)],1)
 torch.testing.assert_close(q,ref,atol=1e-7,rtol=1e-6)
 assert (p-q).abs().max()>1e-6
 assert tensor_state_hash(a.state_dict())==tensor_state_hash(b.state_dict())
 for k in range(4):
  changed=aux['experts'].clone();changed[:,k]=torch.randn_like(changed[:,k])
  prediction=(expert_forecasts(a.base.head,changed,23)*aux['routing'][...,None]).sum(2)
  assert (prediction-p).abs().max()>1e-6
 for k in range(4):
  w=torch.nn.functional.one_hot(torch.full((3,4),k),4).float()
  torch.testing.assert_close(a(**x,routing_override=w),candidates[:,:,k],atol=1e-7,rtol=1e-6)

def test_live_not_detached_caches_prefix_causality_and_invalid_inputs(models):
 m=models['C5'].eval();x=batch();changed={k:v.clone() for k,v in x.items()};changed['X'][:,10:]=torch.randn_like(changed['X'][:,10:])
 _,a=m(**x,return_aux=True);_,b=m(**changed,return_aux=True)
 torch.testing.assert_close(a['H'][:,:10],b['H'][:,:10],atol=0,rtol=0)
 torch.testing.assert_close(a['Z'][:,:10],b['Z'][:,:10],atol=0,rtol=0)
 expected=m(**x)
 def poison(module,args):
  m.encoder.diagnostics['long_memory']=torch.full_like(args[2],float('nan'))
  m.encoder.diagnostics['latent_states']=torch.full_like(args[3],float('nan'))
 hook=m.experts.register_forward_pre_hook(poison)
 torch.testing.assert_close(m(**x),expected,atol=0,rtol=0);hook.remove()
 x['input_valid_mask'][:,:,:2]=False;p=m(**x);x['X'][:,:,:2]=float('nan')
 torch.testing.assert_close(m(**x),p,atol=0,rtol=0)

def test_controlled_expert_inputs_change_candidate_forecasts(models):
 m=models['C5'].eval();p,a=m(**batch(),return_aux=True)
 arguments=[(a['Z'],),(a['H'],),(a['s'],),(a['s'],a['t'])]
 for k,expert in enumerate(m.experts.children()):
  args=arguments[k];changed=[torch.randn_like(v) for v in args]
  original=expert(*args);alternative=expert(*changed)
  assert not torch.equal(original,alternative)
  bank=a['experts'].clone();bank[:,k]=alternative
  candidate=expert_forecasts(m.base.head,bank,23)
  assert (candidate[:,:,k]-a['expert_prediction'][:,:,k]).abs().max()>1e-6

def test_native_kl_once_optimizer_coverage_and_updates(models):
 m=models['C5'].train();m.set_epoch(20);b=batch(2);y=torch.randn(2,4,23)*.02
 original={n:v.detach().clone() for n,v in m.named_parameters()};optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
 assert {id(v) for g in optimizer.param_groups for v in g['params']}=={id(v) for v in m.parameters()}
 with patch.object(m.switching_latent_transformer,'switch_loss',wraps=m.switching_latent_transformer.switch_loss) as kl:
  p=m(**b);pred=prediction_loss(p,y,torch.ones_like(y,dtype=torch.bool));value=m.switch_loss();loss=pred+value
  assert kl.call_count==1 and torch.isfinite(value)
  loss.backward()
 assert m.switching_latent_transformer.regime_filter.current_beta==pytest.approx(.0005)
 for prefix in ['encoder.native.type_proj','encoder.native.switching_latent_transformer','experts.','router.']:
  assert any(v.grad is not None and v.grad.abs().sum()>0 for n,v in m.named_parameters() if n.startswith(prefix)),prefix
 optimizer.step();assert any(not torch.equal(original[n],v) for n,v in m.named_parameters() if n.startswith('router.'))

def test_dataset_whitelist_and_test_access_sentinel(tmp_path,monkeypatch):
 x=np.ones((25,29,26),np.float32)
 a=dict(features=x,input_valid_mask=np.ones(x.shape[:-1],bool),feature_valid_mask=np.ones_like(x,bool),feature_applicable_mask=np.ones((29,26),bool),
  reference_dates=np.array([f'2024-01-{i:02d}' for i in range(1,26)]),train_targets=np.ones((2,4,23),np.float32),train_label_valid_mask=np.ones((2,4,23),bool),
  train_origin_indices=np.array([19,20]),train_window_indices=np.array([np.arange(20),np.arange(1,21)]),test_targets=np.array([999]))
 for k,v in a.items():np.save(tmp_path/(k+'.npy'),v,allow_pickle=False)
 orig=np.load
 def guard(path,*args,**kw):
  if Path(path).name.startswith('test_'):pytest.fail('TEST opened')
  return orig(path,*args,**kw)
 monkeypatch.setattr(np,'load',guard)
 ds=TrainValDataset(tmp_path,'train');assert len(ds)==2 and ds[1]['origin_date']=='2024-01-21'
 with pytest.raises(ValueError,match='Only TRAIN/VAL'):TrainValDataset(tmp_path,'test')

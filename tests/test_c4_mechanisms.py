from unittest.mock import patch
import pytest
import torch
from cmgm_c4.models.c4 import C4Model
from cmgm_c4.models.ablations import C4Ablation,KEYS
from cmgm_c4.models.experts import expert_forecasts
from cmgm_c4.common import seed_all,prediction_loss,project_root,tensor_hash
from cmgm_c4.factory import build

SHAPE=dict(nodes=29,stocks=4,bonds=2,targets=23,features=26)

def batch(n=3):
    seed_all(51);x=torch.randn(n,20,29,26)
    return dict(X=x,input_valid_mask=torch.ones(n,20,29,dtype=torch.bool),
        feature_valid_mask=torch.ones_like(x,dtype=torch.bool),feature_applicable_mask=torch.ones(29,26,dtype=torch.bool))

def test_single_forward_live_gradients_shared_head_global_router_and_updates():
    seed_all();m=C4Model(**SHAPE).eval();b=batch();counts=dict(spatial=0,temporal=0,head=0)
    s,t=m.encoder.spatial,m.encoder.temporal
    def spatial(*a,**k):counts['spatial']+=1;return s(*a,**k)
    def temporal(*a,**k):counts['temporal']+=1;return t(*a,**k)
    hook=m.base.head[0].register_forward_hook(lambda *a:counts.__setitem__('head',counts['head']+1))
    with patch.object(m.encoder,'spatial',side_effect=spatial),patch.object(m.encoder,'temporal',side_effect=temporal):
        p,a=m(**b,return_aux=True)
    hook.remove()
    assert counts==dict(spatial=1,temporal=1,head=1)
    assert not hasattr(m.base,'gate_fc')
    assert p.shape==(3,4,23) and a['expert_prediction'].shape==(3,4,4,23)
    assert a['H'].shape==(3,20,128) and a['Z'].shape==(3,20,64)
    assert a['H'].requires_grad and a['Z'].requires_grad
    a['H'].retain_grad();a['Z'].retain_grad()
    w=a['routing'];assert (w>=0).all()
    torch.testing.assert_close(w.sum(-1),torch.ones(3,4))
    torch.testing.assert_close(w,w[:,:1].expand_as(w),atol=0,rtol=0)
    torch.testing.assert_close(p,(w[...,None]*a['expert_prediction']).sum(2))
    m.set_epoch(20)
    with patch.object(m.switching_latent_transformer,'switch_loss',wraps=m.switching_latent_transformer.switch_loss) as kl:
        loss=prediction_loss(p,torch.randn_like(p)*.02,torch.ones_like(p,dtype=torch.bool))+m.switch_loss()
        assert kl.call_count==1
        loss.backward()
    assert a['H'].grad.abs().sum()>0 and a['Z'].grad.abs().sum()>0
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    for module in list(m.experts.children())+[m.router,m.base.head,m.base.graph_learner,m.base.attn_mixhop1,m.base.attn_mixhop2,m.switching_latent_transformer]:
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in module.parameters())
    # One synthetic update only; this is not real-data training.
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    assert {id(v) for g in opt.param_groups for v in g['params']}=={id(v) for v in m.parameters()}
    before=tensor_hash(m.state_dict());opt.step();assert tensor_hash(m.state_dict())!=before

def test_horizon_major_rows_and_all_expert_paths():
    m=C4Model(**SHAPE).eval();b=batch();p,a=m(**b,return_aux=True)
    ref=torch.stack([m.base.head(a['experts'][:,k]).reshape(3,4,23) for k in range(4)],2)
    torch.testing.assert_close(a['expert_prediction'],ref,atol=1e-7,rtol=1e-6)
    for k in range(4):
        w=torch.nn.functional.one_hot(torch.full((3,4),k),4).float()
        torch.testing.assert_close(m(**b,routing_override=w),ref[:,:,k],atol=1e-7,rtol=1e-6)
        bank=a['experts'].clone();bank[:,k]=torch.randn_like(bank[:,k])
        assert not torch.allclose(expert_forecasts(m.base.head,bank,23)[:,:,k],ref[:,:,k])

def test_masks_prefix_causality_permutation_and_no_diagnostic_cache_inputs():
    m=C4Model(**SHAPE).eval();b=batch();p,a=m(**b,return_aux=True)
    order=torch.tensor([2,0,1]);perm={k:(v[order] if k!='feature_applicable_mask' else v) for k,v in b.items()}
    torch.testing.assert_close(m(**perm),p[order],atol=1e-6,rtol=1e-6)
    altered={k:v.clone() for k,v in b.items()};altered['X'][:,10:]=torch.randn_like(b['X'][:,10:])
    _,c=m(**altered,return_aux=True)
    torch.testing.assert_close(a['H'][:,:10],c['H'][:,:10],atol=0,rtol=0)
    torch.testing.assert_close(a['Z'][:,:10],c['Z'][:,:10],atol=0,rtol=0)
    def poison(module,args):
        m.encoder.diagnostics['long_memory']=torch.full_like(args[2],float('nan'))
        m.encoder.diagnostics['latent_states']=torch.full_like(args[3],float('nan'))
    hook=m.experts.register_forward_pre_hook(poison)
    torch.testing.assert_close(m(**b),p,atol=0,rtol=0);hook.remove()
    b['input_valid_mask'][:,:,0]=False;reference=m(**b);b['X'][:,:,0]=float('nan')
    torch.testing.assert_close(m(**b),reference,atol=0,rtol=0)
    b['input_valid_mask'][:]=False
    assert torch.isfinite(m(**b)).all()

@pytest.mark.parametrize('key',KEYS)
def test_thirteen_ablations_shapes_and_roundtrip(key,tmp_path):
    seed_all();m=C4Ablation(key,**SHAPE).eval();b=batch(2)
    p,a=m(**b,return_aux=True);assert p.shape==(2,4,23) and torch.isfinite(p).all()
    if key=='no_moe':assert not hasattr(m,'experts') and hasattr(m.base,'gate_fc')
    elif key=='no_dynamic_routing':torch.testing.assert_close(a['routing'],torch.full((2,4,4),.25))
    elif key=='no_spatial':assert (a['routing'][:,:,2:]==0).all()
    elif key=='no_temporal':assert (a['routing'][:,:,2]==1).all()
    path=tmp_path/'model.pt';torch.save(m.state_dict(),path)
    other=C4Ablation(key,**SHAPE).eval();other.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(other(**b),p,atol=0,rtol=0)

def test_original_train_eval_rng_and_gradient_parity():
    # Match the original capture's CPU reduction/thread setting as well as RNG.
    previous_threads=torch.get_num_threads();torch.set_num_threads(4)
    ref=torch.load(project_root()/'provenance/legacy_synthetic_reference.pt',map_location='cpu',weights_only=True)
    seed_all();m=C4Model(nodes=311,stocks=263,bonds=25,targets=23,features=26)
    assert tensor_hash(m.state_dict())==ref['constructor_hash']
    m=build('c4/full',initial=True)
    assert tensor_hash(m.state_dict())==ref['initial_hash']
    for mode in ('eval','train'):
        m.train(mode=='train');m.set_epoch(20);m.zero_grad(set_to_none=True);seed_all(11013)
        p,a=m(**ref['batch'],return_aux=True)
        pred=prediction_loss(p,ref['target'],torch.ones_like(p,dtype=torch.bool));kl=m.switch_loss()
        (pred+kl).backward();old=ref['outputs'][mode]
        for k,v in dict(prediction=p,prediction_loss=pred,KL=kl,**{k:a[k] for k in ('routing','expert_prediction','H','Z')}).items():
            torch.testing.assert_close(v,old[k],atol=1e-7,rtol=1e-5)
        assert torch.equal(torch.get_rng_state(),old['rng'])
        for n,v in m.named_parameters():torch.testing.assert_close(v.grad,old['gradients'][n],atol=1e-7,rtol=1e-4)
    torch.set_num_threads(previous_threads)

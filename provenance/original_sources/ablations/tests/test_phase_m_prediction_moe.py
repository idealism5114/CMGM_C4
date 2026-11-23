import copy
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from cmgm.models.phase_m_prediction_moe import (
    PhaseMModel,PhaseMRouter,KEYS,matched_models,expert_forecasts,checked_weights,
)
from cmgm.models.horizon_tensor_fusion import predict_by_horizon
from cmgm.models.frozen_d0b_task_experts import tensor_state_hash
from cmgm.scripts.phase_m_analysis import population,intervention_routes,moving_block_bootstrap

KW=dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,commodity_indices=(6,30))


@pytest.fixture
def models():
    torch.manual_seed(42)
    return matched_models(KW)[0]


def test_explicit_shared_initialization_and_rng_preservation():
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    models,audit=matched_models(KW)
    assert torch.equal(before,torch.get_rng_state())
    assert audit['PASS'] and audit['C5_C6_entire_initial_state_identical']
    assert len(set(audit['expert_state_sha256'].values()))==1
    for k,m in models.items():
        if k!='C0':
            assert not hasattr(m.base,'gate_fc')
            assert not any(n.startswith('gate_fc') for n in m.base.state_dict())
    assert len({next(m.experts.parameters()).data_ptr() for k,m in models.items() if k!='C0'})==6


@pytest.mark.parametrize('control',KEYS)
def test_full_shape_gradient_permutation_single_and_roundtrip(control,models,tmp_path):
    model=models[control].eval();x=torch.randn(3,20,30,21);y=torch.randn(3,4,24)*.02
    before=tensor_state_hash(model.state_dict())
    p,aux=model(x,return_aux=True)
    assert p.shape==(3,4,24) and torch.isfinite(p).all()
    permutation=torch.tensor([2,0,1])
    torch.testing.assert_close(model(x[permutation]),p[permutation],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model(x[:1]),p[:1],atol=1e-6,rtol=1e-6)
    if control!='C0':
        assert aux['experts'].shape==(3,4,64) and aux['routing'].shape==(3,4,4)
        assert aux['H'].requires_grad and aux['Z'].requires_grad
        aux['H'].retain_grad();aux['Z'].retain_grad()
        torch.testing.assert_close(aux['routing'].sum(-1),torch.ones(3,4))
    loss=sum(nn.functional.huber_loss(p[:,h],y[:,h],delta=.02) for h in range(4))
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    if control!='C0':
        assert aux['H'].grad.abs().sum()>0 and aux['Z'].grad.abs().sum()>0
        for name,module in model.experts.named_children():
            assert any(v.grad.abs().sum()>0 for v in module.parameters()),name
        assert any(v.grad.abs().sum()>0 for v in model.router.parameters())
    assert tensor_state_hash(model.state_dict())==before
    path=tmp_path/'best.pt';torch.save(model.state_dict(),path)
    restored=PhaseMModel(control,KW).eval()
    restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(restored(x),p,atol=0,rtol=0)


def test_c0_unchanged_original_forward_and_gate_order(models):
    m=models['C0'].eval();x=torch.randn(2,20,30,21)
    torch.testing.assert_close(m(x),m.base(x),atol=0,rtol=0)
    b=m.base;hs=b._temp_weighted_spatial(x);ht=b.switching_latent_transformer(x)
    g=b.gate_fc(torch.cat((hs,ht),-1)).sigmoid()
    expected=b.head(g*b.lstm_proj(ht)+(1-g)*b.gcn_proj(hs)).view(2,4,24)
    torch.testing.assert_close(m(x),expected,atol=0,rtol=0)


def test_single_backbone_pass_and_shared_head_body(models):
    for key,m in models.items():
        m.eval();calls={'space':0,'time':0,'body':0}
        spatial=m.base._temp_weighted_spatial
        def space(*a,**k):calls['space']+=1;return spatial(*a,**k)
        ht=m.base.switching_latent_transformer.register_forward_hook(lambda *a:calls.__setitem__('time',calls['time']+1))
        hh=m.base.head[0].register_forward_hook(lambda *a:calls.__setitem__('body',calls['body']+1))
        with patch.object(m.base,'_temp_weighted_spatial',side_effect=space):m(torch.randn(2,20,30,21))
        ht.remove();hh.remove()
        assert calls==dict(space=1,time=1,body=1),(key,calls)
        assert sum(isinstance(z,nn.Linear) and z.out_features==96 for z in m.modules())==1


def test_router_control_invariants_and_context_sensitivity():
    s=torch.randn(3,64);t=torch.randn(3,64);h=torch.randn(3,20,128);z=torch.randn(3,20,64)
    for control in KEYS[1:]:
        torch.manual_seed(42);r=PhaseMRouter(control).eval();w=r(s,t,h,z)
        assert w.shape==(3,4,4) and (w>=0).all()
        torch.testing.assert_close(w.sum(-1),torch.ones(3,4))
        if control in ('C1','C2','C3'):
            torch.testing.assert_close(w,r(s*3,t-1,h*2,z+2),atol=0,rtol=0)
            with torch.no_grad():r.logits.copy_(torch.arange(r.logits.numel()).reshape_as(r.logits).sin())
            w=r(s,t,h,z)
            torch.testing.assert_close(w[0],w[1],atol=0,rtol=0)
        if control=='C1':assert torch.equal(w[:,0],w[:,3])
        if control=='C2':
            assert torch.equal(w[:,0],w[:,1]) and torch.equal(w[:,2],w[:,3]) and not torch.equal(w[:,0],w[:,2])
        if control=='C3':assert r.logits.shape==(4,4) and not torch.equal(w[:,0],w[:,3])
        if control in ('C4','C5','C6'):
            assert float((w-r(s*3+1,t-2,h+1,z-1)).abs().max())>1e-7
            torch.testing.assert_close(w,r(s,t,h,z),atol=0,rtol=0)
        if control=='C4':assert torch.equal(w[:,0],w[:,3])
        if control in ('C5','C6'):assert float((w[:,0]-w[:,3]).abs().max())>1e-8


def test_shared_head_expert_rows_and_prediction_vs_representation(models):
    x=torch.randn(2,20,30,21);a=models['C5'].eval();b=models['C6'].eval()
    p,aux=a(x,return_aux=True);q=b(x)
    bank=aux['experts'];pi=aux['routing'];head=a.base.head
    reference=torch.stack([head(bank[:,k]).view(2,4,24) for k in range(4)],2)
    torch.testing.assert_close(aux['expert_prediction'],reference,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(p,(pi[:,:,:,None]*reference).sum(2),atol=1e-7,rtol=1e-6)
    fused=torch.einsum('bhk,bkd->bhd',pi,bank)
    rows=torch.stack([head(fused[:,h])[:,h*24:(h+1)*24] for h in range(4)],1)
    torch.testing.assert_close(q,rows,atol=1e-7,rtol=1e-6)
    assert (p-q).abs().max()>1e-5  # Natural random initialization, no parameter amplification.
    assert tensor_state_hash(a.state_dict())==tensor_state_hash(b.state_dict())
    for k in range(4):
        one=torch.nn.functional.one_hot(torch.full((2,4),k),4).float()
        torch.testing.assert_close(a(x,routing_override=one),reference[:,:,k],atol=1e-7,rtol=1e-6)
    for k in range(4):
        modified=bank.clone();modified[:,k]=torch.randn_like(modified[:,k])
        changed=torch.einsum('bhk,bhkc->bhc',pi,expert_forecasts(head,modified,24))
        assert (changed-p).abs().max()>1e-6


def test_temporal_causality_and_live_interface_equivalence(models):
    m=models['C5'].eval();x=torch.randn(2,20,30,21);later=x.clone();later[:,10:]*=3
    _,a=m(x,return_aux=True);_,b=m(later,return_aux=True)
    torch.testing.assert_close(a['H'][:,:10],b['H'][:,:10],atol=0,rtol=0)
    torch.testing.assert_close(a['Z'][:,:10],b['Z'][:,:10],atol=0,rtol=0)
    branch=m.switching_latent_transformer
    torch.testing.assert_close(branch(x),branch(x,return_live_sequences=True)[0],atol=0,rtol=0)


def test_training_epoch_uses_native_kl_once_and_optimizer_scope(models):
    from cmgm.training.train import train_epoch
    m=models['C5'];m.set_epoch(10)
    x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.02
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    assert {id(v) for g in optimizer.param_groups for v in g['params']}=={id(v) for v in m.parameters()}
    capture=[];handle=m.register_forward_hook(lambda mod,args,p:capture.append(p.detach()))
    branch=m.switching_latent_transformer
    with patch.object(branch,'switch_loss',wraps=branch.switch_loss) as mock:
        value=train_epoch(m,DataLoader(TensorDataset(x,y),batch_size=2),torch.empty(2,0,dtype=torch.long),
            torch.empty(0),optimizer,nn.HuberLoss(delta=.02),torch.device('cpu'))
        assert mock.call_count==1
    handle.remove()
    expected=sum(nn.functional.huber_loss(capture[0][:,h],y[:,h],delta=.02) for h in range(4))+branch.switch_loss().detach()
    assert value==pytest.approx(float(expected),rel=1e-6)
    assert branch.regime_filter.current_beta==pytest.approx(.0005*9/19)
    m.set_epoch(20);assert branch.regime_filter.current_beta==pytest.approx(.0005)


def test_macro_metric_is_not_batch_mean_or_pooled_with_unequal_masks():
    p=np.zeros((3,4,24));y=np.zeros_like(p);mask=np.ones_like(p,bool)
    y[:,1,0]=[1,3,9];mask[1:,1,0]=False;y[:,1,1:]=2
    r=population(p,y,mask)
    assert r['primary']==pytest.approx((1+23*2)/24)
    assert r['primary']!=r['horizons']['5']['pooled_MAE']


def test_interventions_preserve_horizon_axes_and_groups():
    rng=np.random.default_rng(42);train=rng.random((5,4,4));train/=train.sum(-1,keepdims=True)
    val=rng.random((7,4,4));val/=val.sum(-1,keepdims=True)
    ops=intervention_routes(train,val)
    assert np.array_equal(ops['swap_1d_20d'][:,1:3],val[:,1:3])
    assert np.array_equal(ops['swap_5d_10d'][:,[0,3]],val[:,[0,3]])
    assert np.allclose(ops['train_mean'][0],train.mean(0))
    assert all(np.allclose(w.sum(-1),1) for w in ops.values())
    with pytest.raises(ValueError):checked_weights(np.zeros((2,4,4)),2,'cpu',torch.float32)


def test_paired_bootstrap_grouped_origins():
    target=np.zeros((45,4,24));a=np.ones_like(target)*.2;b=np.ones_like(target)*.3
    r=moving_block_bootstrap(a,b,target,repeats=8)
    for ci in r['delta_candidate_minus_reference_CI95'].values():assert np.allclose(ci,[-.1,-.1])
    with pytest.raises(ValueError):moving_block_bootstrap(a,b,target,block=5)


def test_train_val_cache_has_test_access_sentinels(tmp_path,monkeypatch):
    import json
    from cmgm.scripts.phase_m_horizon_prediction_moe import data_from_cache,split_loader
    from cmgm.scripts.d0b_task_conditioned_moe import data_audit
    from cmgm.data.data_loader import MarketSequenceDataset
    import cmgm.data.data_loader as dl
    import cmgm.scripts.main_ablation as legacy
    monkeypatch.setattr(dl,'create_data_loaders',lambda *a,**k:pytest.fail('TEST-capable factory called'))
    monkeypatch.setattr(legacy,'build_data',lambda *a,**k:pytest.fail('TEST-capable entry called'))
    rng=np.random.default_rng(42);prices=(100+rng.normal(size=(45,30)).cumsum(0)*.01)
    features=rng.normal(size=(45,30,21)).astype('float32')
    data=dict(loaders={},feature_names=[str(i) for i in range(30)],market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)),n_nodes=30,n_commodities=24)
    content={}
    for split in ('train','val'):
        ds=MarketSequenceDataset(prices,data['market_indices'],seq_len=20,feature_matrix=features,raw_prices=prices,
            target_type='return',horizons=[1,5,10,20])
        data['loaders'][split]=DataLoader(ds,batch_size=64)
        for key,v in [('prices',ds.prices),('raw_prices',ds.raw_prices),('feature_matrix',ds.feature_matrix)]:content[split+'_'+key]=v
        content[split+'_origin_dates']=np.arange(len(ds)).astype('datetime64[D]')
    expected=data_audit(data)
    content['metadata']=np.array(json.dumps(dict(data_audit=expected,source_sha256={})))
    cache=tmp_path/'train_val.npz';np.savez(cache,**content)
    restored,arrays,meta=data_from_cache(cache,expected)
    assert set(restored['loaders'])=={'train','val'}
    assert data_audit(restored)==expected
    with pytest.raises(ValueError,match='TRAIN/VAL'):split_loader(restored,'test')
    # Names are rejected before loading any forbidden member's numeric data.
    content['test_targets']=np.array([999]);np.savez(tmp_path/'bad.npz',**content)
    with pytest.raises(ValueError,match='forbidden field'):data_from_cache(tmp_path/'bad.npz',expected)


def test_data_gate_requires_explicit_scope_and_cannot_override_identity(tmp_path):
    from cmgm.scripts.phase_m_horizon_prediction_moe import save,authorized_data_gate
    save(tmp_path/'DATA_FINGERPRINT_AUDIT.json',dict(status='IDENTITY_AND_FINGERPRINT_PASS_CAUSALITY_FAIL'))
    with pytest.raises(ValueError,match='STOP'):authorized_data_gate(tmp_path)
    assert not (tmp_path/'TRAINING_AUTHORIZATION.json').exists()
    # The real legacy audit writes this file before the authorization gate.
    save(tmp_path/'AUDIT_RESULTS.json',dict(status='DATA_GATE_BLOCKED'))
    authorized_data_gate(tmp_path,True)
    import json
    r=json.loads((tmp_path/'TRAINING_AUTHORIZATION.json').read_text())
    assert r['DATA_CAUSALITY']=='FAIL_KNOWN_LEGACY_P0' and not r['test_authorized']
    save(tmp_path/'DATA_FINGERPRINT_AUDIT.json',dict(status='IDENTITY_FAILED'))
    with pytest.raises(ValueError,match='Mandatory identity'):authorized_data_gate(tmp_path,True)


def test_optimizer_updates_experts_and_router_and_restore_switch_epoch(models,tmp_path):
    from cmgm.scripts.phase_m_horizon_prediction_moe import checkpoint
    m=models['C5'];m.train();m.set_epoch(10)
    original={n:v.detach().clone() for n,v in m.named_parameters()}
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    p=m(torch.randn(2,20,30,21));target=torch.randn_like(p)*.03
    loss=sum(nn.functional.huber_loss(p[:,h],target[:,h],delta=.02) for h in range(4))+m.switching_latent_transformer.switch_loss()
    loss.backward()
    # Verify prediction gradients before weight_decay could masquerade as learning.
    for prefix in ('experts.recent_change.','experts.long_memory.','experts.spatial_structure.','experts.st_interaction.','router.'):
        assert any(v.grad is not None and v.grad.abs().sum()>0 for n,v in m.named_parameters() if n.startswith(prefix))
    optimizer.step()
    assert any(not torch.equal(original[n],v) for n,v in m.named_parameters() if n.startswith('router.'))
    path=tmp_path/'latest.pt';checkpoint(path,dict(model_state_dict=m.state_dict(),epoch=10,optimizer=optimizer.state_dict()))
    state=torch.load(path,weights_only=False);restored=PhaseMModel('C5',KW).eval()
    restored.load_state_dict(state['model_state_dict'],strict=True);restored.set_epoch(state['epoch'])
    assert restored.switching_latent_transformer.regime_filter.current_beta==pytest.approx(.0005*9/19)
    x=torch.randn(2,20,30,21);m.eval()
    a,aa=m(x,return_aux=True);b,bb=restored(x,return_aux=True)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(aa['routing'],bb['routing'],atol=0,rtol=0)

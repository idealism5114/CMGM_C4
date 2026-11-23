"""Focused new-data adaptation checks; no real TEST and no formal fitting."""
import importlib.util
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from cmgm.models.comparison_baselines import ORDER, make_model, BaselineMultimodalAdapter
from cmgm.models.triweave_baseline_adapter import MaskedNeutralAdapter, prepared_model
from cmgm.scripts.triweave_baseline_benchmark import cache_inputs, NeutralDataset, evaluate, write_report
from cmgm.scripts.formal_main_innovation_ablation import prediction_loss, metrics, save

MI = dict(stock=(0,4),bond=(4,6),commodity=(6,30))


def legacy_module(path, name):
    source = subprocess.check_output(['git','show','HEAD:'+path],text=True)
    mod=types.ModuleType(name);mod.__package__='cmgm.models.baselines'
    exec(compile(source,path,'exec'),mod.__dict__)
    return mod


@pytest.mark.parametrize('name',ORDER)
def test_unchanged_legacy_default_state_and_forward(name):
    old=legacy_module('cmgm/models/comparison_baselines.py','old_comparison')
    modules={}
    for key in ('graph_wavenet','msgnet','crossgnn'):
        modules['cmgm.models.baselines.'+key]=legacy_module('cmgm/models/baselines/'+key+'.py','old_'+key)
    with patch.dict(sys.modules,modules):
        torch.manual_seed(42);original=old.make_model(name,MI).eval()
    torch.manual_seed(42);current=make_model(name,MI).eval()
    assert original.state_dict().keys()==current.state_dict().keys()
    for key,v in original.state_dict().items():assert torch.equal(v,current.state_dict()[key]),key
    x=torch.randn(2,20,30,21)
    torch.testing.assert_close(original(x),current(x),atol=0,rtol=0)


def test_masked_adapter_all_valid_agrees_with_original_and_excludes_invalid():
    x=torch.randn(3,20,30,21)
    adapter=MaskedNeutralAdapter(MI,21)
    iv=torch.ones(3,20,30,dtype=torch.bool);fv=torch.ones_like(x,dtype=torch.bool);app=torch.ones(30,21,dtype=torch.bool)
    torch.testing.assert_close(adapter(x,iv,fv,app),BaselineMultimodalAdapter(4,2,24)(x),atol=2e-7,rtol=2e-7)
    iv[:,:,0]=False;fv[:,:,1,3]=False;app[4,2]=False
    original=adapter(x,iv,fv,app)
    valid=iv[...,None]&fv&app
    poisoned=torch.where(valid,x,torch.full_like(x,float('nan')))
    torch.testing.assert_close(adapter(poisoned,iv,fv,app),original,atol=0,rtol=0)
    assert torch.isfinite(adapter(poisoned,torch.zeros_like(iv),fv,app)).all()
    assert (adapter(poisoned,torch.zeros_like(iv),fv,app)==0).all()


@pytest.mark.parametrize('name',ORDER)
def test_new_dimensions_order_gradient_and_checkpoint(name,tmp_path):
    torch.manual_seed(42);model=prepared_model(name).eval()
    x=torch.randn(3,20,27,26);p=model(x)
    assert p.shape==(3,4,23) and torch.isfinite(p).all()
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(model(x[perm]),p[perm],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model(x[:1]),p[:1],atol=1e-6,rtol=1e-6)
    target=torch.randn_like(p)*.03;mask=torch.ones_like(p,dtype=torch.bool)
    loss=prediction_loss(p,target,mask);loss.backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    unused=[n for n,v in model.named_parameters() if v.grad is None]
    # The existing GWN's final graph output is outside the returned skip path.
    assert unused == (['core.layers.7.graph.projection.weight','core.layers.7.graph.projection.bias'] if name=='GraphWaveNet' else [])
    path=tmp_path/'best.pt';torch.save(model.state_dict(),path)
    other=prepared_model(name).eval();other.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(other(x),p,atol=0,rtol=0)
    if name in ('RNN','GRU','LSTM','VanillaTransformer'):
        final=model.head;labels=torch.arange(92).reshape(4,23)
    elif name=='MTGNN':final=model.head;labels=torch.arange(4)[:,None].expand(4,23)
    elif name=='GraphWaveNet':final=model.core.end[-1];labels=torch.arange(4)[:,None].expand(4,23)
    else:final=model.core.horizon_head;labels=torch.arange(4)[:,None].expand(4,23)
    with torch.no_grad():final.weight.zero_();final.bias.copy_(torch.arange(len(final.bias)))
    torch.testing.assert_close(model(x),labels.float()[None].expand(3,4,23),atol=0,rtol=0)


def test_cache_origin_target_alignment_padding_and_no_test_access(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    x=np.random.default_rng(42).normal(size=(22,30,21)).astype('float32')
    a=dict(features=x,input_valid_mask=np.ones(x.shape[:-1],bool),feature_valid_mask=np.ones_like(x,bool),
           feature_applicable_mask=np.ones((30,21),bool),reference_dates=np.arange(22).astype('str'))
    for split,origins in [('train',[18,19]),('val',[20,21])]:
        a[split+'_origin_indices']=np.array(origins)
        a[split+'_window_indices']=np.array([np.arange(i-19,i+1) for i in origins])
        a[split+'_targets']=np.arange(2*4*24).reshape(2,4,24).astype('float32')
        a[split+'_label_valid_mask']=np.ones((2,4,24),bool)
    a['test_targets']=np.array([999])
    for k,v in a.items():np.save(source/(k+'.npy'),v,allow_pickle=False)
    old=np.load
    def guarded(path,*args,**kwargs):
        if str(path).split('/')[-1].startswith('test_'):pytest.fail('TEST accessed during preparation')
        return old(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    dest=tmp_path/'cache';cache_inputs(source,dest,MI,('train','val'))
    dataset=NeutralDataset(dest,'train')
    first=dataset[0]
    assert first['origin_date']=='18' and first['X'].shape==(20,28,21)
    assert np.all(first['X'][0]==0)
    assert np.array_equal(first['Y'],a['train_targets'][0])
    assert not (dest/'test_targets.npy').exists()


def test_masked_loss_and_global_evaluation_population():
    p=torch.zeros(3,4,23);y=torch.ones_like(p)*.03;m=torch.ones_like(p,dtype=torch.bool)
    m[0,:,0]=False;y[0,:,0]=float('nan')
    loss=prediction_loss(p,y,m)
    assert torch.isfinite(loss)
    expected=4*(.02*(.03-.01))
    assert float(loss)==pytest.approx(expected,rel=1e-6)
    r=metrics(p.numpy(),y.numpy(),m.numpy(),[str(i) for i in range(23)])
    assert r['selection']==pytest.approx(expected,rel=1e-6)
    assert all(v['valid_labels']==68 for v in r['horizons'].values())
    with pytest.raises(ValueError,match='Empty horizon'):prediction_loss(p,y,torch.zeros_like(m))


def test_report_with_only_one_baseline_and_frozen_d0b(tmp_path):
    ref=tmp_path/'ref';ref.mkdir()
    values=dict(horizons={str(h):dict(MAE=.02,MSE=.001,RMSE=np.sqrt(.001),Hit=.5) for h in (1,5,10,20)})
    save(ref/'test_metrics.json',{'Full D0B':values})
    cfg=dict(reference=str(ref),reference_checkpoint=dict(best_epoch=45,parameters=522269))
    write_report(tmp_path,cfg,{'RNN':dict(best_epoch=2,parameters=123)},{'RNN':values})
    assert '1/8' in (tmp_path/'FINAL_REPORT.md').read_text()

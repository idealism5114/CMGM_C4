import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
from cmgm_c4.common import project_root,read,sha,tensor_hash,prediction_loss,rng_state,restore_rng,checkpoint
from cmgm_c4.factory import build,restore
from cmgm_c4.data.dataset import FrozenDataset,loader
from cmgm_c4.verification import train_val_guard
from cmgm_c4.evaluation.metrics import population

REGISTRY=read(project_root()/'configs/registry.json')

@pytest.mark.parametrize('identifier',list(REGISTRY))
def test_best_latest_initial_strict_and_inactive_parameters(identifier):
    m,cp=restore(identifier);latest,c=restore(identifier,which='latest')
    assert cp['epoch']==REGISTRY[identifier]['best_epoch']
    assert cp['selection']=='VAL_5D_COMMODITY_MACRO_MAE'
    initial=build(identifier,initial=True)
    assert m.state_dict().keys()==initial.state_dict().keys()==latest.state_dict().keys()
    if identifier.startswith('ablations/'):
        inactive=read(project_root()/'configs/project.json')['inactive_parameters'][identifier.split('/')[1]]
        assert {n for n,v in m.named_parameters() if not v.requires_grad}==set(inactive)
    if identifier=='c4/full':assert sum(v.numel() for v in m.parameters())==567169

@pytest.mark.parametrize('identifier',[k for k in REGISTRY if k.startswith('baselines/')])
def test_eight_baseline_shape_and_parameter_counts(identifier):
    m,cp=restore(identifier);m.eval()
    with torch.inference_mode():p=m(torch.randn(2,20,27,26))
    assert p.shape==(2,4,23) and torch.isfinite(p).all()
    expected=REGISTRY[identifier]['original_record']['parameters']
    assert sum(v.numel() for v in m.parameters())==expected

def test_train_val_whitelist_coverage_alignment_and_test_sentinel(monkeypatch):
    orig=np.load;opened=[]
    def checked(path,*a,**kw):
        assert not any('test_release' in s for s in Path(path).parts)
        assert not Path(path).name.startswith('test_')
        opened.append(str(path));return orig(path,*a,**kw)
    monkeypatch.setattr(np,'load',checked)
    with train_val_guard():
        for neutral in (False,True):
            train,val=loader('train',neutral),loader('val',neutral)
            assert len(train.dataset)==2726 and len(train)==43
            assert len(val.dataset)==483 and len(val)==8
            assert train.drop_last is False and val.drop_last is False
            assert train.dataset.dates[-1]=='2022-12-29' and val.dataset.dates[-1]=='2024-12-30'
            for ds in (train.dataset,val.dataset):
                first=ds[0];assert first['X'].shape==(20,27 if neutral else 311,26)
                assert first['Y'].shape==(4,23)
                assert first['origin_date']==str(ds.a['reference_dates'][ds.origins[0]])
        before=len(opened)
        with pytest.raises(ValueError,match='TEST'):FrozenDataset('test')
        assert before==len(opened)
    monkeypatch.setattr(np,'load',orig)
    with pytest.raises(RuntimeError,match='TEST numeric access'):
        with train_val_guard():np.load(project_root()/'data/test_release/features.npy')

def test_macro_mae_mask_denominator_and_four_horizon_huber():
    p=np.zeros((3,4,2));y=np.ones_like(p);y[:,:,1]=3
    mask=np.ones_like(p,dtype=bool);mask[1:,:,1]=False
    r=population(p,y,mask)
    assert r['primary']==2 and r['horizons']['5']['pooled_MAE']==1.5
    tt=lambda a:torch.as_tensor(a,dtype=torch.float64)
    loss=prediction_loss(tt(p),tt(y),torch.from_numpy(mask))
    assert float(loss)==pytest.approx(4*(.02*(1-.01)*3+.02*(3-.01))/4)
    y[~mask]=np.nan
    assert float(prediction_loss(tt(p),tt(y),torch.from_numpy(mask)))==pytest.approx(float(loss))
    mask[:,0]=False
    with pytest.raises(ValueError,match='Empty horizon'):prediction_loss(tt(p),tt(y),torch.from_numpy(mask))

def test_checkpoint_optimizer_scheduler_and_rng_roundtrip(tmp_path):
    # Tiny synthetic update only, no real-data training.
    m=torch.nn.Linear(2,1);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
    m(torch.ones(2,2)).sum().backward();opt.step();scheduler.step(.2)
    path=tmp_path/'cp.pt';checkpoint(path,dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),epoch=1))
    cp=torch.load(path,weights_only=False);n=torch.nn.Linear(2,1);n.load_state_dict(cp['model'],strict=True)
    torch.testing.assert_close(n(torch.ones(2,2)),m(torch.ones(2,2)),atol=0,rtol=0)
    restore_rng(cp['rng']);a=torch.rand(3);restore_rng(cp['rng']);assert torch.equal(a,torch.rand(3))

def test_train_without_execute_only_preflight(monkeypatch,tmp_path):
    from cmgm_c4.cli import main
    monkeypatch.setattr(sys,'argv',['run.py','train','--output',str(tmp_path/'untouched')])
    with patch('cmgm_c4.cli.subprocess.call',return_value=0) as child:
        assert main()==0
        assert 'pytest' in child.call_args[0][0]
    assert not (tmp_path/'untouched').exists()

def test_relocated_copy_restores_without_original_repo(tmp_path):
    root=project_root();dest=tmp_path/'relocated'
    shutil.copytree(root/'src',dest/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'configs',dest/'configs')
    (dest/'checkpoints/c4/full').mkdir(parents=True)
    shutil.copy2(root/'checkpoints/c4/full/best.pt',dest/'checkpoints/c4/full/best.pt')
    code="""
import sys
from cmgm_c4.factory import restore
from cmgm_c4.common import project_root
m,c=restore();assert c['epoch']==35
assert sum(p.numel() for p in m.parameters())==567169
assert not any(n=='cmgm' or n.startswith(('cmgm.','cmgm_sep17_frozen')) for n in sys.modules)
assert not any('Commedities' in str(getattr(v,'__file__','')) for v in sys.modules.values())
print('RELOCATED PASS',project_root())
"""
    env=dict(os.environ,PYTHONPATH=str(dest/'src'),CMGM_C4_ROOT=str(dest),PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'RELOCATED PASS' in r.stdout

def test_runtime_imports_do_not_use_old_repository():
    for n,m in list(sys.modules.items()):
        assert not (n=='cmgm' or n.startswith(('cmgm.','cmgm_sep17_frozen')))
        assert 'Commedities' not in str(getattr(m,'__file__',''))

def test_summary_uses_historical_json_not_test_arrays(tmp_path,monkeypatch):
    from cmgm_c4.reporting import summarize
    def blocked(*a,**k):pytest.fail('Summary must not load numeric arrays or checkpoints')
    monkeypatch.setattr(np,'load',blocked);monkeypatch.setattr(torch,'load',blocked)
    rows=summarize(tmp_path/'summary')
    assert len({r['model'] for r in rows})==22
    assert len([r for r in rows if r['split']=='val'])==22
    with pytest.raises(ValueError,match='new'):summarize(tmp_path/'summary')

def test_evaluate_test_requires_explicit_flag_before_loading(monkeypatch,tmp_path):
    from cmgm_c4.cli import main
    monkeypatch.setattr(sys,'argv',['run.py','evaluate','--split','test','--output',str(tmp_path/'never')])
    with patch('cmgm_c4.cli.restore',side_effect=AssertionError('Checkpoint must not load')):
        with pytest.raises(ValueError,match='allow-test'):main()
    assert not (tmp_path/'never').exists()

def test_standalone_new_run_restore(tmp_path):
    from cmgm_c4.factory import restore_run
    from cmgm_c4.common import save
    m,cp=restore('c4/full');folder=tmp_path/'run';folder.mkdir()
    checkpoint(folder/'best.pt',dict(model_state_dict=m.state_dict(),identifier='c4/full',epoch=35))
    save(folder/'models_frozen.json',dict(identifier='c4/full',best_epoch=35,checkpoint='best.pt',checkpoint_sha256=sha(folder/'best.pt'),selection='VAL_5D_COMMODITY_MACRO_MAE'))
    other,_,_,_=restore_run(folder)
    assert tensor_hash(m.state_dict())==tensor_hash(other.state_dict())

def test_legacy_graphwavenet_unused_parameters_are_preserved():
    m=build('baselines/GraphWaveNet',initial=True).eval()
    m(torch.randn(2,20,27,26)).square().sum().backward()
    unused=sum(v.numel() for v in m.parameters() if v.grad is None)
    assert unused==3104

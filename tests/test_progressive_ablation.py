import numpy as np
import pytest
import torch

from cmgm_c4.common import prediction_loss, project_root, read, save
from cmgm_c4.factory import build
from cmgm_c4.models.progressive import ROWS
from cmgm_c4.training.progressive import comparison, routing_diagnostic, summarize


@pytest.mark.parametrize('row', ROWS)
def test_paired_model_paths_and_gradients(row):
    model = build('progressive/'+row, seed=43)
    full = build('progressive/E', seed=43)
    full_state = full.state_dict()
    for name,tensor in model.state_dict().items():
        if name in full_state:
            assert torch.equal(tensor,full_state[name]), name
    x = torch.randn(2,20,311,26)
    batch = dict(X=x,input_valid_mask=torch.ones_like(x[...,0],dtype=torch.bool),
                 feature_valid_mask=torch.ones_like(x,dtype=torch.bool),
                 feature_applicable_mask=torch.ones_like(x[0,0],dtype=torch.bool))
    model.eval()
    pred,aux = model(**batch,return_aux=True)
    weights = aux['routing']
    assert pred.shape==(2,4,23) and torch.isfinite(pred).all()
    if row=='A':
        torch.testing.assert_close(weights, weights.new_tensor([0,0,1,0]).expand_as(weights))
        assert aux['H'] is None and not model.switch_loss()
    if row=='B':
        torch.testing.assert_close(weights, weights.new_tensor([1/3,1/3,1/3,0]).expand_as(weights))
        assert not list(model.experts.st_interaction.parameters())
    if row in ('C','D'):
        torch.testing.assert_close(weights, torch.full_like(weights,.25))
        equal = build('progressive/C', seed=43).eval()
        torch.testing.assert_close(pred, equal(**batch))
    if row=='D':
        with torch.no_grad():
            model.router.logits[2].fill_(1)
        pred,aux=model(**batch,return_aux=True)
        assert (aux['routing'][:,:,2]>.25).all()
        torch.testing.assert_close(aux['routing'],aux['routing'][:1,:1].expand_as(weights))
    if row=='E':
        original=build('ablations/full_c4',seed=43).eval()
        torch.testing.assert_close(pred,original(**batch))
    model.train()
    model.set_epoch(10)
    pred=model(**batch)
    target=torch.randn_like(pred)*.02
    loss=prediction_loss(pred,target,torch.ones_like(pred,dtype=torch.bool))+model.switch_loss()
    loss.backward()
    for name,param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    if row=='D':
        assert model.router.logits.grad.abs().sum()>0


def test_seed_changes_initialization():
    a=build('progressive/E',seed=43)
    b=build('progressive/E',seed=44)
    assert not torch.equal(a.base.head[0].weight,b.base.head[0].weight)


def cached_split(folder, split, value, start=0):
    folder.mkdir(parents=True,exist_ok=True)
    shape=(8,4,2)
    target=np.zeros(shape)
    pred=np.full(shape,value)
    weights=np.tile([.1,.2,.3,.4],(8,4,1))
    experts=np.tile(pred[:,:,None,:],(1,1,4,1))
    dates=np.array([f'origin-{i:02}' for i in range(start,start+8)])
    np.savez(folder/(split+'_predictions.npz'),prediction=pred,target=target,
             label_valid_mask=np.ones(shape,bool),origin_dates=dates,routing=weights,expert_prediction=experts)


def test_frozen_routing_uses_train_weights(tmp_path):
    save(tmp_path/'models_frozen.json',dict(identifier='progressive/E',checkpoint_sha256='frozen'))
    cached_split(tmp_path,'train',1.)
    cached_split(tmp_path,'val',1.,8)
    for split,expert in [('train',0),('val',1)]:
        path=tmp_path/(split+'_predictions.npz')
        with np.load(path) as f:
            arrays=dict(f)
        arrays['routing'][:]=np.eye(4)[expert]
        arrays['expert_prediction'][:,:,0]=3.
        np.savez(path,**arrays)
    result=routing_diagnostic(tmp_path)
    assert result['VAL5']=={'dynamic':1.,'train_mean':3.,'uniform':1.5}
    assert result['dynamic_gain_vs_train_mean_pct']==pytest.approx(200/3)


def test_paired_reporting(tmp_path):
    import csv
    rows=comparison({'A':2.,'B':1.,'C':1.5},'tuned',42,'val')
    assert rows[2]['gain_vs_previous_pct']==-50 and rows[2]['gain_vs_A_pct']==25
    assert comparison({'D':0.,'E':1.},'original',43,'val')[1]['gain_vs_previous_pct'] is None
    for seed,values in [(42,[2.,1.]),(43,[1.,1.5])]:
        for key,value in zip('AB',values):
            folder=tmp_path/'tuned'/f'seed{seed}'/'models'/key
            cached_split(folder,'val',value)
            save(folder/'train_val_metrics.json',dict(val=dict(primary=value)))
    summarize(tmp_path,dict(profiles=dict(tuned=dict(models=['A','B'])),seeds=[42,43],report_splits=['val']))
    with (tmp_path/'ABLATION_SUMMARY.csv').open() as f:
        summary=list(csv.DictReader(f))
    assert float(summary[1]['mean_MAE5'])==1.25
    assert float(summary[1]['gain_vs_previous_pct'])==0
    assert (tmp_path/'PERIOD_STABILITY_5D.csv').exists()


def test_suite_interrupt_resume_and_protocol_lock(tmp_path,monkeypatch,capsys):
    from cmgm_c4.training import progressive as suite
    from cmgm_c4.common import sha

    cfg=read(project_root()/'configs/project.json')
    save(tmp_path/'configs/project.json',cfg)
    monkeypatch.setenv('CMGM_C4_ROOT',str(tmp_path))
    out=tmp_path/'runs/progressive'
    calls=[]
    interrupt=[True]

    def synthetic_train(identifier,folder,device,resume,seed,lr,weight_decay):
        key=identifier.split('/')[1]
        if key=='E' and interrupt:
            interrupt.pop()
            raise InterruptedError('simulated')
        calls.append((seed,key,lr))
        cached_split(folder,'train',1.)
        cached_split(folder,'val',1.,8)
        (folder/'best.pt').write_bytes(identifier.encode())
        sig=dict(identifier=identifier,protocol=dict(cfg['train'],seed=seed,lr=lr,weight_decay=weight_decay),
                 source_hashes={},dataset_sha256=cfg['dataset_manifest_sha256'],selector=cfg['selector'])
        save(folder/'history.json',[dict(epoch=1,VAL_5D_COMMODITY_MACRO_MAE=1.)])
        save(folder/'models_frozen.json',dict(identifier=identifier,selection=cfg['selector'],best_epoch=1,
                                              checkpoint_sha256=sha(folder/'best.pt')))
        save(folder/'train_val_metrics.json',dict(val=dict(primary=1.)))
        save(folder/'COMPLETE.json',dict(signature=sig))

    monkeypatch.setattr(suite,'train',synthetic_train)
    with pytest.raises(InterruptedError):
        suite.run(out,'cpu',False,False,[43],True)
    assert not (out/'MODELS_FROZEN.json').exists()
    with pytest.raises(ValueError,match='Resume protocol'):
        suite.run(out,'cpu',True,True,[43],True)
    suite.run(out,'cpu',True,False,[43],True)
    display = capsys.readouterr().out
    assert 'Full 相对此行改善' in display and '+0.000%' in display
    assert len(calls)==7 and len(set(calls))==7
    assert len(read(out/'MODELS_FROZEN.json'))==7
    suite.run(out,'cpu',True,False,[43],True)
    assert len(calls)==7
    assert read(out/'COMPLETE.json')['TEST_evaluated'] is False


def test_seed42_uses_shared_archive_before_configuring_rows(tmp_path,monkeypatch):
    import shutil
    from cmgm_c4.common import sha

    root=project_root()
    shutil.copytree(root/'configs',tmp_path/'configs')
    state=build('c4/full',seed=77).state_dict()
    archive=tmp_path/'initial_states/c4.pt'
    archive.parent.mkdir()
    torch.save({'C4':state},archive)
    save(tmp_path/'provenance/initial_states.json',dict(destination_sha256=sha(archive)))
    monkeypatch.setenv('CMGM_C4_ROOT',str(tmp_path))
    for key in ROWS:
        model=build('progressive/'+key,initial=True,seed=42)
        for name,weight in model.state_dict().items():
            if name in state:
                torch.testing.assert_close(weight,state[name],rtol=0,atol=0)

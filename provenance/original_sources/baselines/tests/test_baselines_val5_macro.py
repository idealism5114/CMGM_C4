"""Selector repair checks: CPU synthetic only, no original experiment writes."""
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pytest
import torch
from cmgm.scripts.baselines_val5_macro import (
    SELECTOR, NeutralSplit, freeze_models, scored, same_population,
)


def test_macro_is_not_pooled_and_uses_complete_population():
    p=np.zeros((5,4,2)); y=np.zeros_like(p); m=np.ones_like(p,dtype=bool)
    y[:,:,0]=1.; y[:,:,1]=3.; m[1:,:,1]=False
    r=scored(p,y,m)
    assert r['primary']==2.
    assert r['horizons']['5']['pooled_MAE']==pytest.approx(8/6)
    assert r['primary_name']==SELECTOR


def test_invalid_targets_never_enter_score():
    p=np.zeros((3,4,2)); y=np.ones_like(p)*.02; m=np.ones_like(p,dtype=bool)
    m[0,:,0]=False; y[0,:,0]=np.nan
    assert scored(p,y,m)['primary']==pytest.approx(.02)
    m[:,:,1]=False
    with pytest.raises(ValueError,match='No valid'): scored(p,y,m)


def test_huber_and_macro_can_choose_different_epochs():
    y=np.zeros((3,4,2)); m=np.ones_like(y,dtype=bool)
    p1=np.full_like(y,.01); p1[:,1]=.03
    p2=np.full_like(y,.04); p2[:,1]=.02
    a,b=scored(p1,y,m),scored(p2,y,m)
    assert a['four_horizon_prediction_Huber_sum']<b['four_horizon_prediction_Huber_sum']
    assert a['primary']>b['primary']


def test_train_val_dataset_does_not_open_test(tmp_path):
    np.save(tmp_path/'features.npy',np.zeros((22,27,26),dtype='float32'))
    np.save(tmp_path/'reference_dates.npy',np.arange('2024-01-01','2024-01-23',dtype='datetime64[D]').astype(str))
    for key,v in dict(train_origin_indices=np.array([0,19]),train_window_indices=np.maximum(np.array([0,19])[:,None]+np.arange(-19,1),-1),
                      train_targets=np.zeros((2,4,23),dtype='float32'),train_label_valid_mask=np.ones((2,4,23),dtype=bool)).items():
        np.save(tmp_path/(key+'.npy'),v)
    original=np.load
    def guarded(path,*a,**kw):
        assert not Path(path).name.startswith('test_')
        return original(path,*a,**kw)
    with patch('numpy.load',side_effect=guarded):
        ds=NeutralSplit(tmp_path,'train')
        assert len(ds)==2 and ds[0]['X'].shape==(20,27,26)
        assert np.all(ds[0]['X'][:19]==0)
        with pytest.raises(ValueError,match='frozen'): NeutralSplit(tmp_path,'test')


def test_test_freeze_rejects_partial_suite_without_touching_files(tmp_path):
    with pytest.raises(ValueError,match='all eight'): freeze_models(tmp_path,{}, {'RNN':{}})
    assert not (tmp_path/'models_frozen.json').exists()


def test_population_order_is_enforced():
    a=dict(target=np.zeros((3,4,2)),label_valid_mask=np.ones((3,4,2),dtype=bool),
           origin_dates=np.array(['a','b','c']),commodity_order=np.array(['x','y']))
    same_population(a,a)
    b=dict(a,origin_dates=a['origin_dates'][::-1])
    with pytest.raises(ValueError,match='origin_dates'): same_population(a,b)


def test_scheduler_monitor_changes_learning_rate_using_macro_only():
    opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
    scores=[.02]+[.03]*6
    for score in scores: scheduler.step(score)
    assert opt.param_groups[0]['lr']==5e-5
    best=min(enumerate(scores,1),key=lambda item:item[1])[0]
    assert best==1

import copy
import pytest
from cmgm_c4.common import project_root,read
from cmgm_c4.training.train import effective_protocol
from cmgm_c4.training.tuning import grid,choose,lock_selection,test_locked_models as evaluate_locked_models


def test_exact_nine_preregistered_points_control_first():
    g=grid();assert len(g)==9 and len({(x['lr'],x['weight_decay']) for x in g})==9
    assert g[0]==dict(trial='T00',lr=1e-4,weight_decay=1e-5,default_control=True)
    assert {x['lr'] for x in g}=={3e-5,1e-4,3e-4}
    assert {x['weight_decay'] for x in g}=={1e-6,1e-5,1e-4}


def test_optimizer_overrides_do_not_change_defaults_or_other_protocol_fields():
    base=read(project_root()/'configs/project.json')['train'];before=copy.deepcopy(base)
    assert effective_protocol(base)==base
    p=effective_protocol(base,3e-5,1e-4)
    assert base==before and p['lr']==3e-5 and p['weight_decay']==1e-4
    assert {k:v for k,v in p.items() if k not in ('lr','weight_decay')}=={k:v for k,v in base.items() if k not in ('lr','weight_decay')}


@pytest.mark.parametrize('lr,wd',[(0,1e-5),(-1,1e-5),(float('nan'),1e-5),(1e-4,-1)])
def test_invalid_optimizer_parameters_rejected(lr,wd):
    with pytest.raises(ValueError):effective_protocol({},lr,wd)


def rows():
    return [dict(**x,VAL5_macro_MAE=.03-i*.0001,checkpoint_sha256=f'hash{i}',best_epoch=20) for i,x in enumerate(grid())]


def test_VAL_selection_all_nine_required_and_tie_rule():
    r=rows();assert choose(r)['trial']=='T08'
    with pytest.raises(ValueError):choose(r[:-1])
    for row in r:row['VAL5_macro_MAE']=.03
    assert choose(r)['trial']=='T00'


def test_selection_lock_is_immutable(tmp_path):
    r=rows();a=lock_selection(tmp_path,r);assert a['winner']=='T08'
    assert lock_selection(tmp_path,r)==a
    r[0]['VAL5_macro_MAE']=0
    with pytest.raises(ValueError,match='selection changed'):lock_selection(tmp_path,r)


def test_TEST_cannot_open_before_VAL_selection_lock(tmp_path):
    with pytest.raises(FileNotFoundError):evaluate_locked_models(tmp_path,rows(),'cpu')
    assert not (tmp_path/'TEST_OPEN_AUTHORIZATION.json').exists()

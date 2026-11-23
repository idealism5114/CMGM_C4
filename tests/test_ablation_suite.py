import pytest
from cmgm_c4.training.ablation_suite import initialization_audit,freeze_all,evaluate_test as evaluate_locked,make_protocol
from cmgm_c4.models.ablations import KEYS


def test_exact_ablation_definitions_and_tuned_protocol():
    p=make_protocol(False)
    assert len(p['models'])==13 and len(set(p['models']))==13
    assert p['fixed_training']['lr']==3e-4 and p['fixed_training']['weight_decay']==1e-4
    assert p['selector']=='VAL_5D_COMMODITY_MACRO_MAE'
    assert p['fixed_training']['max_epochs']==200 and p['fixed_training']['patience']==10
    assert p['TEST_authorized'] is False


def test_all_ablation_common_initial_tensors_match():
    a=initialization_audit();assert a['status']=='PASS'
    assert len(a['records'])==13
    assert all(r['common_max_difference']==0 for r in a['records'].values())
    assert a['records']['full_c4']['nominal_parameters']==567169
    assert a['records']['no_moe']['nominal_parameters']==522269


def test_all_thirteen_required_and_freeze_immutable(tmp_path):
    rows=[dict(key=k,checkpoint_sha256=k,best_epoch=1) for k in KEYS]
    with pytest.raises(ValueError):freeze_all(tmp_path,rows[:-1])
    a=freeze_all(tmp_path,rows);assert a['main_model']=='full_c4'
    assert freeze_all(tmp_path,rows)==a
    rows[0]['best_epoch']=2
    with pytest.raises(ValueError,match='Locked'):freeze_all(tmp_path,rows)


def test_TEST_cannot_open_on_partial_ablation_results(tmp_path):
    with pytest.raises(ValueError):evaluate_locked(tmp_path,[],'cpu')
    assert not (tmp_path/'TEST_OPEN_AUTHORIZATION.json').exists()

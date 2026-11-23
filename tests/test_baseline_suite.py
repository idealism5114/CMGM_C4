import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('baseline_runner',ROOT/'tools/run_c4_baselines.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_baseline_freeze_requires_all_eight_and_is_immutable(tmp_path):
    rows=[dict(key=k,best_epoch=1,checkpoint_sha256=k) for k in module.MODELS]
    assert len(rows)==8 and len({r['key'] for r in rows})==8
    with pytest.raises(ValueError,match='All eight'):module.freeze_all(tmp_path,rows[:-1])
    assert not (tmp_path/'MODELS_FROZEN.json').exists()
    saved=module.freeze_all(tmp_path,rows)
    assert module.freeze_all(tmp_path,rows)==saved
    rows[0]['best_epoch']=2
    with pytest.raises(ValueError,match='Locked'):module.freeze_all(tmp_path,rows)


def test_baseline_TEST_blocked_before_all_selections(tmp_path):
    with pytest.raises(ValueError,match='All eight'):module.evaluate_test(tmp_path,[],'cpu',{})
    assert not (tmp_path/'TEST_OPEN_AUTHORIZATION.json').exists()


def test_baseline_requires_explicit_TEST_authorization(tmp_path):
    rows=[dict(key=k,best_epoch=1,checkpoint_sha256=k) for k in module.MODELS]
    module.freeze_all(tmp_path,rows)
    with pytest.raises(ValueError,match='authorization'):module.evaluate_test(tmp_path,rows,'cpu',{'TEST_authorized':False})
    assert not (tmp_path/'TEST_OPEN_AUTHORIZATION.json').exists()

import csv
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest
import torch

from cmgm_c4.evaluation.metrics import population
from cmgm_c4.training.train import make_optimizer, training_protocol

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


@pytest.mark.parametrize('name,lr,wd,batch,schedule', [
    ('RNN', 1e-3, 0, 64, 'constant'),
    ('GRU', 1e-3, 0, 64, 'constant'),
    ('LSTM', 1e-3, 0, 64, 'constant'),
    ('VanillaTransformer', 128**-0.5 * 4000**-0.5, 0, 64, 'transformer_warmup'),
    ('GraphWaveNet', 1e-3, 1e-4, 64, 'constant'),
    ('MTGNN', 1e-3, 1e-4, 64, 'constant'),
    ('MSGNet', 1e-4, 0, 32, 'type1'),
    ('CrossGNN', 1e-4, 0, 32, 'type1'),
])
def test_independent_baseline_optimizer_and_lr_schedule(name, lr, wd, batch, schedule):
    defaults = module.read(ROOT/'configs/project.json')['train']
    defaults.update(lr=0.02, weight_decay=0.03, batch_size=12)
    config = training_protocol('baselines/'+name, defaults, root=ROOT)
    assert (config['lr'], config['weight_decay'], config['batch_size']) == pytest.approx((lr, wd, batch))
    assert config['scheduler'] == schedule
    assert training_protocol('c4/full', defaults) == defaults
    override = training_protocol('baselines/'+name, defaults, 0.002, 0.01, root=ROOT)
    assert override == {**config, 'lr': 0.002, 'weight_decay': 0.01}

    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = make_optimizer(model, config)
    group = optimizer.param_groups[0]
    assert group['weight_decay'] == wd
    rates = []
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().sum().backward()
        rates.append(group['lr'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if schedule == 'transformer_warmup':
        assert rates == pytest.approx([128**-0.5 * step * 4000**-1.5 for step in (1, 2, 3, 4)])
        assert group['betas'] == (0.9, 0.98) and group['eps'] == 1e-9
        assert scheduler.lr_lambdas[0](3999) == 1
        assert scheduler.lr_lambdas[0](7999) == pytest.approx(math.sqrt(0.5))
    elif schedule == 'type1':
        assert rates == pytest.approx([lr, lr, lr/2, lr/4])
    else:
        assert rates == [lr]*4


def test_report_uses_split_matched_c4_and_baseline_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    reference = module.REFERENCE/'models/full_c4'
    out = tmp_path/'run'
    folder = out/'models/RNN'
    actual = np.ones((2, 4, 1))
    c4 = population(actual*1.08, actual)
    baseline_val = population(actual*1.1, actual)
    baseline_test = population(actual*1.04, actual)
    for dest, metrics in ((tmp_path/reference, c4), (folder, baseline_val)):
        module.save(dest/'train_val_metrics.json', {'train': metrics, 'val': metrics})
    module.save(tmp_path/reference/'models_frozen.json', {'best_epoch': 15, 'parameters': 123})
    module.save(tmp_path/module.REFERENCE/'test/full_c4/metrics.json', c4)
    module.save(folder/'history.json', [dict(epoch=1, VAL_5D_COMMODITY_MACRO_MAE=0.1,
        train_Huber_batch_mean=0.01, train_KL_batch_mean=0, optimizer_steps=43,
        train_origins_used=2726, lr=1e-3, seconds=1)])
    defaults = module.read(ROOT/'configs/project.json')['train']
    config = training_protocol('baselines/RNN', defaults, root=ROOT)
    protocol = {'reference_C4': str(reference), 'targets': ['commodity'], 'training_by_model': {'RNN': config}}
    rows = [dict(key='RNN', identifier='baselines/RNN', best_epoch=1, parameters=42)]
    module.summarize(out, rows, {}, protocol)
    assert not (out/'TEST_COMPARISON.csv').exists()
    display = module.summarize(out, rows, {'RNN': baseline_test}, protocol)
    assert '+20.0000%' in display and '-100.0000%' in display

    for filename, expected in (('VAL_RANKING.csv', 20), ('TEST_COMPARISON.csv', -100)):
        with (out/filename).open() as handle:
            comparison = {row['key']: row for row in csv.DictReader(handle)}
        assert float(comparison['RNN']['c4_improvement_pct']) == pytest.approx(expected)
    with (out/'ALL_HORIZONS.csv').open() as handle:
        four_horizons = [row for row in csv.DictReader(handle) if row['key'] == 'RNN' and row['split'] == 'test']
    assert {int(row['horizon']) for row in four_horizons} == {1, 5, 10, 20}
    assert all(float(row['c4_improvement_pct_commodity_macro_MAE']) == pytest.approx(-100)
               for row in four_horizons)
    assert '-100.0000%' in (out/'REPORT.md').read_text()


def test_zero_denominator_and_hit_gain():
    from cmgm_c4.reporting import c4_improvement_pct

    assert c4_improvement_pct(0, 0.1) is None
    assert c4_improvement_pct(0.5, 0.6, higher_is_better=True) == pytest.approx(20)

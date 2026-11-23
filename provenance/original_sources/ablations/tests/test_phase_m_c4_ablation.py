"""Synthetic mechanism checks, not evidence of forecast improvement."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from cmgm.models.phase_m_c4_ablation import C4Ablation, KEYS
from cmgm.models.phase_m_new_data import NewDataPhaseMModel
from cmgm.scripts.formal_main_innovation_ablation import prediction_loss

HISTORY = Path(__file__).resolve().parents[1]/'experiments/sep17_d0b_triweave/20261022_153322_689200/historical_source'
SHAPE = dict(nodes=29, stocks=4, bonds=2, targets=23, features=26)


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def batch():
    x = torch.randn(2, 20, 29, 26)
    return dict(X=x, input_valid_mask=torch.ones(2, 20, 29, dtype=torch.bool),
        feature_valid_mask=torch.ones_like(x, dtype=torch.bool), feature_applicable_mask=torch.ones(29, 26, dtype=torch.bool))


def make(key):
    torch.manual_seed(42)
    source = NewDataPhaseMModel('C0' if key == 'no_moe' else 'C4', HISTORY, **SHAPE)
    return C4Ablation(key, HISTORY, initial_state=source.state_dict(), **SHAPE), source


@pytest.mark.parametrize('key', KEYS)
def test_shared_init_shapes_finite_backward_and_strict_restore(key, tmp_path):
    m, source = make(key)
    for n, v in source.state_dict().items():
        assert torch.equal(v, m.state_dict()[n]), n
    b = batch(); m.eval(); m.set_epoch(20)
    b['input_valid_mask'][:, :, 0] = False
    p, a = m(**b, return_aux=True)
    assert p.shape == (2, 4, 23) and torch.isfinite(p).all()
    loss = prediction_loss(p, torch.randn_like(p)*.02, torch.ones_like(p, dtype=torch.bool))+m.switch_loss()
    loss.backward()
    inactive = [n for n, v in m.named_parameters() if v.grad is None]
    m.freeze_inactive(inactive)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters() if v.requires_grad)
    optimizer = torch.optim.Adam([v for v in m.parameters() if v.requires_grad], lr=1e-4)
    assert {id(v) for group in optimizer.param_groups for v in group['params']} == {id(v) for v in m.parameters() if v.requires_grad}
    dirty = {k: v.clone() for k, v in b.items()}; dirty['X'][:, :, 0] = float('nan')
    torch.testing.assert_close(m(**dirty), p, atol=0, rtol=0)
    perm = {k: v.flip(0) if k != 'feature_applicable_mask' else v for k, v in b.items()}
    torch.testing.assert_close(m(**perm), p.flip(0), atol=1e-6, rtol=1e-5)
    torch.save(m.state_dict(), tmp_path/'model.pt')
    restored, _ = make(key); restored.load_state_dict(torch.load(tmp_path/'model.pt', weights_only=True), strict=True)
    restored.eval(); restored.set_epoch(20)
    torch.testing.assert_close(restored(**b), p, atol=0, rtol=0)
    assert hasattr(m.base, 'gate_fc') == (key == 'no_moe')


@pytest.mark.parametrize('key', ['full_c4', 'no_moe'])
def test_reused_controls_exact_original_forward(key):
    m, original = make(key); m.eval(); original.eval(); b = batch()
    torch.testing.assert_close(m(**b), original(**b), atol=0, rtol=0)


def test_uniform_means_candidates_not_states_and_no_router_gradient():
    m, _ = make('no_dynamic_routing'); m.eval(); b = batch()
    p, a = m(**b, return_aux=True)
    assert torch.equal(a['routing'], torch.full_like(a['routing'], .25))
    torch.testing.assert_close(p, a['expert_prediction'].mean(2), atol=1e-7, rtol=1e-6)
    p.sum().backward()
    assert all(v.grad is None for v in m.router.parameters())


@pytest.mark.parametrize('key', ['no_spatial', 'no_temporal'])
def test_branch_removed_once_and_source_dependent_experts_disabled(key):
    m, _ = make(key); m.eval(); b = batch()
    with patch.object(m.encoder, 'spatial', wraps=m.encoder.spatial) as s, patch.object(m.encoder, 'temporal', wraps=m.encoder.temporal) as t:
        p, a = m(**b, return_aux=True)
    assert s.call_count == (0 if key == 'no_spatial' else 1)
    assert t.call_count == (0 if key == 'no_temporal' else 1)
    torch.testing.assert_close(a['routing'].sum(-1), torch.ones(2, 4))
    if key == 'no_spatial':
        assert not a['routing'][:, :, 2:].any()
        assert a['H'].requires_grad and a['Z'].requires_grad
        p.sum().backward()
        assert all(v.grad is None for v in m.experts.spatial_structure.parameters())
        assert all(v.grad is None for v in m.experts.st_interaction.parameters())
    else:
        assert not a['routing'][:, :, [0, 1, 3]].any()
        assert torch.equal(a['routing'][:, :, 2], torch.ones(2, 4))
        assert m.switch_loss().item() == 0
        expected = m.base.head(a['experts'][:, 2]).view(2, 4, 23)
        torch.testing.assert_close(p, expected, atol=1e-7, rtol=1e-6)


def test_original_regime_microstate_and_graph_interventions():
    b = batch(); full, _ = make('full_c4'); full.eval()
    _, a = full(**b, return_aux=True)
    micro, _ = make('no_microstate'); micro.eval()
    _, c = micro(**b, return_aux=True)
    torch.testing.assert_close(a['Z'], c['Z'], atol=0, rtol=0)
    torch.testing.assert_close(a['experts'][:, 0], c['experts'][:, 0], atol=0, rtol=0)
    assert not micro.switching_latent_transformer.last_h_micro_effective.any()
    uniform, _ = make('no_adaptive_regime_routing'); uniform.eval(); uniform.set_epoch(20); uniform(**b)
    d = uniform.encoder.diagnostics
    torch.testing.assert_close(d['regime_p'], torch.full_like(d['regime_p'], 1/3))
    assert uniform.switch_loss().item() == 0
    assert not torch.equal(d['candidates'][:, :, 0], d['candidates'][:, :, 1])
    shared, _ = make('no_regime_specific_transition'); shared.eval(); shared.set_epoch(20); shared(**b)
    d = shared.encoder.diagnostics
    assert torch.equal(d['candidates'][:, :, 0], d['candidates'][:, :, 2])
    assert shared.switch_loss().item() > 0
    edge, _ = make('no_edge_attn_mixhop'); edge.eval(); edge(**b)
    torch.testing.assert_close(edge.encoder.diagnostics['adjacency'], edge.base.graph_learner())
    assert edge.encoder.ordinary_mixhop1.K == 2 and edge.encoder.ordinary_mixhop1.beta == .05
    graph, _ = make('no_adaptive_graph'); graph.eval(); graph(**b)
    assert torch.equal(graph.encoder.diagnostics['adjacency'], torch.ones(29, 29))


def test_preflight_train_val_loader_still_forbids_test():
    from cmgm.scripts.phase_m_new_data import TrainValDataset
    with pytest.raises(ValueError, match='Only TRAIN/VAL'):
        TrainValDataset('/does/not/exist', 'test')


def test_report_before_test_has_no_fake_test_table(tmp_path):
    import json
    from cmgm.scripts.phase_m_c4_ablation import summarize
    folder = tmp_path/'runs/full_c4'
    folder.mkdir(parents=True)
    v = dict(primary=.025, horizons={str(h): dict(commodity_macro_MAE=.025) for h in (1, 5, 10, 20)})
    (folder/'metrics.json').write_text(json.dumps(dict(val=v)))
    cfg = dict(counts=dict(full_c4=dict(Trainable=567169)))
    done = dict(full_c4=dict(best_epoch=35))
    summarize(tmp_path, cfg, done, {})
    assert (tmp_path/'VAL_ablation_table.csv').exists()
    assert not (tmp_path/'main_TEST5_ablation_table.csv').exists()
    assert not (tmp_path/'multi_horizon_TEST_ablation.csv').exists()
    assert '完成 TEST 0/13' in (tmp_path/'REPORT.md').read_text()

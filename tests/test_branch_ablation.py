"""Regression checks for the four independent C4v2 rows."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cmgm_c4.common import prediction_loss, save
from cmgm_c4.factory import build
from cmgm_c4.models.horizon import BRANCHES
from cmgm_c4.training.progressive import summarize


class BranchAblationTest(unittest.TestCase):
    def test_branch_paths_and_gradients(self):
        x = torch.randn(2, 20, 311, 26)
        batch = dict(X=x, input_valid_mask=torch.ones_like(x[..., 0], dtype=torch.bool),
                     feature_valid_mask=torch.ones_like(x, dtype=torch.bool),
                     feature_applicable_mask=torch.ones_like(x[0, 0], dtype=torch.bool))
        full = build('c4v2/full').state_dict()
        for key in BRANCHES:
            with self.subTest(row=key):
                model = build('c4v2/' + key)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, full[name], rtol=0, atol=0)
                model.set_epoch(10)
                with patch.object(model.encoder, 'spatial', wraps=model.encoder.spatial) as spatial, \
                     patch.object(model.encoder, 'temporal', wraps=model.encoder.temporal) as temporal:
                    pred, aux = model(**batch, return_aux=True)
                self.assertEqual(spatial.call_count, int(key != 'no_spatial'))
                self.assertEqual(temporal.call_count, int(key != 'no_temporal'))
                self.assertEqual(tuple(pred.shape), (2, 4, 23))
                if key == 'no_moe':
                    self.assertFalse(hasattr(model, 'experts') or hasattr(model, 'router'))
                    self.assertEqual(aux, {})
                else:
                    expected = {'full': [.25]*4, 'no_temporal': [0, 0, 1, 0],
                                'no_spatial': [.5, .5, 0, 0]}[key]
                    w = aux['routing']
                    torch.testing.assert_close(w, w.new_tensor(expected).expand_as(w))
                if key == 'no_temporal':
                    self.assertEqual(model.switch_loss().item(), 0)
                loss = prediction_loss(pred, torch.randn_like(pred), torch.ones_like(pred, dtype=torch.bool))
                (loss + model.switch_loss()).backward()
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                if key in ('full', 'no_spatial'):
                    self.assertGreater(model.router.global_logits.grad.abs().sum().item(), 0)
                    self.assertGreater(model.router.network[-1].weight.grad.abs().sum().item(), 0)
                    with torch.no_grad():
                        model.router.network[-1].bias[4] = 1
                        weights = model(**batch, return_aux=True)[1]['routing']
                    self.assertTrue((weights[:, 1, 0] > weights[:, 0, 0]).all())
                    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))

    def test_full_percentage_uses_each_removed_branch_as_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for key, value in zip(BRANCHES, [1., 2., .5, 1.]):
                folder = out/'tuned/seed42/models'/key
                save(folder/'train_val_metrics.json', {'val': {'primary': value}})
                shape = (8, 4, 2)
                np.savez(folder/'val_predictions.npz', origin_dates=np.arange(8), target=np.zeros(shape),
                         label_valid_mask=np.ones(shape, bool), prediction=np.full(shape, value))
            p = dict(family='c4v2', profiles={'tuned': {'models': list(BRANCHES)}}, seeds=[42], report_splits=['val'])
            display = summarize(out, p)
            self.assertIn('去时间分支 | 2.000000000 | +50.000%', display)
            self.assertIn('去空间分支 | 0.500000000 | -100.000%', display)
            self.assertEqual((out/'REPORT.md').read_text().strip(), display)

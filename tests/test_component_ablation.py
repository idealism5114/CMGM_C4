"""C4v2 component interventions and read-only use of the branch Full."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cmgm_c4.common import prediction_loss, save, sha
from cmgm_c4.factory import build
from cmgm_c4.models.ablations import KEYS, LABELS
from cmgm_c4.models.horizon import BRANCHES, COMPONENTS
from cmgm_c4.training.progressive import component_reference, summarize


class ComponentAblationTest(unittest.TestCase):
    def test_mechanisms_share_initialization_and_use_horizon_routing(self):
        self.assertEqual(set(COMPONENTS), set(KEYS) - set(BRANCHES) - {'full_c4'})
        x = torch.randn(2, 20, 311, 26)
        batch = dict(X=x, input_valid_mask=torch.ones_like(x[..., 0], dtype=torch.bool),
                     feature_valid_mask=torch.ones_like(x, dtype=torch.bool),
                     feature_applicable_mask=torch.ones_like(x[0, 0], dtype=torch.bool))
        full = build().state_dict()
        for key in COMPONENTS:
            with self.subTest(key=key):
                model = build('c4v2/' + key)
                self.assertEqual(model.ablation, key)
                if key in LABELS and key != 'no_dynamic_routing':
                    self.assertEqual(model.encoder.main_name, LABELS[key])
                for name, value in model.state_dict().items():
                    if name in full:
                        torch.testing.assert_close(value, full[name], rtol=0, atol=0)
                with torch.no_grad():
                    model.router.network[-1].bias[4] = 2.
                model.set_epoch(10)
                pred, aux = model(**batch, return_aux=True)
                self.assertEqual(tuple(pred.shape), (2, 4, 23))
                weights = aux['routing']
                if key == 'no_dynamic_routing':
                    torch.testing.assert_close(weights, torch.full_like(weights, .25))
                    self.assertFalse(any(p.requires_grad for p in model.router.parameters()))
                else:
                    self.assertTrue((weights[:, 1, 0] > weights[:, 0, 0]).all())
                loss = prediction_loss(pred, torch.randn_like(pred), torch.ones_like(pred, dtype=torch.bool))
                (loss + model.switch_loss()).backward()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_full_is_read_from_completed_reference_for_each_split(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CMGM_C4_ROOT=temp):
            root = Path(temp)
            out = root/'runs/c4v2_components/seed42'
            source = root/'runs/c4v2_branches/seed42'
            model_rel = Path('tuned/seed42/models/full')
            model = source/model_rel
            save(root/'configs/project.json', {})
            protocol = dict(id='C4_COMPONENT_V2', family='c4v2', seeds=[42], selector='val5', training={},
                            dataset_sha256='data', project_sha256='cfg', report_splits=['val','test'],
                            profiles={'tuned': dict(lr=.0003, weight_decay=.0001, models=['no_rpe'])})
            source_protocol = dict(protocol, id='C4_BRANCH_V2', profiles={'tuned':dict(protocol['profiles']['tuned'], models=list(BRANCHES))})
            save(source/'PROTOCOL.json', source_protocol)
            model.mkdir(parents=True)
            checkpoint = model/'best.pt'
            checkpoint.write_bytes(b'frozen full')
            digest = sha(checkpoint)
            save(model/'models_frozen.json', dict(identifier='c4v2/full',selection='val5',checkpoint_sha256=digest))
            lock = source/'MODELS_FROZEN.json'
            save(lock, {str(model_rel):digest})
            save(source/'COMPLETE.json', dict(lock_sha256=sha(lock),TEST_evaluated=True))
            stamp = dict(lock_sha256=sha(lock),checkpoint_sha256=digest)
            save(source/'tuned/seed42/test/full/COMPLETE.json', stamp)
            for base, key, val, test in [(source, 'full', 1., 2.), (out, 'no_rpe', 2., 1.)]:
                for split, value in [('val', val), ('test',test)]:
                    group = base/'tuned/seed42'
                    folder = group/'models'/key if split=='val' else group/'test'/key
                    if split=='val':
                        save(folder/'train_val_metrics.json', dict(val=dict(primary=value)))
                    else:
                        save(folder/'metrics.json',dict(primary=value))
                    shape=(8,4,2)
                    np.savez(folder/('val_predictions.npz' if split=='val' else 'predictions.npz'),
                             prediction=np.full(shape,value),target=np.zeros(shape),
                             label_valid_mask=np.ones(shape,bool),origin_dates=np.arange(8))
            before={f:sha(f) for f in source.rglob('*') if f.is_file()}
            protocol['reference_full'] = component_reference(root, protocol)
            summarize(out,protocol)
            with (out/'ABLATION_SUMMARY.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([float(r['full_gain_vs_model_pct']) for r in rows], [0.,50.,0.,-100.])
            self.assertEqual(before,{f:sha(f) for f in source.rglob('*') if f.is_file()})
            checkpoint.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                component_reference(root,protocol)

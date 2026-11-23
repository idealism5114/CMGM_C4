import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c4_paired_statistics',ROOT/'tools/analyze_c4_ablation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_statistics_cannot_read_unfinished_run(tmp_path):
    with pytest.raises(ValueError,match='Wait for all'):module.analyze(tmp_path)


def test_paired_bootstrap_macro_masks_and_constant_deltas(tmp_path):
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    save('COMPLETE.json',{})
    save('PAIRED_STATISTICS_PROTOCOL.json',dict(seed=42,block_origins=40,repeats=2000,splits=['val']))
    save('PREREGISTERED_PROTOCOL.json',dict(models=['full_c4','better','worse']))
    n=80;target=np.zeros((n,4,3),dtype='float32');mask=np.ones_like(target,dtype=bool)
    mask[::3,:,1]=False;mask[::5,:,2]=False
    for split in ('val',):
        for key,value in [('full_c4',1.),('better',0.),('worse',2.)]:
            folder=tmp_path/('test' if split=='test' else 'models')/key;folder.mkdir(parents=True,exist_ok=True)
            path=folder/('predictions.npz' if split=='test' else 'val_predictions.npz')
            np.savez(path,prediction=np.full_like(target,value),target=target,label_valid_mask=mask,origin_dates=np.array([f'row{i}' for i in range(n)]))
    module.analyze(tmp_path)
    rows=list(csv.DictReader((tmp_path/'PAIRED_BOOTSTRAP_5D.csv').open()));assert len(rows)==3
    for row in rows:
        expected={'full_c4':0.,'better':-1.,'worse':1.}[row['key']]
        for column in ('delta_vs_Full','percentile95_low','percentile95_high'):assert float(row[column])==pytest.approx(expected)
    periods=list(csv.DictReader((tmp_path/'PERIOD_STABILITY_5D.csv').open()));assert len(periods)==12
    for row in periods:
        assert float(row['delta_vs_Full'])==pytest.approx({'full_c4':0.,'better':-1.,'worse':1.}[row['key']])

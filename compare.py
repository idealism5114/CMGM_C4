"""Compare the completed seed-42 C4 v2 Full and eight model-specific baselines."""
import argparse
import math
from pathlib import Path

import numpy as np

from tools.run_c4_baselines import MODELS, ROOT, read, sha
from cmgm_c4.evaluation.metrics import population


BASELINE_ID = 'CMGM_C4_DEEP_V3_MODEL_SPECIFIC_BASELINES_SEED42_V1'
FULL_RUN = ROOT / 'runs/c4v2_branches/seed42'
ALIGNMENT = ('origin_dates', 'target', 'label_valid_mask')


def find_baselines():
    candidates = []
    incomplete = []
    for path in (ROOT / 'runs').glob('*/*/PREREGISTERED_PROTOCOL.json'):
        if read(path).get('id') != BASELINE_ID:
            continue
        folder = path.parent
        stamp = folder / 'COMPLETE.json'
        if stamp.is_file() and read(stamp).get('TEST_evaluated'):
            candidates.append(folder)
        else:
            incomplete.append(str(folder))
    if not candidates:
        detail = '。尚未完成 TEST 的目录：' + '、'.join(sorted(incomplete)) if incomplete else ''
        raise ValueError('未找到已完成 TEST 的八模型 baseline 实验' + detail
                         + '。若结果在其他位置，请运行 python compare.py <baseline运行目录>')
    return max(candidates, key=lambda folder: ((folder / 'COMPLETE.json').stat().st_mtime_ns, str(folder)))


def test_score(folder, reference=None):
    with np.load(folder / 'predictions.npz') as arrays:
        labels = {name: arrays[name] for name in ALIGNMENT}
        if reference is not None:
            for name, values in labels.items():
                if not np.array_equal(values, reference[name], equal_nan=values.dtype.kind == 'f'):
                    raise ValueError(f'{folder} 的 {name} 与 Full 不一致')
        score = population(arrays['prediction'], labels['target'], labels['label_valid_mask'])['primary']
    if not math.isclose(score, read(folder / 'metrics.json')['primary'], rel_tol=1e-12):
        raise ValueError(f'{folder} 的 TEST MAE5 与保存的预测不一致')
    return score, labels


def compare(full_run, baselines):
    full = read(full_run / 'PROTOCOL.json')
    baseline = read(baselines / 'PREREGISTERED_PROTOCOL.json')
    if full['id'] != 'C4_BRANCH_V2' or full['seeds'] != [42] or baseline['id'] != BASELINE_ID:
        raise ValueError('需要新版分支消融 Full 和使用各自训练参数的 seed 42 baseline')
    if any(full[key] != baseline[key] for key in ('dataset_sha256', 'selector')):
        raise ValueError('两组实验的数据或选模指标不同')
    if not all(read(folder / 'COMPLETE.json')['TEST_evaluated'] for folder in (full_run, baselines)):
        raise ValueError('两组实验都需要完成 TEST')
    full_dir = full_run / 'tuned/seed42/test/full'
    full_lock = full_run / 'MODELS_FROZEN.json'
    if read(full_dir / 'COMPLETE.json') != dict(lock_sha256=sha(full_lock),
            checkpoint_sha256=read(full_lock)['tuned/seed42/models/full']):
        raise ValueError('Full 的测试结果与冻结记录不符')
    full_score, labels = test_score(full_dir)
    baseline_lock = baselines / 'MODELS_FROZEN.json'
    locked = read(baseline_lock)['models']
    digest = sha(baseline_lock)
    rows = [('Full C4（分周期路由）', full_score)]
    for model in MODELS:
        folder = baselines / 'test' / model
        stamp = read(folder / 'provenance.json')
        if (read(folder / 'COMPLETE.json')['status'] != 'COMPLETE'
                or stamp['models_frozen_sha256'] != digest
                or stamp['checkpoint_sha256'] != locked[model]['checkpoint_sha256']):
            raise ValueError(f'{model} 的测试结果与冻结记录不符')
        score, _ = test_score(folder, labels)
        rows.append((model, score))
    print(f'Full: {full_run}\nBaselines: {baselines}\n')
    print('| 模型（seed 42） | Test MAE5 ↓ | Full 相对此模型误差降幅 |')
    print('|---|---:|---:|')
    for model, score in rows:
        gain = f'{100 * (score - full_score) / score:+.3f}%' if score else '—'
        print(f'| {model} | {score:.9f} | {gain} |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baselines', nargs='?', type=Path, help='指定已有的 baseline 运行目录')
    args = parser.parse_args()
    try:
        compare(FULL_RUN, args.baselines or find_baselines())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'无法生成对比表：{exc}\n')

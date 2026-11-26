"""Recover the interrupted seed-42 progressive ablation and run it in the background."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'runs/ablations_progressive/seed42_20261124_02'
WORK = ROOT / 'runs/abl_recovery'
PROJECT = WORK / 'project'
RUN = Path('runs/seed42')
LOG = WORK / 'run.log'


def restore_sources(destination, hashes):
    snapshots = sorted((ROOT / 'runs').glob('*/*/source_snapshot'))
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_relative_to('src') or '..' in path.parts or path.suffix != '.py':
            raise ValueError(f'实验协议中的源码路径无效：{name}')
        relative = path.relative_to('src')
        for base in [ROOT / 'src', *snapshots]:
            candidate = base / relative
            if candidate.is_file():
                data = candidate.read_bytes()
                if hashlib.sha256(data).hexdigest() == expected:
                    break
        else:
            # ea90672 contains the training code used at the start of this run.
            git = shutil.which('git')
            result = subprocess.run([git, 'show', f'ea90672:{name}'], cwd=ROOT,
                                    capture_output=True) if git else None
            if not result or result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
                raise ValueError(f'本地源码、实验快照及 Git 历史中均未找到协议要求的版本：{name}')
            data = result.stdout
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def prepare():
    protocol_file = ORIGINAL / 'PROTOCOL.json'
    protocol = json.loads(protocol_file.read_text())
    if protocol['seeds'] != [42] or list(protocol['profiles']) != ['tuned']:
        raise ValueError(f'实验协议与单种子恢复任务不符：{protocol_file}')
    with tempfile.TemporaryDirectory(dir=WORK) as temp:
        stage = Path(temp) / 'project'
        stage.mkdir()
        restore_sources(stage, protocol['source_hashes'])
        shutil.copytree(ROOT / 'configs', stage / 'configs')
        if hashlib.sha256((stage / 'configs/project.json').read_bytes()).hexdigest() != protocol['project_sha256']:
            raise ValueError('configs/project.json 与原实验不一致')
        for name in ('data', 'initial_states', 'provenance'):
            source = ROOT / name
            if not source.is_dir():
                raise FileNotFoundError(source)
            (stage / name).symlink_to(source, target_is_directory=True)
        (stage / 'tools').mkdir()
        shutil.copy2(ROOT / 'tools/run_c4_ablation.py', stage / 'tools/run_c4_ablation.py')
        (stage / 'tests').mkdir()
        for name in ('conftest.py', 'test_progressive_ablation.py'):
            shutil.copy2(ROOT / 'tests' / name, stage / 'tests' / name)
        models = Path('tuned/seed42/models')
        (stage / RUN / models).mkdir(parents=True)
        shutil.copy2(protocol_file, stage / RUN / 'PROTOCOL.json')
        for key in ('A', 'B', 'C'):
            source = ORIGINAL / models / key
            if not (source / 'COMPLETE.json').is_file():
                raise ValueError(f'缺少已完成模型记录：{source}')
            (stage / RUN / models / key).symlink_to(source, target_is_directory=True)
        stage.rename(PROJECT)


def show_routing(group):
    import numpy as np

    diagnostic = json.loads((group / 'ROUTING_VAL.json').read_text())
    model = group / 'models/E'
    if diagnostic['checkpoint_sha256'] != json.loads((model / 'models_frozen.json').read_text())['checkpoint_sha256']:
        raise ValueError('验证缓存的模型记录与路由报告不一致')
    with np.load(model / 'val_predictions.npz', allow_pickle=False) as cache:
        weights = cache['routing'][:, 1].astype('float64')
        experts = cache['expert_prediction'][:, 1].astype('float64')
        target = cache['target'][:, 1].astype('float64')
        mask = cache['label_valid_mask'][:, 1].astype(bool)

    counts = mask.sum(axis=0)
    if not counts.all():
        raise ValueError('部分商品没有有效的 5 日验证标签')
    # Average over valid dates within each commodity, then equally over commodities.
    def macro_mean(values):
        return (np.where(mask, values, 0.).sum(axis=0) / counts).mean()

    mean_weights = np.asarray(diagnostic['train_mean_weights'])
    forecast_change = ((weights - mean_weights)[..., None] * experts).sum(axis=1)
    valid_weights = weights[mask.any(axis=1)]
    print('E 组验证集，5 日预测。权重按缓存中的四个专家顺序列出。')
    print('各专家权重跨日期标准差（百分点）：', *(f'{s:.8f}' for s in valid_weights.std(axis=0) * 100))
    print(f'四位专家预测的平均极差（商品等权）：{macro_mean(np.ptp(experts, axis=1)):.10g}')
    print(f'动态权重相对训练平均权重的平均绝对预测变化（商品等权）：{macro_mean(np.abs(forecast_change)):.10g}')
    print(f'参照：同一检查点的动态权重 MAE5 为 {diagnostic["VAL5"]["dynamic"]:.10g}')

    # Compare selections made with hindsight on the same frozen forecasts.
    fixed = np.einsum('k,bkc->bc', mean_weights, experts)
    fixed_error = np.abs(fixed - target)
    expert_error = np.abs(experts - target[:, None, :])
    errors = np.concatenate((fixed_error[:, None, :], expert_error), axis=1)
    weighted_errors = np.where(mask[:, None, :], errors, 0.) / counts
    scores = weighted_errors.sum(axis=0).mean(axis=1)
    by_date = weighted_errors.sum(axis=2).min(axis=1).sum() / len(counts)
    by_date_commodity = weighted_errors.min(axis=1).sum(axis=0).mean()
    names = ('训练集平均权重', '近期变化专家', '长期记忆专家', '空间结构专家', '时空交互专家')
    print('同一检查点的验证集 MAE5')
    for name, score in zip(names, scores):
        print(f'  {name}  {score:.10g}')
    print('事后选择使用了验证标签，仅用于判断候选预测的互补空间。')
    for name, score in (('每个日期共用一个候选', by_date),
                        ('每个日期和商品各选一个候选', by_date_commodity)):
        gain = f'，相对训练集平均权重改善 {(scores[0] - score) / scores[0]:.3%}' if scores[0] else ''
        print(f'  {name}  MAE5 {score:.10g}{gain}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routing', action='store_true', help='查看已有验证缓存中的权重波动和专家预测差异')
    if parser.parse_args().routing:
        show_routing(PROJECT / RUN / 'tuned/seed42')
        return
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'恢复任务正在运行。查看日志：tail -f {LOG.relative_to(ROOT)}')
            return
        if not PROJECT.exists():
            prepare()
        output = PROJECT / RUN
        if (output / 'COMPLETE.json').is_file():
            print((output / 'REPORT.md').read_text())
            return
        protocol = json.loads((output / 'PROTOCOL.json').read_text())
        command = [sys.executable, '-u', 'tools/run_c4_ablation.py', '--execute', '--resume', '--output', str(RUN)]
        if 'test' in protocol['report_splits']:
            command.append('--allow-test')
        env = dict(os.environ, CMGM_C4_ROOT=str(PROJECT))
        env.setdefault('CUDA_VISIBLE_DEVICES', '1')
        with LOG.open('a') as log:
            # The child keeps the launch lock and its own session after SSH disconnects.
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        print(f'后台任务 PID {child.pid}。查看日志：tail -f {LOG.relative_to(ROOT)}')
        print('完成后再次运行 python resume.py 即可查看百分比报告。')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))

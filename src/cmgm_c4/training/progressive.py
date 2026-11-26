"""Paired A–E experiments and frozen-checkpoint routing comparisons."""
import fcntl
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from cmgm_c4.common import project_root, read, runtime_hashes, save, sha, tensor_hash
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import evaluate
from cmgm_c4.evaluation.metrics import population
from cmgm_c4.factory import build, restore_run
from cmgm_c4.models.progressive import ROWS
from cmgm_c4.models.horizon import BRANCHES, COMPONENTS, V2_MODELS
from cmgm_c4.reporting import c4_improvement_pct
from cmgm_c4.verification import train_val_guard
from .train import train
from .tuning import write_csv


def routing_diagnostic(folder):
    folder = Path(folder)
    identity = read(folder/'models_frozen.json')
    if identity['identifier'] not in ('progressive/E', 'c4/full', 'ablations/full_c4'):
        raise ValueError('Routing diagnostics require a frozen Full C4 run')
    with np.load(folder/'train_predictions.npz') as train_cache:
        train_weights = train_cache['routing']
        mean_weights = train_weights.mean(axis=0, dtype=np.float64)
        last_train = train_cache['origin_dates'][-1]
    with np.load(folder/'val_predictions.npz') as val:
        if last_train >= val['origin_dates'][0]:
            raise ValueError('TRAIN must precede VAL')
        weights = val['routing']
        experts = val['expert_prediction']
        for w in (weights, train_weights):
            if w.ndim != 3 or w.shape[1:] != (4, 4) or not np.allclose(w.sum(-1), 1) or not (w >= 0).all():
                raise ValueError('Invalid cached routing probabilities')
            np.testing.assert_allclose(w, np.repeat(w[:, :1], 4, axis=1), atol=1e-6)
        scores = {}
        for label, w in [('uniform', np.full((4,4), .25)), ('train_mean', mean_weights), ('dynamic', weights)]:
            prediction = (experts * w[..., None]).sum(axis=2)
            if label=='dynamic':
                np.testing.assert_allclose(prediction, val['prediction'], rtol=1e-5, atol=1e-6)
            scores[label] = population(prediction, val['target'], val['label_valid_mask'])['primary']
    saved_metrics = folder/'train_val_metrics.json'
    if saved_metrics.exists():
        np.testing.assert_allclose(scores['dynamic'],read(saved_metrics)['val']['primary'],rtol=1e-6,atol=1e-9)
    return dict(split='val', checkpoint_sha256=identity['checkpoint_sha256'], VAL5=scores,
                train_mean_weights=mean_weights[0].tolist(),
                dynamic_gain_vs_train_mean_pct=c4_improvement_pct(scores['train_mean'],scores['dynamic']))


def comparison(scores, profile, seed, split, period='all', *, branches=False):
    def gain(base, score):
        return None if base is None or score is None else c4_improvement_pct(base,score)

    previous = None
    rows = []
    for key, score in scores.items():
        if branches:
            rows.append(dict(profile=profile, seed=seed, split=split, period=period, model=key, MAE5=score,
                             full_gain_vs_model_pct=gain(score, scores['full'])))
            continue
        rows.append(dict(profile=profile, seed=seed, split=split, period=period, model=key, MAE5=score,
                         gain_vs_previous_pct=gain(previous,score), gain_vs_A_pct=gain(scores.get('A'),score),
                         full_gain_vs_model_pct=gain(score,scores.get('E'))))
        previous = score
    return rows


def component_reference(root, protocol):
    """Pin the completed branch run's exact Full checkpoint and training protocol."""
    path = Path('runs/c4v2_branches/seed42')
    reference = root / path
    if not (reference/'COMPLETE.json').is_file():
        raise ValueError(f'组件消融需要已完成的新版分支实验：{reference}')
    source = read(reference/'PROTOCOL.json')
    profile = source['profiles']['tuned']
    if (source['id'] != 'C4_BRANCH_V2' or source['seeds'] != protocol['seeds']
            or not set(protocol['report_splits']).issubset(source['report_splits'])
            or any(source[k] != protocol[k] for k in ('selector', 'training', 'dataset_sha256', 'project_sha256'))
            or any(profile[k] != protocol['profiles']['tuned'][k] for k in ('lr', 'weight_decay'))):
        raise ValueError('新版 Full 的数据、随机种子或训练协议与组件消融不匹配')
    lock = reference/'MODELS_FROZEN.json'
    complete = read(reference/'COMPLETE.json')
    lock_hash = sha(lock)
    if lock_hash != complete['lock_sha256'] or ('test' in protocol['report_splits'] and not complete['TEST_evaluated']):
        raise ValueError('新版 Full 的冻结记录不完整')
    full_key = 'tuned/seed42/models/full'
    digest = read(lock)[full_key]
    frozen = read(reference/full_key/'models_frozen.json')
    if (frozen['identifier'] != 'c4v2/full' or frozen['selection'] != source['selector']
            or frozen['checkpoint_sha256'] != digest or sha(reference/full_key/'best.pt') != digest):
        raise ValueError('新版 Full 的 checkpoint 与已完成的分支实验不匹配')
    return dict(run=str(path), protocol_sha256=sha(reference/'PROTOCOL.json'),
                lock_sha256=lock_hash, checkpoint_sha256=digest)


def summarize(out, protocol):
    branches = protocol.get("family") == "c4v2"
    reference_full = protocol.get("reference_full")
    rows, periods = [], []
    for profile, settings in protocol['profiles'].items():
        for seed in protocol['seeds']:
            group = out/profile/f'seed{seed}'
            for split in protocol['report_splits']:
                scores = {}
                segments = [{} for _ in range(4)]
                reference = None
                models = ['full', *settings['models']] if reference_full else settings['models']
                for key in models:
                    source_group = group
                    if reference_full and key == 'full':
                        source_group = project_root()/reference_full['run']/profile/f'seed{seed}'
                        if split == 'test':
                            stamp = read(source_group/'test/full/COMPLETE.json')
                            if stamp != {k: reference_full[k] for k in ('lock_sha256', 'checkpoint_sha256')}:
                                raise ValueError('新版 Full 的 TEST 缓存与冻结记录不匹配')
                    folder = source_group/'models'/key if split=='val' else source_group/'test'/key
                    filename = 'val_predictions.npz' if split=='val' else 'predictions.npz'
                    with np.load(folder/filename) as cache:
                        dates, y, mask = (cache[k] for k in ('origin_dates','target','label_valid_mask'))
                        if reference is None:
                            reference = (dates,y,mask)
                        else:
                            for a,b in zip(reference,(dates,y,mask)):
                                np.testing.assert_array_equal(a,b)
                        pred = cache['prediction']
                        scores[key] = population(pred,y,mask)['primary']
                        saved = (read(folder/'train_val_metrics.json')['val'] if split=='val'
                                 else read(folder/'metrics.json'))
                        if scores[key] != saved['primary']:
                            raise ValueError(f'Cached {split} MAE differs for {profile}/{seed}/{key}')
                        for segment, idx in zip(segments,np.array_split(np.arange(len(y)),4)):
                            # A segment with an empty commodity denominator has no macro MAE.
                            valid = mask[idx,1]
                            segment[key] = None
                            if valid.any(axis=0).all():
                                error = np.abs(pred[idx,1].astype('float64')-y[idx,1])
                                segment[key] = float((np.where(valid,error,0).sum(0)/valid.sum(0)).mean())
                rows.extend(comparison(scores,profile,seed,split,branches=branches))
                for i,segment in enumerate(segments,1):
                    dates = np.array_split(reference[0],4)[i-1]
                    periods.extend(dict(r, first_origin=str(dates[0]) if len(dates) else None,
                                        last_origin=str(dates[-1]) if len(dates) else None)
                                   for r in comparison(segment,profile,seed,split,i,branches=branches))
    write_csv(out/'PAIRED_SEEDS.csv',rows)
    if periods:
        write_csv(out/'PERIOD_STABILITY_5D.csv',periods)
    if branches:
        write_csv(out/'ABLATION_SUMMARY.csv', rows)
        title = '# C4 组件消融' if reference_full else '# C4 分支消融'
        lines = [title, '', '指标为 5 日商品 Macro MAE，越低越好。',
                 'Full 误差降幅 =（对照 MAE − Full MAE）/ 对照 MAE × 100%。', '',
                 '| seed | 划分 | 模型 | MAE5 | Full 相对此行误差降幅 |', '|---:|---|---|---:|---:|']
        for row in rows:
            gain = row['full_gain_vs_model_pct']
            percent = '—' if gain is None else f'{gain:+.3f}%'
            lines.append(f"| {row['seed']} | {row['split']} | {V2_MODELS[row['model']]} | {row['MAE5']:.9f} | {percent} |")
        if reference_full:
            lines.extend(['', f"Full 沿用已完成实验 {reference_full['run']} 中同一冻结检查点的预测。"])
        report = '\n'.join(lines)
        (out/'REPORT.md').write_text(report + '\n')
        return report

    groups = {}
    for row in rows:
        groups.setdefault((row['profile'],row['split'],row['model']),[]).append(row)
    summary = []
    for (profile,split,key), members in groups.items():
        values = [r['MAE5'] for r in members]
        item = dict(profile=profile,split=split,model=key,seeds=len(values),mean_MAE5=mean(values),
                    sd_MAE5=stdev(values) if len(values)>1 else None)
        for col in ('gain_vs_previous_pct','gain_vs_A_pct','full_gain_vs_model_pct'):
            gains = [r[col] for r in members]
            item[col] = mean(gains) if all(v is not None for v in gains) else None
        summary.append(item)
    write_csv(out/'ABLATION_SUMMARY.csv',summary)
    def pct(x):
        return '—' if x is None else f'{x:+.3f}%'
    seeds = protocol['seeds']
    metric_label = 'MAE5' if len(seeds) == 1 else '平均 MAE5'
    lines = ['# C4 递进消融','',
             '实验种子为 ' + '、'.join(map(str, seeds)) + '。',
             '5 日商品 Macro MAE 越低越好。百分比在相同种子和划分下配对计算。正值表示误差下降。','',
             f'| 配置 | 划分 | 模型 | 种子数 | {metric_label} | 相对前一行改善 | 相对 A 改善 | Full 相对此行改善 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        key=r['model']
        lines.append(f"| {r['profile']} | {r['split']} | {key} {ROWS[key]} | {r['seeds']} | "
                     f"{r['mean_MAE5']:.9f} | {pct(r['gain_vs_previous_pct'])} | {pct(r['gain_vs_A_pct'])} | "
                     f"{pct(r['full_gain_vs_model_pct'])} |")
    terminal_summary='\n'.join(lines)
    lines.extend(['','逐种子与分时段差值见 PAIRED_SEEDS.csv 和 PERIOD_STABILITY_5D.csv。',
                  '各组 E 的 ROUTING_VAL.json 包含等权、TRAIN 平均权重和动态权重的冻结诊断。','',
                  '历史 Full＋12 消融附表见项目 runs/ablations_c4_tuned/seed42_20261118_01/ 下的 VAL_ABLATION.csv 和 TEST_ABLATION.csv。'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return terminal_summary


def run(out, device, resume, allow_test, seeds, sensitivity, *, branches=False, components=False):
    root = project_root()
    out = Path(out).resolve()
    if not out.is_relative_to(root/'runs') or out == root/'runs':
        raise ValueError('Choose a new subdirectory under runs/')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 or s>=2**32 for s in seeds):
        raise ValueError('Expected distinct uint32 seeds')
    cfg = read(root/'configs/project.json')
    if components:
        branches = True
        if seeds != [42]:
            raise ValueError('组件对照使用已完成 Full 的种子 42')
    if branches and sensitivity:
        raise ValueError('D/E sensitivity applies only to progressive ablations')
    definitions = BRANCHES if branches else ROWS
    if components:
        definitions = COMPONENTS
    profiles = dict(tuned=dict(lr=3e-4,weight_decay=1e-4,models=list(definitions)))
    if sensitivity:
        profiles['original'] = dict(lr=1e-4, weight_decay=1e-5, models=['D','E'])
    protocol = dict(id='C4_PROGRESSIVE_V1', seeds=list(seeds), profiles=profiles, definitions=definitions,
                    training=cfg['train'], selector=cfg['selector'], source_hashes=runtime_hashes(),
                    project_sha256=sha(root/'configs/project.json'),dataset_sha256=cfg['dataset_manifest_sha256'],
                    report_splits=['val','test'] if allow_test else ['val'])
    if branches:
        protocol.update(id='C4_BRANCH_V2', family='c4v2')
    if components:
        protocol.update(id='C4_COMPONENT_V2', reference_full=component_reference(root, protocol))
        if out.is_relative_to(root/protocol['reference_full']['run']):
            raise ValueError('组件消融需要独立于原分支实验的输出目录')
    if resume and not (out/'PROTOCOL.json').is_file():
        raise ValueError('Resume requires an existing progressive run')
    out.mkdir(parents=True,exist_ok=resume)
    with (out/'.run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(out,root,protocol,device,resume)


def _run_locked(out,root,protocol,device,resume):
    profiles, seeds = protocol['profiles'], protocol['seeds']
    allow_test = 'test' in protocol['report_splits']
    if resume:
        if read(out/'PROTOCOL.json') != protocol:
            raise ValueError('Resume protocol mismatch')
    else:
        save(out/'PROTOCOL.json',protocol)
    locks = {}
    with train_val_guard():
        for profile,settings in profiles.items():
            for seed in seeds:
                group = out/profile/f'seed{seed}'
                for key in settings['models']:
                    folder = group/'models'/key
                    identifier = protocol.get('family', 'progressive')+'/'+key
                    if not (folder/'COMPLETE.json').exists():
                        train(identifier,folder,device,resume=folder.exists(),seed=seed,
                              lr=settings['lr'],weight_decay=settings['weight_decay'])
                    sig = read(folder/'COMPLETE.json')['signature']
                    expected = dict(protocol['training'],seed=seed,lr=settings['lr'],weight_decay=settings['weight_decay'])
                    if (sig['identifier']!=identifier or sig['protocol']!=expected or sig['source_hashes']!=protocol['source_hashes']
                            or sig['dataset_sha256']!=protocol['dataset_sha256'] or sig['selector']!=protocol['selector']):
                        raise ValueError(f'Training signature mismatch at {folder}')
                    frozen = read(folder/'models_frozen.json')
                    best = min(read(folder/'history.json'),key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
                    val5 = read(folder/'train_val_metrics.json')['val']['primary']
                    if (frozen['identifier']!=identifier or frozen['selection']!=protocol['selector'] or
                            not np.isclose(val5,best['VAL_5D_COMMODITY_MACRO_MAE'],rtol=0,atol=1e-9) or
                            frozen['best_epoch']!=best['epoch'] or sha(folder/'best.pt')!=frozen['checkpoint_sha256']):
                        raise ValueError(f'Invalid VAL checkpoint at {folder}')
                    locks[str(folder.relative_to(out))] = frozen['checkpoint_sha256']
                if 'E' in settings['models']:
                    save(group/'ROUTING_VAL.json',routing_diagnostic(group/'models/E'))
        if runtime_hashes()!=protocol['source_hashes'] or sha(root/'configs/project.json')!=protocol['project_sha256']:
            raise ValueError('Source or config changed during TRAIN/VAL')
        lock = out/'MODELS_FROZEN.json'
        if lock.exists() and read(lock)!=locks:
            raise ValueError('Locked checkpoint collection changed')
        save(lock,locks)
    if allow_test:
        for item in read(root/'provenance/migration_manifest.json')['files']:
            if item['destination'].startswith('data/test_release/') and sha(root/item['destination'])!=item['sha256']:
                raise ValueError('Frozen TEST data changed')
        for rel,digest in locks.items():
            folder = out/rel
            dest = folder.parents[1]/'test'/folder.name
            stamp = dict(checkpoint_sha256=digest, lock_sha256=sha(lock))
            if (dest/'COMPLETE.json').exists() and read(dest/'COMPLETE.json')==stamp:
                continue
            model,_,_,checkpoint = restore_run(folder,device)
            if sha(checkpoint)!=digest:
                raise ValueError('TEST checkpoint differs from VAL selection')
            metrics,arrays = evaluate(model,loader('test',allow_test=True),device,capture=True)
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'predictions.npz',**arrays)
            save(dest/'metrics.json',metrics)
            save(dest/'COMPLETE.json',stamp)
    if 'reference_full' in protocol and component_reference(root, protocol) != protocol['reference_full']:
        raise ValueError('Full 参照在组件实验期间发生变化')
    terminal_summary=summarize(out,protocol)
    save(out/'COMPLETE.json',dict(models=len(locks),TEST_evaluated=allow_test,lock_sha256=sha(lock)))
    print(terminal_summary,flush=True)


def verify(out):
    """Check frozen identities, paired initializations and metrics against split labels."""
    out = Path(out)
    protocol = read(out/'PROTOCOL.json')
    if 'reference_full' in protocol and component_reference(project_root(), protocol) != protocol['reference_full']:
        raise ValueError('Full 参照与组件实验结果不匹配')
    complete = read(out/'COMPLETE.json')
    lock = out/'MODELS_FROZEN.json'
    locks = read(lock)
    if sha(lock)!=complete['lock_sha256']:
        raise ValueError('Checkpoint lock changed')
    expected_paths = {f'{profile}/seed{seed}/models/{key}'
                      for profile,spec in protocol['profiles'].items()
                      for seed in protocol['seeds'] for key in spec['models']}
    if set(locks)!=expected_paths or complete['models']!=len(expected_paths):
        raise ValueError('Incomplete paired model collection')
    splits = protocol['report_splits']
    if ('test' in splits)!=complete['TEST_evaluated']:
        raise ValueError('TEST authorization mismatch')
    datasets = {s:loader(s,allow_test=s=='test').dataset for s in ('train',*splits)}
    for rel,digest in locks.items():
        profile,seed_dir,_,key = Path(rel).parts
        folder = out/rel
        seed = int(seed_dir.removeprefix('seed'))
        identifier = protocol.get('family', 'progressive')+'/'+key
        spec = protocol['profiles'][profile]
        training = dict(protocol['training'],seed=seed,lr=spec['lr'],weight_decay=spec['weight_decay'])
        frozen = read(folder/'models_frozen.json')
        history = read(folder/'history.json')
        best = min(history,key=lambda row:row['VAL_5D_COMMODITY_MACRO_MAE'])
        if (sha(folder/'best.pt')!=digest or frozen['checkpoint_sha256']!=digest
                or frozen['best_epoch']!=best['epoch'] or frozen['identifier']!=identifier):
            raise ValueError(f'VAL selection mismatch at {rel}')
        model = build(identifier,initial=True,seed=seed)
        initial = tensor_hash(model.state_dict())
        for name,epoch in [('best',best['epoch']),('latest',history[-1]['epoch'])]:
            cp = torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)
            sig = cp['signature']
            if (cp['identifier']!=identifier or cp['epoch']!=epoch or sig['protocol']!=training
                    or sig['initial_state_sha256']!=initial or sig['source_hashes']!=protocol['source_hashes']
                    or sig['dataset_sha256']!=protocol['dataset_sha256'] or sig['selector']!=protocol['selector']):
                raise ValueError(f'Checkpoint signature mismatch at {rel}')
            model.load_state_dict(cp['model_state_dict'],strict=True)
        for split,ds in datasets.items():
            if split=='test':
                dest=folder.parents[1]/'test'/key
                if read(dest/'COMPLETE.json')!=dict(checkpoint_sha256=digest,lock_sha256=complete['lock_sha256']):
                    raise ValueError(f'TEST checkpoint mismatch at {rel}')
                path=dest/'predictions.npz'
                recorded=read(dest/'metrics.json')
            else:
                path=folder/(split+'_predictions.npz')
                recorded=read(folder/'train_val_metrics.json')[split]
            with np.load(path) as arrays:
                for field,expected in [('origin_dates',ds.dates),('target',ds.a[split+'_targets']),
                                       ('label_valid_mask',ds.a[split+'_label_valid_mask'])]:
                    np.testing.assert_array_equal(arrays[field],expected)
                if population(arrays['prediction'],arrays['target'],arrays['label_valid_mask'])!=recorded:
                    raise ValueError(f'Metric mismatch at {rel}/{split}')
    save(out/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',models=len(locks),splits=list(datasets)))
    print('Verified paired ablations:',out)

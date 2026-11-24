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


def comparison(scores, profile, seed, split, period='all'):
    def gain(base, score):
        return None if base is None or score is None else c4_improvement_pct(base,score)

    previous = None
    rows = []
    for key, score in scores.items():
        rows.append(dict(profile=profile, seed=seed, split=split, period=period, model=key, MAE5=score,
                         gain_vs_previous_pct=gain(previous,score), gain_vs_A_pct=gain(scores.get('A'),score),
                         full_gain_vs_model_pct=gain(score,scores.get('E'))))
        previous = score
    return rows


def summarize(out, protocol):
    rows, periods = [], []
    for profile, settings in protocol['profiles'].items():
        for seed in protocol['seeds']:
            group = out/profile/f'seed{seed}'
            for split in protocol['report_splits']:
                scores = {}
                segments = [{} for _ in range(4)]
                reference = None
                for key in settings['models']:
                    folder = group/'models'/key if split=='val' else group/'test'/key
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
                rows.extend(comparison(scores,profile,seed,split))
                for i,segment in enumerate(segments,1):
                    dates = np.array_split(reference[0],4)[i-1]
                    periods.extend(dict(r, first_origin=str(dates[0]) if len(dates) else None,
                                        last_origin=str(dates[-1]) if len(dates) else None)
                                   for r in comparison(segment,profile,seed,split,i))
    write_csv(out/'PAIRED_SEEDS.csv',rows)
    if periods:
        write_csv(out/'PERIOD_STABILITY_5D.csv',periods)
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
    lines.extend(['','逐种子与分时段差值见 PAIRED_SEEDS.csv 和 PERIOD_STABILITY_5D.csv。',
                  '各组 E 的 ROUTING_VAL.json 包含等权、TRAIN 平均权重和动态权重的冻结诊断。','',
                  '历史 Full＋12 消融附表见项目 runs/ablations_c4_tuned/seed42_20261118_01/ 下的 VAL_ABLATION.csv 和 TEST_ABLATION.csv。'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out, device, resume, allow_test, seeds, sensitivity):
    root = project_root()
    out = Path(out).resolve()
    if not out.is_relative_to(root/'runs') or out == root/'runs':
        raise ValueError('Choose a new subdirectory under runs/')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 or s>=2**32 for s in seeds):
        raise ValueError('Expected distinct uint32 seeds')
    cfg = read(root/'configs/project.json')
    profiles = dict(tuned=dict(lr=3e-4,weight_decay=1e-4,models=list(ROWS)))
    if sensitivity:
        profiles['original'] = dict(lr=1e-4, weight_decay=1e-5, models=['D','E'])
    protocol = dict(id='C4_PROGRESSIVE_V1', seeds=list(seeds), profiles=profiles, definitions=ROWS,
                    training=cfg['train'], selector=cfg['selector'], source_hashes=runtime_hashes(),
                    project_sha256=sha(root/'configs/project.json'),dataset_sha256=cfg['dataset_manifest_sha256'],
                    report_splits=['val','test'] if allow_test else ['val'])
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
                    identifier = 'progressive/'+key
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
    summarize(out,protocol)
    save(out/'COMPLETE.json',dict(models=len(locks),TEST_evaluated=allow_test,lock_sha256=sha(lock)))


def verify(out):
    """Check frozen identities, paired initializations and metrics against split labels."""
    out = Path(out)
    protocol = read(out/'PROTOCOL.json')
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
        identifier = 'progressive/'+key
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

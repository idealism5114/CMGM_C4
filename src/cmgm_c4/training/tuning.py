"""Preregistered 3x3 C4 optimizer search; TEST opens only after the VAL lock."""
from contextlib import redirect_stdout
import csv
from datetime import datetime
import io
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,sha,runtime_hashes,tensor_hash,seed_all
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import evaluate,forward
from cmgm_c4.factory import build,restore,restore_run
from cmgm_c4.verification import train_val_guard
from cmgm_c4.training.train import train,effective_protocol


def grid():
    points=[(1e-4,1e-5)]+[(lr,wd) for lr in (3e-5,1e-4,3e-4) for wd in (1e-6,1e-5,1e-4) if (lr,wd)!=(1e-4,1e-5)]
    return [dict(trial=f'T{i:02d}',lr=lr,weight_decay=wd,default_control=i==0) for i,(lr,wd) in enumerate(points)]


def choose(rows):
    if len(rows)!=9 or {r['trial'] for r in rows}!={r['trial'] for r in grid()}:
        raise ValueError('All nine VAL results are required before selection or TEST')
    if any(not np.isfinite(r['VAL5_macro_MAE']) for r in rows):raise ValueError('Nonfinite VAL selection score')
    return min(rows,key=lambda r:(r['VAL5_macro_MAE'],r['trial']))


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class Tee(io.TextIOBase):
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for s in self.streams:s.write(value);s.flush()
        return len(value)
    def flush(self):
        for s in self.streams:s.flush()


def protocol(allow_test):
    root=project_root();cfg=read(root/'configs/project.json')
    return dict(id='CMGM_C4_LR_WD_GRID_SEED42_V1',created_date='2026-11-13',model='c4/full',grid=grid(),
        fixed_training=cfg['train'],selector=cfg['selector'],shape=cfg['shape'],targets=cfg['targets'],
        source_hashes=runtime_hashes(),project_config_sha256=sha(root/'configs/project.json'),
        initial_archive_sha256=sha(root/'initial_states/c4.pt'),dataset_sha256=cfg['dataset_manifest_sha256'],
        exact_tie_rule='Smaller preregistered trial ID; control first',TEST_authorized=allow_test,
        TEST_rule='Only VAL winner and newly trained default control, after all9 complete and immutable selection lock; no TEST-based reselection',
        epochs='1..200; original epoch0 is not a candidate',unchanged='Architecture, head/graph/temporal dropout, four-horizon masked Huber, Switch KL, batch/data order, scheduler/early stopping',
        exploratory='VAL repeatedly used; historical TEST repeatedly viewed; single seed; no independent confirmation or significance claim')


def prereg_text(p):
    return '\n'.join(['# C4 学习率与权重衰减调参预注册','',
        '仅执行9组 Seed42，从同一未训练 C4 state 开始。T00为默认 lr1e-4/wd1e-5，首先训练。',
        '其余为 lr={3e-5,1e-4,3e-4} × wd={1e-6,1e-5,1e-4}。不依据中间结果修改网格。',
        '四周期masked Huberδ0.02求和 + 原Switch KL一次；所有dropout、架构、batch64、顺序、mask不变。',
        '最多200epochs，early stopping10，scheduler factor0.5/patience5；三者按VAL5商品Macro MAE，epoch0不作为候选。',
        '全部9组完成后，按完整VAL5选择配置；精确并列取较小trial ID。先保存SELECTED_MODEL.json与哈希，再打开TEST。',
        'TEST仅评价最终VAL胜出配置和本轮默认参数对照；如果两者相同只评价一次。不得根据TEST重选。',
        'TRAIN/VAL/TEST使用冻结311节点/26特征/23商品及原顺序；无重新处理、标准化或切分。',
        '只运行9组，不自动做Dropout第二轮，不运行Seed43/44。历史文件不覆盖。',
        'VAL与TEST均已多次用于历史研究，结果属于探索性。单seed调参收益不是结构创新或跨seed证据。',''])


def cuda_preflight(out,device):
    seed_all(42);m,cp=restore('c4/full',device=device);b=next(iter(loader('train')))
    m.eval()
    with torch.inference_mode():p,_=forward(m,b,device)
    root=project_root()
    with np.load(root/'results/c4/full/train_predictions.npz') as ref:
        diff=float(np.max(np.abs(p.cpu().numpy()-ref['prediction'][:len(p)])))
    if diff>1e-6:raise ValueError('GPU checkpoint reproduction failed')
    del m,p
    m=build('c4/full',initial=True,device=device);m.train();m.set_epoch(20)
    x=torch.randn(2,20,311,26,device=device)
    synthetic=dict(X=x,input_valid_mask=torch.ones(x.shape[:-1],dtype=torch.bool,device=device),
        feature_valid_mask=torch.ones_like(x,dtype=torch.bool),feature_applicable_mask=torch.ones(311,26,dtype=torch.bool,device=device))
    p=m(**synthetic);loss=p.square().mean()+m.switch_loss();loss.backward()
    finite=bool(torch.isfinite(p).all()) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    if not finite:raise ValueError('GPU synthetic backward failed')
    result=dict(status='PASS',device=device,GPU=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
        checkpoint_epoch=cp['epoch'],TRAIN_prediction_max_abs_diff=diff,synthetic_backward_finite=finite,optimizer_updates=0,
        fresh_state_sha256=tensor_hash(m.state_dict()),TEST_opened=False)
    save(out/'GPU_PREFLIGHT.json',result);del m
    if str(device).startswith('cuda'):torch.cuda.empty_cache()


def validate_trial(out,point,p):
    folder=out/'trials'/point['trial'];f=read(folder/'models_frozen.json');cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    config=cp['signature'];expected=effective_protocol(p['fixed_training'],point['lr'],point['weight_decay'])
    if config['protocol']!=expected or config['source_hashes']!=p['source_hashes'] or config['dataset_sha256']!=p['dataset_sha256']:
        raise ValueError('Trial protocol/source/data mismatch')
    if sha(folder/'best.pt')!=f['checkpoint_sha256']:raise ValueError('Trial checkpoint changed')
    history=read(folder/'history.json');selected=min(history,key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
    if selected['epoch']!=f['best_epoch']:raise ValueError('Best epoch not selected by VAL5 history')
    scores=read(folder/'train_val_metrics.json');score=scores['val']['primary']
    if abs(score-selected['VAL_5D_COMMODITY_MACRO_MAE'])>1e-9:raise ValueError('Best VAL inference mismatch')
    row=dict(**point,VAL5_macro_MAE=score,best_epoch=f['best_epoch'],last_epoch=history[-1]['epoch'],
        training_seconds=f['training_seconds'],parameters=f['parameters'],checkpoint_sha256=f['checkpoint_sha256'],
        initial_state_sha256=config['initial_state_sha256'])
    for h in ('1','5','10','20'):row['VAL_MAE_'+h+'d']=scores['val']['horizons'][h]['commodity_macro_MAE']
    return row


def lock_selection(out,rows):
    winner=choose(rows);ordered=sorted(rows,key=lambda r:(r['VAL5_macro_MAE'],r['trial']))
    write_csv(out/'VAL_RANKING.csv',ordered)
    selected=dict(rule='Minimum VAL5 commodity Macro MAE only',winner=winner['trial'],lr=winner['lr'],weight_decay=winner['weight_decay'],
        checkpoint=f"trials/{winner['trial']}/best.pt",checkpoint_sha256=winner['checkpoint_sha256'],best_epoch=winner['best_epoch'],
        VAL5_macro_MAE=winner['VAL5_macro_MAE'],grid_results=ordered,VAL_ranking_sha256=sha(out/'VAL_RANKING.csv'))
    path=out/'SELECTED_MODEL.json'
    if path.exists():
        if read(path)!=selected:raise ValueError('Frozen VAL selection changed; TEST cannot continue')
    else:save(path,selected)
    return selected


def test_locked_models(out,rows,device):
    lock=read(out/'SELECTED_MODEL.json');expected=choose(rows)
    if lock['winner']!=expected['trial'] or lock['checkpoint_sha256']!=expected['checkpoint_sha256']:raise ValueError('Invalid VAL selection lock')
    if lock['grid_results']!=sorted(rows,key=lambda r:(r['VAL5_macro_MAE'],r['trial'])) or sha(out/'VAL_RANKING.csv')!=lock['VAL_ranking_sha256']:
        raise ValueError('VAL ranking changed after selection')
    root=project_root()
    for row in read(root/'provenance/migration_manifest.json')['files']:
        if row['destination'].startswith('data/test_release/') and sha(root/row['destination'])!=row['sha256']:
            raise ValueError('Frozen TEST release changed')
    # This file is written before constructing any TEST Dataset/Loader.
    auth=out/'TEST_OPEN_AUTHORIZATION.json'
    if not auth.exists():save(auth,dict(locked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        selection_lock_sha256=sha(out/'SELECTED_MODEL.json'),winner=lock['winner'],models=sorted({lock['winner'],'T00'}),
        rule='User requested TEST; all9 completed; no further parameter fitting or selection'))
    if read(auth)['selection_lock_sha256']!=sha(out/'SELECTED_MODEL.json'):raise ValueError('TEST lock altered')
    for trial in sorted({lock['winner'],'T00'}):
        dest=out/'test'/trial
        if (dest/'metrics.json').exists():continue
        m,cp,f,cp_path=restore_run(out/'trials'/trial,device);before=sha(cp_path);m.requires_grad_(False)
        metrics,arrays=evaluate(m,loader('test',allow_test=True),device,capture=True)
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',**arrays);save(dest/'metrics.json',metrics)
        if sha(cp_path)!=before:raise ValueError('TEST mutated checkpoint')
        save(dest/'provenance.json',dict(trial=trial,selection_lock_sha256=sha(out/'SELECTED_MODEL.json'),checkpoint_sha256=before,
            best_epoch=cp['epoch'],origins=len(arrays['origin_dates']),first_origin=str(arrays['origin_dates'][0]),last_origin=str(arrays['origin_dates'][-1]),
            role='VAL winner' if trial==lock['winner'] else 'default hyperparameter control',TEST_exploratory=True))
        print('TEST',trial,'MAE5',metrics['primary'],flush=True)
    return {trial:read(out/'test'/trial/'metrics.json') for trial in sorted({lock['winner'],'T00'})}


def make_report(out,rows,test_scores):
    selected=read(out/'SELECTED_MODEL.json');winner=selected['winner'];control=next(r for r in rows if r['trial']=='T00')
    best=next(r for r in rows if r['trial']==winner);table=[];per=[];history=[]
    targets=read(project_root()/'configs/project.json')['targets']
    for r in rows:
        scores=read(out/'trials'/r['trial']/'train_val_metrics.json')
        for split,score in scores.items():
            for h,m in score['horizons'].items():table.append(dict(trial=r['trial'],split=split,horizon=int(h),**m))
            for c in score['per_commodity']:
                if c['horizon']==5:per.append(dict(trial=r['trial'],split=split,commodity=targets[c['commodity_index']],**c))
        for epoch in read(out/'trials'/r['trial']/'history.json'):
            history.append(dict(trial=r['trial'],epoch=epoch['epoch'],VAL5=epoch['VAL_5D_COMMODITY_MACRO_MAE'],
                train_Huber=epoch['train_Huber_batch_mean'],train_KL=epoch['train_KL_batch_mean'],lr=epoch['lr'],seconds=epoch['seconds'],
                optimizer_steps=epoch['optimizer_steps'],origins=epoch['train_origins_used']))
    for trial,score in test_scores.items():
        for h,m in score['horizons'].items():table.append(dict(trial=trial,split='test',horizon=int(h),**m))
        for c in score['per_commodity']:
            if c['horizon']==5:per.append(dict(trial=trial,split='test',commodity=targets[c['commodity_index']],**c))
    write_csv(out/'ALL_HORIZONS.csv',table);write_csv(out/'PER_COMMODITY_5D.csv',per);write_csv(out/'TRAINING_HISTORY.csv',history)
    lines=['# C4 学习率 × Weight Decay 调参结果','',f"VAL选择：**{winner}**，lr={best['lr']:.0e}，weight_decay={best['weight_decay']:.0e}，best epoch={best['best_epoch']}。",'',
        f"VAL5 Macro MAE={best['VAL5_macro_MAE']:.12f}；相对本轮默认T00的差值={best['VAL5_macro_MAE']-control['VAL5_macro_MAE']:+.12f}。",'',
        '| Trial | LR | Weight decay | VAL1 | VAL5 | VAL10 | VAL20 | Best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['VAL5_macro_MAE']):
        lines.append(f"| {r['trial']} | {r['lr']:.0e} | {r['weight_decay']:.0e} | {r['VAL_MAE_1d']:.9f} | {r['VAL5_macro_MAE']:.9f} | {r['VAL_MAE_10d']:.9f} | {r['VAL_MAE_20d']:.9f} | {r['best_epoch']} |")
    if test_scores:
        lines+=['','## 锁定配置后的 TEST','', '| Trial | Role | TEST1 | TEST5 | TEST10 | TEST20 | 5d RMSE | 5d Hit% |','|---|---|---:|---:|---:|---:|---:|---:|']
        for trial,s in test_scores.items():
            hs=s['horizons'];role='VAL winner' if trial==winner else 'Default control'
            lines.append(f"| {trial} | {role} | {hs['1']['commodity_macro_MAE']:.9f} | {hs['5']['commodity_macro_MAE']:.9f} | {hs['10']['commodity_macro_MAE']:.9f} | {hs['20']['commodity_macro_MAE']:.9f} | {hs['5']['RMSE']:.9f} | {100*hs['5']['Hit']:.3f} |")
        delta=test_scores[winner]['primary']-test_scores['T00']['primary']
        lines+=['',f'锁定配置相对本轮默认对照 TEST5 差值：{delta:+.12f}（负值为改善）。',
            'TEST结果不改变VAL选择；若TEST退化，如实报告，不改选其他配置。']
    lines+=['','## 边界','',
        '只训练9组Seed42。所有初值相同，只改变Adam初始学习率与weight_decay。四周期Huber、Switch KL、架构、dropout和数据不变。',
        '每epoch2726 origins/43 steps，VAL483 origins完整遍历；三个选择/调度/早停监控均为VAL5商品Macro MAE。',
        '历史TEST已多次查看，VAL也反复用于开发；结果是探索性调参，不能宣称独立盲测或跨seed显著性。',
        '没有自动训练Dropout第二轮、其他模型或Seed43/44。baseline没有获得本轮相同调参预算，不将差距归因于结构。',
        '旧数据、原项目、归档checkpoint和结果未覆盖。各trial独立保存best/latest、optimizer/scheduler/RNG及预测。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def run_search(out,device='cuda:0',resume=False,allow_test=False):
    out=Path(out).resolve();root=project_root();p=protocol(allow_test)
    if not out.is_relative_to(root/'runs'):raise ValueError('HPO must use a new CMGM_C4/runs folder')
    if out.exists() and not resume:raise ValueError('HPO output already exists')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    if resume:
        if read(out/'PREREGISTERED_PROTOCOL.json')!=p:raise ValueError('Preregistered source/protocol changed')
    else:
        save(out/'PREREGISTERED_PROTOCOL.json',p);(out/'PREREGISTERED_PROTOCOL.md').write_text(prereg_text(p))
        shutil.copytree(root/'src',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(root/'configs/project.json',out/'project_config.json')
        preflight_log=out.parent/(out.name+'_preflight.txt')
        if preflight_log.exists():shutil.copy2(preflight_log,out/'tests.txt')
        shutil.copy2(root/'tools/tune_c4.py',out/'run_entry_snapshot.py')
    started=time.perf_counter();rows=[]
    try:
        # This guard forbids TEST numeric files, labels, cached predictions and
        # even saved TEST metrics throughout preflight, training and VAL ranking.
        with train_val_guard() as hits:
            if not (out/'GPU_PREFLIGHT.json').exists():cuda_preflight(out,device)
            for point in grid():
                folder=out/'trials'/point['trial']
                if not (folder/'COMPLETE.json').exists():
                    print('START',point,flush=True)
                    with (out/'logs'/(point['trial']+'.log')).open('a') as log,redirect_stdout(Tee(sys.stdout,log)):
                        train('c4/full',folder,device,resume=folder.exists(),lr=point['lr'],weight_decay=point['weight_decay'])
                row=validate_trial(out,point,p);rows.append(row)
                if len({r['initial_state_sha256'] for r in rows})!=1:raise ValueError('Trial initial states differ')
                save(out/'PROGRESS.json',dict(completed=len(rows),total=9,rows=rows,TEST_opened=False))
                print('DONE',point['trial'],'VAL5',row['VAL5_macro_MAE'],flush=True)
            selected=lock_selection(out,rows)
            save(out/'TRAIN_VAL_ISOLATION.json',dict(status='PASS',TEST_guard_hits=hits,all9_finished_before_TEST=True,
                source_hashes_unchanged=runtime_hashes()==p['source_hashes'],initial_hashes_identical=True))
        scores=test_locked_models(out,rows,device) if allow_test else {}
        make_report(out,rows,scores)
        protected=read(root/'provenance/migration_manifest.json')['files']
        if any(sha(root/r['destination'])!=r['sha256'] for r in protected):raise ValueError('Copied historical artifact changed')
        if sha(root/'configs/project.json')!=p['project_config_sha256']:raise ValueError('Default config changed')
        if runtime_hashes()!=p['source_hashes']:raise ValueError('Runtime sources changed during search')
        save(out/'FINAL_AUDIT.json',dict(status='PASS',trials=9,seed=42,selected=selected['winner'],
            parameters_all=[r['parameters'] for r in rows],common_initial_state_hash=rows[0]['initial_state_sha256'],
            selection_lock_sha256=sha(out/'SELECTED_MODEL.json'),TEST_models=list(scores),TEST_after_selection_only=True,
            historical_files_unchanged=len(protected),default_config_unchanged=True,source_unchanged=True,seconds=time.perf_counter()-started,
            no_extra_trials=True,no_additional_seeds=True))
        index={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='file_index.json'}
        save(out/'file_index.json',index);save(out/'COMPLETE.json',dict(status='COMPLETE',trials=9,selected=selected['winner'],TEST_evaluated=bool(scores)))
        print('ALL9 COMPLETE; REPORT',out/'REPORT.md',flush=True)
    except BaseException:
        save(out/'INTERRUPTION.json',dict(error=traceback.format_exc(),completed_trials=len(rows),TEST_opened=(out/'TEST_OPEN_AUTHORIZATION.json').exists()))
        raise

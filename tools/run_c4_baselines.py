"""Eight fresh Deep V3 baselines with current tuned defaults; TEST after all8 locks.

Default: CPU preflight only. Explicit --execute trains; --allow-test opens
frozen TEST only after all eight independent VAL checkpoint selections.
No architecture edits, hyperparameter searches or trained-weight initialization.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
MODELS=('RNN','GRU','LSTM','VanillaTransformer','GraphWaveNet','MTGNN','MSGNet','CrossGNN')
REFERENCE=Path('runs/ablations_c4_tuned/seed42_20261118_01')


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def freeze_all(out,rows):
    if len(rows)!=len(MODELS) or {r['key'] for r in rows}!=set(MODELS):
        raise ValueError('All eight VAL selections required before TEST')
    lock=dict(selection='VAL_5D_COMMODITY_MACRO_MAE',models={r['key']:r for r in rows},
        no_TEST_reselection=True,reference_C4=str(REFERENCE/'models/full_c4'))
    path=Path(out)/'MODELS_FROZEN.json'
    if path.exists() and read(path)!=lock:raise ValueError('Locked VAL selections changed')
    save(path,lock);return lock


def protocol(allow_test):
    from cmgm_c4.common import runtime_hashes
    cfg=read(ROOT/'configs/project.json');train=cfg['train']
    if train['lr']!=3e-4 or train['weight_decay']!=1e-4:raise ValueError('Expected user-approved tuned defaults')
    registry=read(ROOT/'configs/registry.json')
    if {x.split('/')[1] for x in registry if x.startswith('baselines/')}!=set(MODELS):raise ValueError('Baseline identity changed')
    ref=read(ROOT/REFERENCE/'PREREGISTERED_PROTOCOL.json')
    if ref['fixed_training']!=train or ref['selector']!=cfg['selector']:raise ValueError('C4 comparison protocol mismatch')
    if ref['dataset_sha256']!=cfg['dataset_manifest_sha256']:raise ValueError('C4 comparison dataset mismatch')
    # Verify original model/trainer implementations; new runner lives outside src.
    hashes=runtime_hashes()
    if hashes!=ref['source_hashes']:raise ValueError('Runtime changed since matched C4 ablation')
    frozen=read(ROOT/REFERENCE/'models/full_c4/models_frozen.json')
    if sha(ROOT/REFERENCE/'models/full_c4/best.pt')!=frozen['checkpoint_sha256']:raise ValueError('C4 reference checkpoint changed')
    return dict(id='CMGM_C4_DEEP_V3_TUNED_BASELINES_SEED42_V1',models=list(MODELS),
        fixed_training=train,selector=cfg['selector'],source_hashes=hashes,
        project_config_sha256=sha(ROOT/'configs/project.json'),runner_sha256=sha(Path(__file__)),
        dataset_sha256=cfg['dataset_manifest_sha256'],shape=cfg['shape'],targets=cfg['targets'],
        baseline_input=['B',20,27,26],C4_input=['B',20,311,26],output=['B',4,23],
        initial_archive_sha256=sha(ROOT/'initial_states/baselines.pt'),
        reference_C4=str(REFERENCE/'models/full_c4'),reference_C4_checkpoint_sha256=frozen['checkpoint_sha256'],
        comparison='Same frozen target/date/mask/selector and optimizer settings; original V3 compressed input differs from full-node C4',
        KL='Baselines have no Switch KL; C4 retains native Switch KL by its original definition',
        initialization='Strict archived untrained seed42 state per architecture; no warm start',
        TEST_authorized=allow_test,TEST_rule='Only after all8 TRAIN/VAL results and checkpoint hashes are locked; no reselection',
        limits='Single seed, fixed C4-selected lr/wd, not individually tuned baseline optima; reused VAL/TEST are exploratory')


def preflight():
    import pytest
    import torch
    from cmgm_c4.verification import train_val_guard
    torch.set_num_threads(2)
    with train_val_guard():
        return pytest.main(['-q',str(ROOT/'tests'),'-k','not test_summary_uses_historical_json_not_test_arrays'])


def prepare(out,device,p):
    import numpy as np
    import torch
    from cmgm_c4.common import audit_train_val_files,tensor_hash,prediction_loss,seed_all
    from cmgm_c4.factory import build
    from cmgm_c4.data.dataset import loader
    data=audit_train_val_files();alignment={}
    for split in ('train','val'):
        ds=loader(split,neutral=True).dataset
        with np.load(ROOT/p['reference_C4']/(split+'_predictions.npz')) as reference:
            for name,value in [('target',ds.a[split+'_targets']),('label_valid_mask',ds.a[split+'_label_valid_mask']),('origin_dates',ds.dates)]:
                np.testing.assert_array_equal(reference[name],value)
        alignment[split]=dict(origins=len(ds),first_origin=str(ds.dates[0]),last_origin=str(ds.dates[-1]),
            same_dates_targets_masks_as_C4=True,input_shape=list(ds[0]['X'].shape))
    save(out/'DATA_AUDIT.json',dict(status='PASS',frozen_files=data,alignment=alignment,TEST_numeric_access=False))
    records={}
    for key in MODELS:
        seed_all(42);m=build('baselines/'+key,initial=True,device=device)
        before=tensor_hash(m.state_dict());m.train()
        x=torch.randn(2,20,27,26,device=device);pred=m(x)
        if pred.shape!=(2,4,23) or not torch.isfinite(pred).all():raise ValueError('Invalid baseline output '+key)
        y=torch.randn_like(pred)*.02;loss=prediction_loss(pred,y,torch.ones_like(pred,dtype=torch.bool));loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('Nonfinite gradients '+key)
        unused={n:v.numel() for n,v in m.named_parameters() if v.grad is None}
        if sum(unused.values())!=(3104 if key=='GraphWaveNet' else 0):raise ValueError('Unexpected disconnected parameters '+key+' '+str(unused))
        # BatchNorm buffers may update during synthetic train forward; no parameter update.
        records[key]=dict(initial_state_sha256=before,parameters=sum(v.numel() for v in m.parameters()),
            trainable_parameters=sum(v.numel() for v in m.parameters() if v.requires_grad),
            structurally_unused_parameter_count=sum(unused.values()),unused_tensors=unused,
            gradient_connected_parameters=sum(v.numel() for v in m.parameters() if v.grad is not None),
            synthetic_shape=[2,4,23],gradients_finite=True,optimizer_updates=0)
        del m
    save(out/'INITIALIZATION_AND_GRADIENT_AUDIT.json',dict(status='PASS',device=device,models=records,TEST_read=False))
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return records


def validate_run(out,key,p,initial):
    import numpy as np
    import torch
    from cmgm_c4.factory import build
    from cmgm_c4.evaluation.metrics import population
    folder=out/'models'/key;meta=read(folder/'models_frozen.json')
    history=read(folder/'history.json');scores=read(folder/'train_val_metrics.json')
    selected=min(history,key=lambda e:e['VAL_5D_COMMODITY_MACRO_MAE'])
    if selected['epoch']!=meta['best_epoch']:raise ValueError('VAL epoch mismatch '+key)
    checkpoint_records={}
    for which in ('best','latest'):
        path=folder/(which+'.pt');cp=torch.load(path,map_location='cpu',weights_only=False)
        m=build('baselines/'+key);m.load_state_dict(cp['model_state_dict'],strict=True)
        signature=cp['signature']
        if signature['protocol']!=p['fixed_training'] or signature['source_hashes']!=p['source_hashes']:raise ValueError('Runtime/protocol mismatch '+key)
        if signature['initial_state_sha256']!=initial['initial_state_sha256']:raise ValueError('Warm-start or wrong initialization '+key)
        if signature['dataset_sha256']!=p['dataset_sha256']:raise ValueError('Dataset fingerprint mismatch '+key)
        if signature['selector']!=p['selector']:raise ValueError('Selection protocol mismatch '+key)
        steps=[float(s['step']) for s in cp['optimizer']['state'].values()]
        if not steps or min(steps)!=cp['epoch']*43 or max(steps)!=cp['epoch']*43:raise ValueError('Optimizer step mismatch '+key)
        if which=='best' and (cp['epoch']!=selected['epoch'] or sha(path)!=meta['checkpoint_sha256']):raise ValueError('Best checkpoint mismatch '+key)
        if which=='latest' and cp['epoch']!=history[-1]['epoch']:raise ValueError('Latest checkpoint mismatch '+key)
        if any(e['optimizer_steps']!=43 or e['train_origins_used']!=2726 or e['train_KL_batch_mean']!=0 for e in history):raise ValueError('Coverage or baseline KL mismatch '+key)
        checkpoint_records[which]=dict(strict_restore=True,epoch=cp['epoch'],optimizer_steps=min(steps),checkpoint_sha256=sha(path))
        del m,cp
    for split in ('train','val'):
        with np.load(folder/(split+'_predictions.npz')) as arrays,np.load(ROOT/p['reference_C4']/(split+'_predictions.npz')) as ref:
            if population(arrays['prediction'],arrays['target'],arrays['label_valid_mask'])!=scores[split]:raise ValueError('Metric recomputation mismatch '+key)
            for name in ('target','label_valid_mask','origin_dates'):np.testing.assert_array_equal(arrays[name],ref[name])
    if abs(selected['VAL_5D_COMMODITY_MACRO_MAE']-scores['val']['primary'])>1e-9:raise ValueError('VAL selected score mismatch '+key)
    save(folder/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',checkpoints=checkpoint_records,metrics_exactly_recomputed=True,dates_targets_masks_equal_C4=True))
    return dict(key=key,identifier='baselines/'+key,VAL5_macro_MAE=scores['val']['primary'],
        best_epoch=meta['best_epoch'],last_epoch=history[-1]['epoch'],parameters=meta['parameters'],
        trainable_parameters=meta['trainable_parameters'],gradient_connected_parameters=initial['gradient_connected_parameters'],
        training_seconds=meta['training_seconds'],checkpoint=str((folder/'best.pt').relative_to(out)),
        checkpoint_sha256=meta['checkpoint_sha256'])


def evaluate_test(out,rows,device,p):
    # Gate BEFORE importing a loader or inspecting any TEST manifest/value.
    if not (out/'MODELS_FROZEN.json').exists():raise ValueError('All eight VAL selections required before TEST')
    locked=freeze_all(out,rows)
    if not p['TEST_authorized']:raise ValueError('Explicit TEST authorization required')
    import numpy as np
    from cmgm_c4.factory import restore_run
    from cmgm_c4.data.dataset import loader
    from cmgm_c4.evaluation.evaluate import evaluate
    from cmgm_c4.evaluation.metrics import population
    lock_hash=sha(out/'MODELS_FROZEN.json')
    save(out/'TEST_OPEN_AUTHORIZATION.json',dict(models_frozen_sha256=lock_hash,
        authorization='Existing user instructions to train and report TEST for the matched C4/baseline suite',
        rule=p['TEST_rule'],all8_VAL_selected=True,TEST_used_for_selection=False))
    for r in read(ROOT/'provenance/migration_manifest.json')['files']:
        if r['category']=='frozen_TEST_bundle' and sha(ROOT/r['destination'])!=r['sha256']:raise ValueError('Frozen TEST bytes changed')
    scores={}
    for key in MODELS:
        dest=out/'test'/key
        if (dest/'COMPLETE.json').exists():
            cp_path=out/'models'/key/'best.pt'
            if read(dest/'provenance.json')['checkpoint_sha256']!=sha(cp_path):raise ValueError('Resumed TEST checkpoint changed')
            scores[key]=read(dest/'metrics.json');continue
        m,cp,_,path=restore_run(out/'models'/key,device);m.requires_grad_(False)
        if sha(path)!=locked['models'][key]['checkpoint_sha256']:raise ValueError('Unlocked TEST checkpoint')
        metrics,a=evaluate(m,loader('test',neutral=True,allow_test=True),device,neutral=True,capture=True)
        with np.load(ROOT/REFERENCE/'test/full_c4/predictions.npz') as ref:
            for name in ('target','label_valid_mask','origin_dates'):np.testing.assert_array_equal(a[name],ref[name])
        if population(a['prediction'],a['target'],a['label_valid_mask'])!=metrics:raise ValueError('TEST metric recomputation failed')
        if sha(path)!=locked['models'][key]['checkpoint_sha256']:raise ValueError('TEST changed checkpoint')
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',**a);save(dest/'metrics.json',metrics)
        save(dest/'provenance.json',dict(checkpoint_sha256=sha(path),best_epoch=cp['epoch'],models_frozen_sha256=lock_hash,
            same_dates_targets_masks_as_C4=True,origins=len(a['prediction']),first_origin=str(a['origin_dates'][0]),last_origin=str(a['origin_dates'][-1]),
            exploratory_reused_TEST=True))
        save(dest/'COMPLETE.json',dict(status='COMPLETE',TEST_after_all8_lock=True));scores[key]=metrics
        print('TEST',key,metrics['primary'],flush=True)
        del m
    return scores


def summarize(out,rows,test,p):
    from cmgm_c4.training.tuning import write_csv
    all_rows=[];per=[];history=[];val=[];test_rows=[]
    c4_scores=read(ROOT/p['reference_C4']/'train_val_metrics.json')
    c4_meta=read(ROOT/p['reference_C4']/'models_frozen.json')
    common=dict(key='C4',identifier='ablations/full_c4',best_epoch=c4_meta['best_epoch'],parameters=c4_meta['parameters'])
    val.append(dict(**common,VAL5_macro_MAE=c4_scores['val']['primary'],delta_vs_C4=0.))
    if test:
        c4_test=read(ROOT/REFERENCE/'test/full_c4/metrics.json')
        test_rows.append(dict(**common,TEST5_macro_MAE=c4_test['primary'],delta_vs_C4=0.))
    else:c4_test={}
    for row in rows:
        key=row['key'];scores=read(out/'models'/key/'train_val_metrics.json');meta=dict(key=key,identifier=row['identifier'],best_epoch=row['best_epoch'],parameters=row['parameters'])
        val.append(dict(**meta,VAL5_macro_MAE=scores['val']['primary'],delta_vs_C4=scores['val']['primary']-c4_scores['val']['primary']))
        if key in test:
            scores['test']=test[key]
            test_rows.append(dict(**meta,TEST5_macro_MAE=test[key]['primary'],delta_vs_C4=test[key]['primary']-c4_test['primary']))
        for split,score in scores.items():
            for h,m in score['horizons'].items():all_rows.append(dict(key=key,split=split,horizon=int(h),**m))
            for c in score['per_commodity']:
                if c['horizon']==5:per.append(dict(key=key,split=split,commodity=p['targets'][c['commodity_index']],**c))
        for e in read(out/'models'/key/'history.json'):
            history.append(dict(key=key,epoch=e['epoch'],VAL5=e['VAL_5D_COMMODITY_MACRO_MAE'],train_Huber=e['train_Huber_batch_mean'],train_KL=e['train_KL_batch_mean'],
                optimizer_steps=e['optimizer_steps'],train_origins=e['train_origins_used'],lr=e['lr'],seconds=e['seconds']))
    for split,score in list(c4_scores.items())+([('test',c4_test)] if test else []):
        for h,m in score['horizons'].items():all_rows.append(dict(key='C4',split=split,horizon=int(h),**m))
        for c in score['per_commodity']:
            if c['horizon']==5:per.append(dict(key='C4',split=split,commodity=p['targets'][c['commodity_index']],**c))
    write_csv(out/'VAL_RANKING.csv',sorted(val,key=lambda x:x['VAL5_macro_MAE']))
    if test_rows:write_csv(out/'TEST_COMPARISON.csv',test_rows)
    write_csv(out/'ALL_HORIZONS.csv',all_rows);write_csv(out/'PER_COMMODITY_5D.csv',per);write_csv(out/'TRAINING_HISTORY.csv',history)
    write_csv(out/'PARAMETER_AND_TIME_AUDIT.csv',rows)
    lines=['# 调优参数下的8个Deep V3 baseline与C4','',
        '8个baseline均从各自保存的未训练Seed42初值重新训练。Adam初始lr=3e-4、weight_decay=1e-4、batch64、最多200epochs、early stopping10、scheduler factor0.5/patience5。',
        '四周期masked Huberδ0.02均值后求和；baseline不包含C4专有的Switch KL。所有checkpoint、scheduler和early stopping均只按VAL5商品Macro MAE。',
        'C4合法复用本轮消融Full：相同源码/默认训练参数/标签日期mask/选模标准，最佳epoch15。','',
        '| Model | VAL5 Macro MAE | TEST5 Macro MAE | Δ TEST5 vs C4 | Params | Best epoch |','|---|---:|---:|---:|---:|---:|']
    by_test={x['key']:x for x in test_rows}
    for r in sorted(val,key=lambda x:x['VAL5_macro_MAE']):
        t=by_test.get(r['key']);columns=(f"{t['TEST5_macro_MAE']:.12f} | {t['delta_vs_C4']:+.12f}" if t else 'not evaluated | not evaluated')
        lines.append(f"| {r['key']} | {r['VAL5_macro_MAE']:.12f} | {columns} | {r['parameters']} | {r['best_epoch']} |")
    lines+=['','## 解释边界','',
        '表按VAL排序，TEST仅是所有8组VAL checkpoint锁定后的冻结评价，不按TEST选模型、epoch或修改默认主模型。',
        '使用Full调优后固定的超参数，并未分别调优每个baseline，因此不代表各架构的超参数最优性能。',
        'V3 baseline输入为27×26中性token（股票均值/标准差、债券均值/标准差、23商品）；C4读取311完整节点，输入读出能力与参数容量不完全匹配。',
        '同一seed不表示不同架构参数值相同：本轮严格复用每种架构自己的未训练初始归档。无任何已训练baseline/C4权重warm start。',
        'GraphWaveNet按历史定义保留3104个结构上无预测梯度的参数；报告名义参数数与有梯度连接的参数数。',
        '所有模型使用同一预算上限与停止规则，实际轮数与耗时不同。未追加seed、架构、超参数搜索或失败后择优重跑。',
        '单Seed42且VAL/TEST历史多次查看，结果属于探索性，不是独立确认性盲测。','',
        '四周期指标见ALL_HORIZONS.csv，全部23商品见PER_COMMODITY_5D.csv，逐epoch见TRAINING_HISTORY.csv。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out,device,resume,allow_test):
    import torch
    from cmgm_c4.common import runtime_hashes
    from cmgm_c4.training.train import train
    from cmgm_c4.training.tuning import Tee
    from cmgm_c4.verification import train_val_guard
    torch.set_num_threads(4);p=protocol(allow_test);out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'runs'):raise ValueError('Use new standalone runs/ directory')
    if out.exists() and not resume:raise ValueError('Existing output must not be overwritten')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    if resume:
        if read(out/'PREREGISTERED_PROTOCOL.json')!=p:raise ValueError('Resume source/default/protocol changed')
    else:
        save(out/'PREREGISTERED_PROTOCOL.json',p);shutil.copytree(ROOT/'src',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'configs/project.json',out/'project_config.json');shutil.copy2(Path(__file__),out/'run_entry_snapshot.py')
        shutil.copy2(out.parent/(out.name+'_preflight.txt'),out/'tests.txt')
        save(out/'LAUNCH_ENVIRONMENT.json',dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),device=device,threads=torch.get_num_threads(),
            GPU_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',python=sys.executable))
    rows=[];start=time.perf_counter()
    try:
        with train_val_guard() as hits:
            if (out/'INITIALIZATION_AND_GRADIENT_AUDIT.json').exists():initial=read(out/'INITIALIZATION_AND_GRADIENT_AUDIT.json')['models']
            else:initial=prepare(out,device,p)
            for key in MODELS:
                folder=out/'models'/key
                if not (folder/'COMPLETE.json').exists():
                    print('START',key,'lr',p['fixed_training']['lr'],'wd',p['fixed_training']['weight_decay'],flush=True)
                    with (out/'logs'/(key+'.log')).open('a') as log,redirect_stdout(Tee(sys.stdout,log)):
                        train('baselines/'+key,folder,device,resume=folder.exists())
                row=validate_run(out,key,p,initial[key]);rows.append(row)
                save(out/'PROGRESS.json',dict(completed=len(rows),total=len(MODELS),rows=rows,TEST_opened=False))
                print('DONE',key,'VAL5',row['VAL5_macro_MAE'],flush=True)
            freeze_all(out,rows)
            save(out/'TRAIN_VAL_ISOLATION.json',dict(status='PASS',blocked_attempts=hits,all8_frozen_before_TEST=True))
        test=evaluate_test(out,rows,device,p) if allow_test else {}
        summarize(out,rows,test,p)
        if runtime_hashes()!=p['source_hashes'] or sha(ROOT/'configs/project.json')!=p['project_config_sha256']:raise ValueError('Sources/default config changed')
        archive=read(ROOT/'provenance/migration_manifest.json')['files']
        if any(sha(ROOT/r['destination'])!=r['sha256'] for r in archive):raise ValueError('Historical archive changed')
        save(out/'FINAL_AUDIT.json',dict(status='PASS',models=8,seed=42,all_from_untrained_states=True,
            strict_best_latest_verified=True,metrics_recomputed=True,source_unchanged=True,default_config_unchanged=True,
            protected_migration_files_unchanged=len(archive),TEST_models=list(test),TEST_after_all8_freeze=True,
            models_frozen_sha256=sha(out/'MODELS_FROZEN.json'),seconds=time.perf_counter()-start,no_extra_models_or_seeds=True))
        save(out/'COMPLETE.json',dict(status='COMPLETE',models=8,TEST_evaluated=bool(test)))
        save(out/'file_index.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='file_index.json'})
        print('ALL8 COMPLETE',out/'REPORT.md',flush=True)
    except BaseException:
        save(out/'INTERRUPTION.json',dict(traceback=traceback.format_exc(),completed=len(rows),TEST_opened=(out/'TEST_OPEN_AUTHORIZATION.json').exists()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu',default='0')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-test',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-child',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    if args.preflight_child or not args.execute:return preflight()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),'--preflight-child','--gpu',args.gpu],cwd=ROOT,capture_output=True,text=True)
    log=args.output.parent/(args.output.name+'_preflight.txt')
    with log.open('a' if args.resume else 'x') as f:f.write(child.stdout+child.stderr)
    print(child.stdout+child.stderr,flush=True)
    if child.returncode:return child.returncode
    run(args.output,args.device,args.resume,args.allow_test)


if __name__=='__main__':raise SystemExit(main())

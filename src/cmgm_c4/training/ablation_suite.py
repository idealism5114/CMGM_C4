"""Thirteen fixed C4 ablations, matched tuned optimizer and VAL-only freezing."""
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,sha,runtime_hashes,tensor_hash,prediction_loss
from cmgm_c4.factory import build,restore_run
from cmgm_c4.models.ablations import KEYS,LABELS,DEFS
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import evaluate
from cmgm_c4.evaluation.metrics import population
from cmgm_c4.verification import train_val_guard
from .train import train,effective_protocol
from .tuning import Tee,write_csv

REFERENCE='runs/tuning_lr_wd/seed42_20261113_01'


def make_protocol(allow_test):
    root=project_root();cfg=read(root/'configs/project.json');selected=read(root/REFERENCE/'SELECTED_MODEL.json')
    old=read(root/REFERENCE/'PREREGISTERED_PROTOCOL.json')
    for rel,h in old['source_hashes'].items():
        if sha(root/rel)!=h:raise ValueError('Existing model/trainer changed since HPO: '+rel)
    if read(root/REFERENCE/'INDEPENDENT_VERIFICATION.json')['status']!='PASS':raise ValueError('HPO reference not verified')
    if selected['lr']!=3e-4 or selected['weight_decay']!=1e-4:raise ValueError('Unexpected tuned C4 identity')
    definitions={k:DEFS[k] for k in KEYS}
    definitions['full_c4']='Original C4, freshly retrained with tuned lr3e-4/wd1e-4; no trained-weight initialization'
    definitions['no_moe']='Original D0B gate/head; fresh matched C0 initial state and same tuned optimizer, not reused epoch46'
    return dict(id='CMGM_C4_TUNED_ABLATION_SEED42_V1',date='2026-11-18',models=list(KEYS),definitions=definitions,
        fixed_training=effective_protocol(cfg['train'],3e-4,1e-4),selector=cfg['selector'],shape=cfg['shape'],targets=cfg['targets'],
        source_hashes=runtime_hashes(),project_config_sha256=sha(root/'configs/project.json'),dataset_sha256=cfg['dataset_manifest_sha256'],
        reference=REFERENCE,reference_selection_lock_sha256=sha(root/REFERENCE/'SELECTED_MODEL.json'),
        reference_best_checkpoint=selected['checkpoint'],reference_checkpoint_sha256=selected['checkpoint_sha256'],
        initial_archives={name:sha(root/'initial_states'/name) for name in ('c4.pt','ablations.pt')},
        inactive_parameters=cfg['inactive_parameters'],TEST_authorized=allow_test,
        selection_rule='Each ablation selects its own best VAL5 checkpoint; Full remains the prespecified main model',
        TEST_rule='Only after all13 TRAIN/VAL runs and all checkpoint SHA/epochs are locked; no TEST-based selection or definition change',
        limits='Fixed Full-selected hyperparameters; ablations not individually tuned; capacity/random execution can differ; single Seed42 and repeatedly viewed historical TEST')


def initialization_audit():
    full=build('c4/full',initial=True).state_dict();records={}
    for key in KEYS:
        m=build('ablations/'+key,initial=True);state=m.state_dict();common=sorted(set(full)&set(state))
        mismatches=[k for k in common if not torch.equal(full[k],state[k])]
        if mismatches:raise ValueError('Unmatched common initialization: '+key+' '+str(mismatches))
        records[key]=dict(initial_state_sha256=tensor_hash(state),common_key_count=len(common),common_max_difference=0,
            extra_keys=sorted(set(state)-set(full)),missing_keys=sorted(set(full)-set(state)),
            nominal_parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in m.parameters() if not p.requires_grad))
    if records['full_c4']['initial_state_sha256']!=tensor_hash(full):raise ValueError('Full initialization differs from C4')
    return dict(status='PASS',Full_C4_initial_state_sha256=tensor_hash(full),records=records,
        note='Frozen inactive tensors remain in state_dict for historical strict restore; trainable count is not a proof of statistical effective capacity')


def gpu_preflight(out,device,p):
    root=project_root();m=build('ablations/full_c4',device=device)
    cp=torch.load(root/p['reference']/p['reference_best_checkpoint'],map_location='cpu',weights_only=False)
    if sha(root/p['reference']/p['reference_best_checkpoint'])!=p['reference_checkpoint_sha256']:raise ValueError('Reference checkpoint changed')
    m.load_state_dict(cp['model_state_dict'],strict=True);m.set_epoch(cp['epoch'])
    metrics,arrays=evaluate(m,loader('val'),device,capture=True)
    with np.load(root/p['reference']/'trials/T08/val_predictions.npz') as old:
        for name in ('target','label_valid_mask','origin_dates'):np.testing.assert_array_equal(arrays[name],old[name])
        diff=float(np.abs(arrays['prediction']-old['prediction']).max())
        if diff>1e-6:raise ValueError('Full ablation does not reproduce tuned C4')
    records={}
    for key in KEYS:
        m=build('ablations/'+key,initial=True,device=device);m.train();m.set_epoch(20)
        x=torch.randn(2,20,311,26,device=device);target=torch.randn(2,4,23,device=device)*.02
        batch=dict(X=x,input_valid_mask=torch.ones(x.shape[:-1],dtype=torch.bool,device=device),
            feature_valid_mask=torch.ones_like(x,dtype=torch.bool),feature_applicable_mask=torch.ones(311,26,dtype=torch.bool,device=device))
        pred=m(**batch);loss=prediction_loss(pred,target,torch.ones_like(target,dtype=torch.bool))+m.switch_loss();loss.backward()
        if pred.shape!=(2,4,23) or not torch.isfinite(pred).all():raise ValueError('Invalid output '+key)
        missing=[n for n,v in m.named_parameters() if v.requires_grad and v.grad is None]
        if missing:raise ValueError('Unexpected disconnected trainable tensors '+key+' '+str(missing))
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('Nonfinite gradient '+key)
        records[key]=dict(output_shape=list(pred.shape),all_trainable_tensors_connected=True,finite_gradients=True)
    save(out/'GPU_PREFLIGHT.json',dict(status='PASS',device=device,reference_VAL5=metrics['primary'],
        reference_prediction_max_difference=diff,synthetic_checks=records,optimizer_updates=0,TEST_read=False))


def validate_run(out,key,p,init):
    folder=out/'models'/key;meta=read(folder/'models_frozen.json');cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    history=read(folder/'history.json');scores=read(folder/'train_val_metrics.json');chosen=min(history,key=lambda e:e['VAL_5D_COMMODITY_MACRO_MAE'])
    signature=cp['signature']
    if signature['protocol']!=p['fixed_training'] or signature['source_hashes']!=p['source_hashes']:raise ValueError('Protocol/source mismatch '+key)
    if signature['initial_state_sha256']!=init['records'][key]['initial_state_sha256']:raise ValueError('Wrong initial state '+key)
    if signature['dataset_sha256']!=p['dataset_sha256']:raise ValueError('Data mismatch '+key)
    if chosen['epoch']!=meta['best_epoch'] or cp['epoch']!=meta['best_epoch']:raise ValueError('Epoch selection mismatch '+key)
    if abs(chosen['VAL_5D_COMMODITY_MACRO_MAE']-scores['val']['primary'])>1e-9:raise ValueError('VAL mismatch '+key)
    if sha(folder/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Checkpoint changed '+key)
    row=dict(key=key,label=LABELS.get(key,key),VAL5_macro_MAE=scores['val']['primary'],best_epoch=meta['best_epoch'],last_epoch=history[-1]['epoch'],
        nominal_parameters=meta['parameters'],trainable_parameters=meta['trainable_parameters'],training_seconds=meta['training_seconds'],
        checkpoint=str((folder/'best.pt').relative_to(out)),checkpoint_sha256=meta['checkpoint_sha256'])
    for h in ('1','5','10','20'):row['VAL_MAE_'+h+'d']=scores['val']['horizons'][h]['commodity_macro_MAE']
    return row


def freeze_all(out,rows):
    if len(rows)!=13 or {r['key'] for r in rows}!=set(KEYS):raise ValueError('All13 must finish before freezing or TEST')
    frozen=dict(selector='VAL_5D_COMMODITY_MACRO_MAE',main_model='full_c4',models={r['key']:r for r in rows})
    path=out/'MODELS_FROZEN.json'
    if path.exists():
        if read(path)!=frozen:raise ValueError('Locked checkpoints changed')
    else:save(path,frozen)
    return frozen


def evaluate_test(out,rows,device):
    frozen=freeze_all(out,rows);lock=sha(out/'MODELS_FROZEN.json');auth=out/'TEST_OPEN_AUTHORIZATION.json'
    if not auth.exists():save(auth,dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),models_frozen_sha256=lock,
        rule='Prior user-approved C4 ablation+TEST protocol; all13 checkpoints locked by VAL before TEST',models=list(KEYS)))
    if read(auth)['models_frozen_sha256']!=lock:raise ValueError('TEST authorization lock changed')
    root=project_root()
    for r in read(root/'provenance/migration_manifest.json')['files']:
        if r['destination'].startswith('data/test_release/') and sha(root/r['destination'])!=r['sha256']:raise ValueError('Frozen TEST changed')
    scores={}
    for key in KEYS:
        dest=out/'test'/key;row=frozen['models'][key]
        if (dest/'metrics.json').exists():scores[key]=read(dest/'metrics.json');continue
        m,cp,_,path=restore_run(out/'models'/key,device);before=sha(path);m.requires_grad_(False)
        if before!=row['checkpoint_sha256']:raise ValueError('Wrong locked TEST checkpoint')
        metrics,arrays=evaluate(m,loader('test',allow_test=True),device,capture=True)
        if sha(path)!=before:raise ValueError('TEST modified checkpoint')
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',**arrays);save(dest/'metrics.json',metrics)
        save(dest/'provenance.json',dict(models_frozen_sha256=lock,checkpoint_sha256=before,best_epoch=cp['epoch'],origins=len(arrays['prediction']),
            first_origin=str(arrays['origin_dates'][0]),last_origin=str(arrays['origin_dates'][-1]),TEST_exploratory=True))
        scores[key]=metrics;print('TEST',key,metrics['primary'],flush=True)
    return scores


def summarize(out,rows,test,p):
    full=next(r for r in rows if r['key']=='full_c4');all_metrics=[];per=[];history=[];val_table=[];test_table=[]
    for row in rows:
        key=row['key'];folder=out/'models'/key;scores=read(folder/'train_val_metrics.json')
        v=dict(**row,delta_VAL5_vs_Full=row['VAL5_macro_MAE']-full['VAL5_macro_MAE']);val_table.append(v)
        if key in test:
            scores['test']=test[key];test_table.append(dict(key=key,TEST5_macro_MAE=test[key]['primary'],delta_vs_Full=test[key]['primary']-test['full_c4']['primary'],
                best_epoch=row['best_epoch'],nominal_parameters=row['nominal_parameters'],trainable_parameters=row['trainable_parameters'],
                **{f'MAE_{h}d':test[key]['horizons'][h]['commodity_macro_MAE'] for h in ('1','5','10','20')}))
        for split,score in scores.items():
            for h,m in score['horizons'].items():all_metrics.append(dict(key=key,split=split,horizon=int(h),**m))
            for c in score['per_commodity']:
                if c['horizon']==5:per.append(dict(key=key,split=split,commodity=p['targets'][c['commodity_index']],**c))
        for e in read(folder/'history.json'):
            history.append(dict(key=key,epoch=e['epoch'],VAL5=e['VAL_5D_COMMODITY_MACRO_MAE'],train_Huber=e['train_Huber_batch_mean'],
                train_KL=e['train_KL_batch_mean'],lr=e['lr'],seconds=e['seconds'],optimizer_steps=e['optimizer_steps'],train_origins=e['train_origins_used']))
    write_csv(out/'VAL_ABLATION.csv',val_table);write_csv(out/'VAL_RANKING.csv',sorted(val_table,key=lambda r:r['VAL5_macro_MAE']))
    if test_table:write_csv(out/'TEST_ABLATION.csv',test_table)
    write_csv(out/'ALL_HORIZONS.csv',all_metrics);write_csv(out/'PER_COMMODITY_5D.csv',per);write_csv(out/'TRAINING_HISTORY.csv',history)
    lines=['# 调优 C4 的 Full＋12 消融','',
        '所有13组均从头训练Seed42：lr3e-4、weight_decay1e-4；四周期masked Huberδ0.02、原Switch KL、数据和选择协议一致。',
        '各组按自己的VAL5商品Macro MAE选择checkpoint。Full为预定主模型，不因消融或TEST排名而切换。','',
        '| Model | VAL5 Macro MAE | Δ vs Full | Trainable params | Best epoch |','|---|---:|---:|---:|---:|']
    for r in val_table:lines.append(f"| {r['key']} | {r['VAL5_macro_MAE']:.12f} | {r['delta_VAL5_vs_Full']:+.12f} | {r['trainable_parameters']} | {r['best_epoch']} |")
    if test_table:
        lines+=['','## 全部VAL checkpoint锁定后的TEST','',
            '| Model | TEST1 | TEST5 | TEST10 | TEST20 | Δ TEST5 vs Full |','|---|---:|---:|---:|---:|---:|']
        for r in test_table:lines.append(f"| {r['key']} | {r['MAE_1d']:.9f} | {r['MAE_5d']:.9f} | {r['MAE_10d']:.9f} | {r['MAE_20d']:.9f} | {r['delta_vs_Full']:+.9f} |")
    with np.load(out/'models/full_c4/val_predictions.npz') as a:
        zero=population(np.zeros_like(a['prediction']),a['target'],a['label_valid_mask'])['primary']
    lines+=['','## 定义与解释边界','',
        '无动态路由=四专家固定等权，不是可学习静态路由；无MoE=原D0B通道gate，不是只删去router。',
        '无EdgeAttnMixHop=两层普通MixHop替代，不是移除所有图传播。无Microstate只移除最终balanced readout的微状态，递推Z和Recent专家仍保留。',
        '整支删除连带屏蔽依赖该分支的专家；每项具体定义见PREREGISTERED_PROTOCOL.json。',
        '仅冻结原审计判定的整组inactive参数，历史state key保留，所以nominal参数数与trainable参数数应分开理解；部分参数列也可能不影响当前消融。',
        'Full调优参数对所有消融固定，未分别调优各变体。消融改变容量、信息路径及随机执行，不能把差值都视为严格单因素因果贡献。',
        f'同一VAL的零收益参照MAE5={zero:.12f}，仅为背景，不是本轮新增选模候选。',
        '历史TEST已被多次查看，单Seed42及重复使用VAL的结果均为探索性；不宣称统计显著或独立盲测。',
        '不覆盖旧实验、默认配置、checkpoint或数据；不追加新变体、其他seed或按TEST重选epoch。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def run_suite(out,device='cuda:0',resume=False,allow_test=False):
    out=Path(out).resolve();root=project_root();p=make_protocol(allow_test)
    if not out.is_relative_to(root/'runs'):raise ValueError('Use a new standalone runs directory')
    if out.exists() and not resume:raise ValueError('Do not overwrite an existing run')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    if resume:
        if read(out/'PREREGISTERED_PROTOCOL.json')!=p:raise ValueError('Resume source/protocol mismatch')
    else:
        save(out/'PREREGISTERED_PROTOCOL.json',p)
        shutil.copytree(root/'src',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(root/'configs/project.json',out/'project_config.json');shutil.copy2(root/'tools/run_c4_ablation.py',out/'run_entry_snapshot.py')
        log=out.parent/(out.name+'_preflight.txt')
        if log.exists():shutil.copy2(log,out/'tests.txt')
    started=time.perf_counter();rows=[]
    try:
        with train_val_guard() as hits:
            init=initialization_audit();save(out/'INITIALIZATION_AUDIT.json',init)
            if not (out/'GPU_PREFLIGHT.json').exists():gpu_preflight(out,device,p)
            for key in KEYS:
                folder=out/'models'/key
                if not (folder/'COMPLETE.json').exists():
                    print('START',key,'lr3e-4 wd1e-4',flush=True)
                    with (out/'logs'/(key+'.log')).open('a') as log,redirect_stdout(Tee(sys.stdout,log)):
                        train('ablations/'+key,folder,device,resume=folder.exists(),lr=3e-4,weight_decay=1e-4)
                row=validate_run(out,key,p,init);rows.append(row);save(out/'PROGRESS.json',dict(completed=len(rows),total=13,rows=rows,TEST_opened=False))
                print('DONE',key,'VAL5',row['VAL5_macro_MAE'],flush=True)
            freeze_all(out,rows)
            if runtime_hashes()!=p['source_hashes']:raise ValueError('Sources changed before TEST')
            save(out/'TRAIN_VAL_ISOLATION.json',dict(status='PASS',blocked_attempts=hits,all13_frozen_before_TEST=True))
        test=evaluate_test(out,rows,device) if allow_test else {}
        summarize(out,rows,test,p)
        reference=torch.load(root/p['reference']/p['reference_best_checkpoint'],map_location='cpu',weights_only=False)
        full=torch.load(out/'models/full_c4/best.pt',map_location='cpu',weights_only=False)
        reproduction=dict(HPO_best_epoch=reference['epoch'],new_best_epoch=full['epoch'],trained_tensors_identical=tensor_hash(reference['model_state_dict'])==tensor_hash(full['model_state_dict']))
        save(out/'FULL_REPRODUCTION.json',reproduction)
        protected=read(root/'provenance/migration_manifest.json')['files']
        if any(sha(root/r['destination'])!=r['sha256'] for r in protected):raise ValueError('Archived data/result/checkpoint changed')
        if runtime_hashes()!=p['source_hashes'] or sha(root/'configs/project.json')!=p['project_config_sha256']:raise ValueError('Source/default config changed')
        save(out/'FINAL_AUDIT.json',dict(status='PASS',models=13,models_all_from_untrained_states=True,main_model='full_c4',
            models_frozen_sha256=sha(out/'MODELS_FROZEN.json'),TEST_models=list(test),TEST_after_VAL_freeze_only=True,
            source_unchanged=True,default_config_unchanged=True,protected_archive_files_unchanged=len(protected),seconds=time.perf_counter()-started,
            no_extra_models=True,no_extra_seeds=True,Full_reproduction=reproduction))
        save(out/'COMPLETE.json',dict(status='COMPLETE',models=13,TEST_evaluated=bool(test)))
        save(out/'file_index.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='file_index.json' and '__pycache__' not in f.parts})
        print('ALL13 COMPLETE',out/'REPORT.md',flush=True)
    except BaseException:
        save(out/'INTERRUPTION.json',dict(traceback=traceback.format_exc(),completed=len(rows),TEST_opened=(out/'TEST_OPEN_AUTHORIZATION.json').exists()))
        raise

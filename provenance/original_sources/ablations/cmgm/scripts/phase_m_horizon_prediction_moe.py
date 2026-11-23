"""Phase M: seven preregistered fresh seed42 fits; TRAIN/VAL ONLY.

Default runs synthetic preflight. --audit-data also verifies the exact historical
TRAIN/VAL cache and original D0B. --run is additionally data-gated. Known Legacy
P0 findings are never labelled causal PASS. No TEST dataset/API exists here.
"""
import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cmgm.models.phase_m_prediction_moe import (
    KEYS,MODES,HORIZONS,EXPERTS,PhaseMModel,matched_models,parameter_counts,
)
from cmgm.models.frozen_d0b_task_experts import tensor_state_hash
from cmgm.scripts.phase_m_analysis import (
    population,routing_rows,expert_competence,intervention_routes,moving_block_bootstrap,
)
from cmgm.scripts.d0b_short_long_dynamic_moe import seed_all,preserved_rng
from cmgm.scripts.d0b_task_conditioned_moe import data_audit
from cmgm.scripts.phase_a_audit import load_cache,raw_reconstruction
from cmgm.data.data_loader import MarketSequenceDataset
from cmgm.training.train import train_epoch

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT=ROOT/'experiments/d0b_source_task_moe/seed42_v1/config.json'
PHASE_A=ROOT/'experiments/phase_a_protocol_data_audit/20261014_205330_847827/audit_results.json'
PROTOCOL=dict(id='CMGM_PHASE_M_LEGACY24_SEED42_MACRO5_V1',models=MODES,seed=42,
    seq_len=20,feature_dim=21,targets=24,horizons=list(HORIZONS),target_unit='decimal cumulative return',
    initialization='Fresh independent original D0B state; explicitly copied common keys; NO trained weights used for fits',
    training_loss='SUM_h mean origin*commodity Huber(delta=.02) + native Switch KL ONCE',
    optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,max_epochs=200,patience=10,
    scheduler_factor=.5,scheduler_patience=5,train_shuffle=False,train_drop_last=True,
    full_evaluation_drop_last=False,checkpoint_selection='VAL_5D_COMMODITY_MACRO_MAE',
    selection_definition='Mean of each commodity valid-origin MAE, then equal mean across24 commodities',
    epoch0_candidate=False,native_KL='beta_max=.0005 * clamp((epoch-1)/19,0,1); original set_epoch/switch_loss interfaces',
    router_temperature=1.,router_hidden=32,horizon_embedding_dim=8,interaction_rank=16,
    router_last_init='weight Normal(mean0,std.001); bias0; embedding Normal(mean0,std.02)',
    auxiliary_loss=False,router_warmup=False,load_balance=False,entropy_loss=False,
    head='One original Linear64->64/ReLU/Dropout.3/Linear64->96; horizon-major rows',
    C5_C6_difference='Fusion location only; head body once B,4,64 in each. Dropout masks index experts in C5, horizons in C6; same shape/count, different operands. Eval disables dropout.',
    bootstrap=dict(default_run=False,block_origins=40,repeats=2000,seed=42,role='exploratory; date-paired all commodities'),
    test_policy='Never load TEST values, build TEST labels or evaluate TEST; no automatic TEST trigger',
    data_gate='Identity/fingerprint/model gates mandatory. Known Legacy P0 requires explicit authorized exploratory override, never causal PASS.')
SOURCE_FILES=(
    'cmgm/config.py','cmgm/models/hetero_mixhop_model.py','cmgm/models/switching_latent_transformer.py',
    'cmgm/models/four_task_moe.py','cmgm/models/horizon_tensor_fusion.py','cmgm/models/phase_m_prediction_moe.py',
    'cmgm/models/frozen_d0b_task_experts.py','cmgm/models/model.py','cmgm/graph/adaptive_graph.py',
    'cmgm/data/data_loader.py','cmgm/data/feature_builder.py','cmgm/training/train.py',
    'cmgm/scripts/main_ablation.py','cmgm/scripts/d0b_task_conditioned_moe.py',
    'cmgm/scripts/d0b_short_long_dynamic_moe.py','cmgm/scripts/d0b_short_long_expert_diagnostics.py',
    'cmgm/scripts/phase_a_audit.py','cmgm/scripts/phase_m_analysis.py',
    'cmgm/scripts/phase_m_horizon_prediction_moe.py','tests/test_phase_m_prediction_moe.py',
    'tests/test_four_task_moe.py','tests/test_horizon_tensor_fusion.py')


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def checkpoint(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(v,tmp);tmp.replace(p)


def csv_file(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    if not fields:raise ValueError('CSV requires explicit fields for an unrun experiment')
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def source_hashes():return {p:sha(ROOT/p) for p in SOURCE_FILES}


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])


def original_context(path=DEFAULT_CONTEXT):
    """Source-task config is used ONLY as an audited D0B/cache locator."""
    locator=read(path);index_path=Path(locator['reference']['file_index']);item=read(index_path)['d0b']
    reference=read(item['config'])
    if reference['variant']!='switching_latent_balanced_readout' or reference['seed']!=42:
        raise ValueError('Not independent original D0B seed42')
    if reference['data_audit']!=locator['data_audit']:raise ValueError('Locator/data identity conflict')
    if sha(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('D0B checkpoint hash mismatch')
    cache=Path(locator['data_cache'])
    if sha(cache)!=locator['data_cache_sha256']:raise ValueError('Audited TRAIN/VAL cache SHA changed')
    return locator,index_path,item,reference,cache


def data_from_cache(cache,expected):
    """Whitelist enforced before array access; never call build_data/load TEST."""
    arrays,meta=load_cache(cache,expected)
    market={k:tuple(v) for k,v in expected['market_indices'].items()}
    data=dict(loaders={},origin_dates={},market_indices=market,n_nodes=len(expected['node_order']),
        n_commodities=len(expected['commodity_order']),feature_names=expected['node_order'])
    for split in ('train','val'):
        a=arrays[split]
        ds=MarketSequenceDataset(a['prices'],market,seq_len=20,feature_matrix=a['feature_matrix'],
            raw_prices=a['raw_prices'],target_type='return',horizons=list(HORIZONS))
        data['loaders'][split]=DataLoader(ds,batch_size=64,shuffle=False,drop_last=split=='train',num_workers=0)
        data['origin_dates'][split]=a['origin_dates']
    if data_audit(data)!=expected:raise ValueError('TRAIN/VAL numeric fingerprint/order mismatch')
    return data,arrays,meta


def split_loader(data,split,training=False):
    if split not in ('train','val'):raise ValueError('Only TRAIN/VAL are authorized')
    return DataLoader(data['loaders'][split].dataset,batch_size=64,shuffle=False,
        drop_last=training and split=='train',num_workers=0)


@torch.inference_mode()
def evaluate(model,loader,device,dates,capture=False):
    model.eval();ps=[];ys=[];routes=[];candidates=[];norms=[];batch_huber=[]
    for batch in loader:
        x,y=batch[:2];x=x.to(device)
        if capture:p,a=model(x,return_aux=True,with_candidates=model.control=='C6')
        else:p=model(x);a={}
        if p.shape!=y.shape or p.shape[1:]!=(4,24) or not torch.isfinite(p).all():raise ValueError('Shape/finite prediction failure')
        ps.append(p.cpu().numpy());ys.append(y.numpy())
        batch_huber.append(float(sum(nn.functional.huber_loss(p[:,h],y[:,h].to(device),delta=.02) for h in range(4))))
        if a:
            routes.append(a['routing'].cpu().numpy());norms.append(a['experts'].norm(dim=-1).cpu().numpy())
            candidates.append(a['expert_prediction'].cpu().numpy())
    p,y=np.concatenate(ps),np.concatenate(ys)
    if len(p)!=len(loader.dataset) or len(dates)!=len(p):raise ValueError('Full origin coverage failed')
    arrays=dict(prediction=p,target=y,origin_dates=np.asarray(dates).astype('datetime64[ns]').astype(str))
    if routes:arrays.update(routing=np.concatenate(routes),expert_prediction=np.concatenate(candidates),expert_norm=np.concatenate(norms))
    result=population(p,y);result['legacy_four_horizon_VAL_Huber_batch_mean_logging_only']=float(np.mean(batch_huber))
    result['origins']=len(p)
    return (result,arrays) if capture else result


def synthetic_preflight(kwargs):
    models,audit=matched_models(kwargs)
    x=torch.randn(2,20,kwargs['num_nodes'],21)
    result={}
    for control,m in models.items():
        m.eval();p,a=m(x,return_aux=True)
        loss=sum(nn.functional.huber_loss(p[:,h],torch.randn_like(p[:,h])*.02,delta=.02) for h in range(4))
        loss.backward()
        finite=all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        if not finite or p.shape!=(2,4,24) or not torch.isfinite(p).all():raise ValueError('Synthetic model/gradient failure '+control)
        active=sum(v.numel() for v in m.parameters() if v.grad is not None)
        result[control]=dict(**parameter_counts(m),effective_gradient_parameter_count=active,
            unused_parameter_names=[n for n,v in m.named_parameters() if v.grad is None],shape=list(p.shape),finite=True,PASS=True)
    return models,audit,result


def write_pre_registration(out,cfg):
    (out/'PREREGISTERED_PROTOCOL.md').write_text('# Phase M 预注册协议\n\n'+
        '固定 C0–C6、seed42、每组从头训练一次，不加载历史训练权重。原始CMGM 21特征/24商品，不使用TriWeave。\n\n'+
        '```json\n'+json.dumps(PROTOCOL,ensure_ascii=False,indent=2)+'\n```\n\n'+
        '唯一checkpoint/scheduler/早停目标为VAL5日商品Macro MAE；四周期仍等权Huber训练。新选择目标与历史Huber选模不同，历史best不进入正式排名。\n\n'+
        '训练预算、专家定义、rank16、Router32/embedding8和固定顺序不可根据VAL中途改动。每个epoch保留原TRAIN drop_last=True，因此1396中固定52起点不参与该epoch梯度；完整指标仍评价全部TRAIN/VAL。\n\n'+
        '数据门禁：Phase A已验证Legacy未来bfill/标签污染。身份与指纹复现不等于因果有效；默认阻止训练。仅在用户明确接受已知缺陷的探索性范围后，通过显式确认记录才能继续；届时也不能将DATA_CAUSALITY标记PASS。\n\n'+
        'Bootstrap若显式启用则配对40个origin移动块、2000次、seed42，同日商品整体重采样；只作探索性描述。未启用显示NOT_RUN。\n')
    (out/'EXPERT_ARCHITECTURE.md').write_text('# Phase M 四专家\n\n'+
        '| 顺序 | 专家输入与公式 | 与旧实现关系 |\n|---|---|---|\n'+
        '| E1 | ReLU(W[Z_last,mean(Z_last5),Z_last−mean(Z_last5)]),192→64 | 复用旧RecentChangeExpert结构并显式加入mean；旧128→64，新增4096参数；全部C1–C6一致 |\n'+
        '| E2 | ReLU(W[H_last,mean(H20)]),256→64 | 直接复用LongMemoryExpert |\n'+
        '| E3 | s+ReLU(Ws),64→64 | 直接复用SpatialStructureExpert |\n'+
        '| E4 | Wo[(Us s)⊙(Ut t)],rank16 | 复用LowRankInteractionExpert三个矩阵；按本轮公式不加旧t残差 |\n\n'+
        'Z具有递推历史，末5步不等于只读最后5天原始信息。H是20-session观察窗口。专家信息可以重叠，名称不是已证明的经济分工。未加新expert LayerNorm或Dropout。\n\n'+
        '一次原空间和时间forward产生live H/Z及原balanced readout，保留gcn_proj/lstm_proj。C1–C6删除无效原gate；只有C0使用原gate。预测共享原head参数，周期输出行保持horizon-major。\n')
    (out/'ROUTER_ARCHITECTURE.md').write_text('# Phase M 路由与融合\n\n'+
        '动态context=[LN_s(s),LN_t(t),LN_change(Z_last−mean5Z),LN_long(Linear128→64(mean20H))]，256维，仅历史输入。C4:256→32→GELU→4。C5/C6拼接8维horizon embedding后264→32→GELU→4。温度1，最后Linear weight Normal(0,.001),bias0；embedding Normal(0,.02)。\n\n'+
        'C1一组4 logits；C2两组4 logits索引[0,0,1,1]；C3四组4 logits；C4每样本一组重复四期限；C5/C6每样本每期限一组。所有期限均可访问全部4专家，无固定专家掩码、辅助均衡、熵或warmup。\n\n'+
        'C1–C5先head(Ek)产生[B,H,K,C]，再按[B,H,K]权重相加；C6先混合[B,H,64]再使用同一head与对应周期输出行。C5/C6整个初始state相同，未复制预测头。由于K=H=4，训练时共享body都仅调用一次、输入shape同为[B,4,64]，Dropout随机数数量相同，但mask分别作用于专家与期限表示；不声称训练轨迹等价。\n')


def prepare(args):
    out=args.output or ROOT/'experiments/phase_m_horizon_prediction_moe'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    locator,index_path,item,reference,cache=original_context(args.context)
    kwargs=reference['model_kwargs']
    if (kwargs['num_nodes'],kwargs['n_commodities'],reference['protocol']['feature_dim'])!=(284,24,21):
        raise ValueError('Original D0B shape identity differs; cannot substitute TriWeave')
    hashes=source_hashes()
    cfg=dict(protocol=PROTOCOL,model_kwargs=kwargs,context=str(args.context.resolve()),
        reference_index=str(index_path),reference_item=item,reference_config=reference,
        data_cache=str(cache),data_cache_sha256=sha(cache),phase_a_audit=str(args.phase_a.resolve()),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        git_diff_stat=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True),
        source_hashes=hashes,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        checkpoint_root=str(ROOT/'checkpoints/phase_m_horizon_prediction_moe'/out.name),test_evaluated=False)
    save(out/'TRAINING_CONFIG.json',cfg)
    for name in SOURCE_FILES:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_phase_m_prediction_moe.py',
        'tests/test_four_task_moe.py','tests/test_horizon_tensor_fusion.py'],cwd=ROOT,env=env,capture_output=True,text=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise ValueError('Mechanism/regression tests failed; STOP before data or training')
    models,initialization,checks=synthetic_preflight(kwargs)
    checkpoint(out/'initial_states.pt',{k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in models.items()})
    save(out/'COMMON_INITIALIZATION_AUDIT.json',initialization)
    save(out/'MODEL_REGISTRY.json',{key:dict(model_id=MODES[key],**checks[key],status='NOT_RUN') for key in KEYS})
    csv_file(out/'MODEL_PARAMETER_COUNTS.csv',[dict(Control=k,Model=MODES[k],**{n:v for n,v in checks[k].items() if n in ('total','trainable','backbone_and_head','router','effective_gradient_parameter_count')},
        experts_total=sum(checks[k]['experts'].values())) for k in KEYS])
    write_pre_registration(out,cfg)
    save(out/'AUDIT_RESULTS.json',dict(status='SYNTHETIC_PASS_DATA_NOT_VERIFIED',mechanism_preflight=checks,
        initialization_PASS=initialization['PASS'],data_fingerprint_status='NOT_VERIFIED',DATA_CAUSALITY='NOT_VERIFIED',
        formal_training='NOT_RUN',TEST_values_read=False,test_evaluated=False))
    save(out/'CHECKPOINT_AUDIT.json',dict(status='NOT_RUN',reason='Only fresh initial states exist; formal best/latest checkpoints not created'))
    save(out/'PROVENANCE.json',dict(git_commit=cfg['git_commit'],git_status=cfg['git_status'],source_sha256=hashes,
        historical_D0B_checkpoint=item['checkpoint'],historical_D0B_sha256=item['checkpoint_sha256'],
        historical_checkpoint_used_for_initialization=False,reference_role='Identity and frozen TRAIN/VAL reproduction only',
        source_snapshot=str(out/'source_snapshot'),test_evaluated=False))
    empty={'TRAINING_HISTORY.csv':['Control','epoch','train_total_loss_batch_mean','VAL_5D_COMMODITY_MACRO_MAE'],
        'VAL_5D_RANKING.csv':['Rank','Control','Model','VAL_1d','VAL_5d','VAL_10d','VAL_20d','Delta5d_vs_D0B','Params','Best_Epoch'],
        'VAL_ALL_HORIZONS.csv':['Control','Horizon','commodity_macro_MAE','pooled_MAE','RMSE','Hit'],
        'EXPERT_ROUTING_BY_HORIZON.csv':['Model','Horizon',*EXPERTS,'mean_sample_entropy_nats'],
        'EXPERT_FORECAST_COMPETENCE.csv':['Model','Expert','horizon','commodity','MAE'],
        'EXPERT_PREDICTION_CORRELATION.csv':['Model','Horizon','Expert_A','Expert_B','prediction_correlation','error_correlation'],
        'ROUTER_INTERVENTION_RESULTS.csv':['Intervention','Horizon','commodity_macro_MAE','DeltaMAE','mean_absolute_prediction_change'],
        'HORIZON_ROUTING_DIFFERENCE.csv':['Model','Horizon_A','Horizon_B','JS_mean_routes_nats'],
        'PER_COMMODITY_5D_RESULTS.csv':['Control','commodity','MAE']}
    for name,fields in empty.items():csv_file(out/name,[],fields)
    save(out/'STATUS.json',dict(status='PREFLIGHT_PASSED',models_complete=0,total=7,test_evaluated=False))
    (out/'REPORT.md').write_text('# Phase M 状态\n\n模型与初始化合成预检通过。7组正式实验均NOT_RUN；空CSV仅含列名，不代表零误差。真实数据身份、指纹与已知P0门禁尚待审计。未读取TEST。\n')
    print('PREPARED',out,flush=True)
    return out


def audit_data(out):
    cfg=read(out/'TRAINING_CONFIG.json')
    if cfg['source_hashes']!=source_hashes():raise ValueError('Source drift after preregistration')
    ref=cfg['reference_config'];item=cfg['reference_item'];cache=Path(cfg['data_cache'])
    if sha(cache)!=cfg['data_cache_sha256']:raise ValueError('Frozen cache changed')
    from cmgm.scripts.d0b_short_long_expert_diagnostics import load_run
    model,old_cfg,old_results,identity=load_run('d0b',item,torch.device('cpu'))
    if sum(v.numel() for v in model.parameters())!=520549:raise ValueError('Original full parameter count differs')
    data,arrays,meta=data_from_cache(cache,ref['data_audit'])
    locator=read(cfg['context'])
    combined,dates,lineage,access=raw_reconstruction(locator,ref['data_audit'],arrays)
    if any(r['TEST_numeric_rows_parsed'] for r in access):raise ValueError('Unauthorized TEST parsing')
    fingerprints={}
    for split in ('train','val'):
        ds=data['loaders'][split].dataset
        y=np.stack([ds[i][1].numpy() for i in range(len(ds))])
        origin=np.asarray(data['origin_dates'][split]).astype('datetime64[ns]')
        expected=np.asarray(ref['origin_date_ranges'][split]['first'],dtype='datetime64[ns]')
        if origin[0]!=expected or str(origin[-1])!=ref['origin_date_ranges'][split]['last']:
            raise ValueError('Frozen origin dates changed')
        if not np.array_equal(origin,np.asarray(dates[split][19:19+len(ds)],dtype='datetime64[ns]')):
            raise ValueError('Calendar reconstruction differs from origin metadata')
        predicted=[];ys=[]
        with torch.inference_mode():
            for x,target in split_loader(data,split):predicted.append(model(x).numpy());ys.append(target.numpy())
        p,yfresh=np.concatenate(predicted),np.concatenate(ys)
        saved_path=Path(item['TRAIN_predictions' if split=='train' else 'VAL_predictions'])
        with np.load(saved_path,allow_pickle=False) as z:
            pkey='prediction' if 'prediction' in z.files else 'predictions'
            ykey='target' if 'target' in z.files else 'targets'
            np.testing.assert_allclose(p,z[pkey],atol=1e-5,rtol=1.3e-6)
            if not np.array_equal(yfresh,z[ykey]):raise ValueError('Frozen targets differ')
            pred_diff=float(np.abs(p-z[pkey]).max())
        r=population(p,yfresh)
        fingerprints[split]=dict(**ref['data_audit']['splits'][split],
            origin_sha256=hashlib.sha256(origin.astype('int64').tobytes()).hexdigest(),
            label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),first_origin=str(origin[0]),last_origin=str(origin[-1]),
            frozen_checkpoint_prediction_max_abs_diff=pred_diff,restored_metrics=r)
    # Independently execute the exact original training model source. The sole
    # documented current change is diagnostics-only get_gate_stats dispatch.
    source=(ROOT/'cmgm/models/hetero_mixhop_model.py').read_text()
    expected_sha=ref['source_sha256']['cmgm/models/hetero_mixhop_model.py']
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha:
        start=source.index('        if self.variant in DYNAMIC_STUDY_VARIANTS:',source.index('    def get_gate_stats('))
        stop=source.index('        if self.variant == "switching_latent_balanced_mean_fusion":',start)
        source=source[:start]+source[stop:]
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha:raise ValueError('Cannot reproduce original D0B training source')
    path=out/'source_reference/original_hetero_mixhop_model.py';path.parent.mkdir(exist_ok=True);path.write_text(source)
    module=types.ModuleType('cmgm_phase_m_original');module.__file__=str(path);module.__package__='cmgm.models'
    exec(compile(source,str(path),'exec'),module.__dict__)
    historical=module.HeteroMixHopCMGM(variant='switching_latent_balanced_readout',**cfg['model_kwargs']).eval()
    historical.load_state_dict(model.state_dict(),strict=True)
    x=next(iter(split_loader(data,'train')))[0][:2]
    with torch.inference_mode():
        max_diff=float((historical(x)-model(x)).abs().max())
    if max_diff>1e-7:raise ValueError('Original D0B forward changed')
    phase_a=read(cfg['phase_a_audit']);actual=phase_a['actual']
    if actual['data_audit']!=ref['data_audit']:raise ValueError('Phase A evidence refers to a different dataset')
    pollution=sum(v['label_bfill_source_after_target_date'] for v in actual['label_price_availability'])
    evidence=dict(report=cfg['phase_a_audit'],report_sha256=sha(cfg['phase_a_audit']),
        input_future_fill_events=actual['future_fill_input_events'],label_future_bfill_endpoints=pollution,
        DATA_CAUSALITY='FAIL_KNOWN_LEGACY_P0',
        consequence='Fingerprint reproduction PASS is not a no-leakage claim; formal training STOP unless explicit exploratory authorization')
    audit=dict(status='IDENTITY_AND_FINGERPRINT_PASS_CAUSALITY_FAIL',original_D0B_identity=identity,
        model_kwargs=cfg['model_kwargs'],original_full_parameters=520549,original_source_forward_max_abs_diff=max_diff,
        original_training_model_sha256=expected_sha,data_audit=ref['data_audit'],splits=fingerprints,
        cache_sha256=sha(cache),calendar=meta['calendar_audit'],bounded_raw_price_reproduction=True,
        numeric_access_log=access,known_data_defects=evidence,TEST_values_read=False,test_evaluated=False)
    save(out/'DATA_FINGERPRINT_AUDIT.json',audit)
    save(out/'AUDIT_RESULTS.json',dict(status='DATA_GATE_BLOCKED',mechanism_preflight='PASS',
        common_initialization='PASS',original_D0B_identity='PASS',data_fingerprint='PASS',
        DATA_CAUSALITY=evidence['DATA_CAUSALITY'],known_data_defects=evidence,
        formal_training='NOT_RUN',test_evaluated=False))
    (out/'CMGM_D0B_SOURCE_AUDIT.md').write_text('# 原始 CMGM D0B 身份审计\n\n'+
        f'独立D0B seed42，历史best epoch{identity["best_epoch"]}，520549参数。checkpoint SHA256 `{identity["checkpoint_sha256"]}`。严格加载、原源码重建与forward一致性通过，最大差{max_diff}。该checkpoint仅审计，绝不用于本轮初始化。\n\n'+
        f'实际模型构造参数：`{json.dumps(cfg["model_kwargs"],ensure_ascii=False)}`。输入284节点/21特征，输出24商品/四周期。数据资产映射来自冻结配置，不从TriWeave继承。原始gate/投影/head的计算顺序保持不变。\n\n'+
        'TRAIN/VAL缓存、标签、origins、日历和原检查点正常预测均复现。未读取TEST。\n\n'+
        f'已知Legacy问题：未来输入填充事件{evidence["input_future_fill_events"]}，标签未来bfill端点{pollution}。因此DATA_CAUSALITY仍FAIL。只有明确授权后才能将复验限定为有已知缺陷的Legacy探索性研究；不能声称已修复泄漏。\n')
    save(out/'STATUS.json',dict(status='BLOCKED_LEGACY_DATA_CAUSALITY',models_complete=0,total=7,test_evaluated=False))
    (out/'REPORT.md').write_text('# Phase M 当前结论\n\n原始D0B身份、TRAIN/VAL指纹和机制预检通过，但Legacy数据存在已确认的未来bfill和污染标签，因果数据验收FAIL。7组正式训练均NOT_RUN；预测、路由分工和互补性的22项结果问题尚无新实验结果。未访问TEST。\n\n'+
        '请查看CMGM_D0B_SOURCE_AUDIT.md、DATA_FINGERPRINT_AUDIT.json和PREREGISTERED_PROTOCOL.md。等待明确的数据缺陷处理决定，不用旧checkpoint或Oracle结果填补新实验排名。\n')
    return data


def authorized_data_gate(out,accept_legacy=False):
    audit=read(out/'DATA_FINGERPRINT_AUDIT.json')
    if audit['status']!='IDENTITY_AND_FINGERPRINT_PASS_CAUSALITY_FAIL':raise ValueError('Mandatory identity/data gate failed')
    p=out/'TRAINING_AUTHORIZATION.json'
    if not p.exists():
        if not accept_legacy:raise ValueError('STOP: Legacy P0 data defects need an explicit exploratory-scope decision before training')
        save(p,dict(accepted_known_legacy_data_defects=True,DATA_CAUSALITY='FAIL_KNOWN_LEGACY_P0',
            scope='Seven seed42 Legacy exploratory fits only; no clean-data/no-leakage claim',
            recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),test_authorized=False))
    if not read(p)['accepted_known_legacy_data_defects']:raise ValueError('No training authorization')
    results=read(out/'AUDIT_RESULTS.json')
    results.update(status='AUTHORIZED_LEGACY_EXPLORATORY_CAUSALITY_STILL_FAIL',
        authorized_exploratory_exception=True,DATA_CAUSALITY='FAIL_KNOWN_LEGACY_P0')
    save(out/'AUDIT_RESULTS.json',results)


def train_one(control,out,cfg,data,device):
    folder=out/'runs'/control;folder.mkdir(parents=True,exist_ok=True)
    cpdir=Path(cfg['checkpoint_root'])/control;cpdir.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        d=read(folder/'COMPLETE.json')
        if sha(d['checkpoint'])!=d['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return d
    model=PhaseMModel(control,cfg['model_kwargs']).to(device)
    initial=torch.load(out/'initial_states.pt',map_location='cpu',weights_only=True)[control]
    model.load_state_dict(initial,strict=True)
    # Audit/model construction RNG never becomes training RNG.
    seed_all(42)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
    train_loader=split_loader(data,'train',True);val_loader=split_loader(data,'val')
    history=[];best=float('inf');best_epoch=None;stale=0;start=1;seconds=0.
    best_path,latest=cpdir/'best.pt',cpdir/'latest.pt'
    if latest.exists():
        cp=torch.load(latest,map_location='cpu',weights_only=False)
        if cp['config_sha256']!=sha(out/'TRAINING_CONFIG.json') or cp['control']!=control:raise ValueError('Resume identity mismatch')
        model.load_state_dict(cp['model_state_dict'],strict=True);optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']);model.set_epoch(cp['epoch']);restore_rng(cp['rng'])
        history=cp['history'];best=cp['best_val5_macro_MAE'];best_epoch=cp['best_epoch'];stale=cp['stale'];seconds=cp['seconds'];start=cp['epoch']+1
    def finite_gradient(gradient):
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite gradient; no automatic retry or tuning')
        return gradient
    guards=[p.register_hook(finite_gradient) for p in model.parameters()]
    for epoch in range(start,201):
        if stale>=10:break
        tic=time.perf_counter();beta=model.set_epoch(epoch);kl_values=[]
        original=model.switching_latent_transformer.switch_loss
        def traced_kl():
            value=original();kl_values.append(float(value.detach()));return value
        with patch.object(model.switching_latent_transformer,'switch_loss',side_effect=traced_kl):
            total=train_epoch(model,train_loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),optimizer,nn.HuberLoss(delta=.02),device)
        if not np.isfinite(total) or len(kl_values)!=len(train_loader):raise ValueError('KL must be finite and added exactly once per optimizer step')
        val=evaluate(model,val_loader,device,data['origin_dates']['val']);score=val['primary'];scheduler.step(score)
        improved=score<best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        elapsed=time.perf_counter()-tic;seconds+=elapsed
        row=dict(Control=control,epoch=epoch,train_total_loss_batch_mean=total,
            train_switch_KL_batch_mean=float(np.mean(kl_values)),train_prediction_loss_batch_mean=total-float(np.mean(kl_values)),
            VAL_5D_COMMODITY_MACRO_MAE=score,VAL_MAE={h:v['commodity_macro_MAE'] for h,v in val['horizons'].items()},
            VAL_Huber_sample_equal=val['four_horizon_prediction_Huber_sum'],VAL_Huber_legacy_batch_mean=val['legacy_four_horizon_VAL_Huber_batch_mean_logging_only'],
            lr=optimizer.param_groups[0]['lr'],switch_beta=beta,optimizer_steps=len(train_loader),
            train_origins_used=sum(len(batch) for batch in train_loader.batch_sampler),train_dataset_origins=len(train_loader.dataset),seconds=elapsed)
        history.append(row);save(folder/'history.json',history)
        state=dict(control=control,model_id=MODES[control],model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=rng_state(),epoch=epoch,best_epoch=best_epoch,best_val5_macro_MAE=best,
            history=history,stale=stale,seconds=seconds,config_sha256=sha(out/'TRAINING_CONFIG.json'),
            data_audit_sha256=sha(out/'DATA_FINGERPRINT_AUDIT.json'),selection='VAL_5D_COMMODITY_MACRO_MAE',switch_beta=beta)
        if improved:checkpoint(best_path,state)
        checkpoint(latest,state)
        print(f'[{control}] epoch={epoch} VAL5_macro_MAE={score:.10g} best={best_epoch} KL_beta={beta:.6g} seconds={elapsed:.1f}',flush=True)
    for handle in guards:handle.remove()
    cp=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(cp['model_state_dict'],strict=True);model.set_epoch(cp['epoch'])
    results={}
    for split in ('train','val'):
        metrics,arrays=evaluate(model,split_loader(data,split),device,data['origin_dates'][split],True)
        np.savez_compressed(folder/(split+'_predictions.npz'),**arrays,commodity_order=np.array(cfg['reference_config']['data_audit']['commodity_order']))
        results[split]=metrics
    if not np.isclose(results['val']['primary'],best,rtol=1e-7,atol=1e-9):raise ValueError('Best checkpoint VAL5 reproduction failed')
    save(folder/'metrics.json',results)
    d=dict(control=control,model_id=MODES[control],status='COMPLETE',best_epoch=cp['epoch'],
        best_VAL5_macro_MAE=best,parameters=parameter_counts(model),training_seconds=seconds,
        checkpoint=str(best_path),checkpoint_sha256=sha(best_path),latest_checkpoint=str(latest),
        strict_restore=True,test_evaluated=False)
    save(folder/'COMPLETE.json',d)
    del model,optimizer,scheduler,cp,initial
    if device.type=='cuda':torch.cuda.empty_cache()
    return d


def summarize(out,bootstrap=False):
    cfg=read(out/'TRAINING_CONFIG.json');done={};metrics={};arrays={};histories=[]
    if cfg['source_hashes']!=source_hashes() or cfg['protocol']!=PROTOCOL:raise ValueError('Summary source/protocol drift')
    for control in KEYS:
        folder=out/'runs'/control
        if not (folder/'COMPLETE.json').exists():raise ValueError('Seven complete runs required; missing '+control)
        d=read(folder/'COMPLETE.json')
        if sha(d['checkpoint'])!=d['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
        cp=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)
        restored=PhaseMModel(control,cfg['model_kwargs']);restored.load_state_dict(cp['model_state_dict'],strict=True)
        history=read(folder/'history.json');selected=min(history,key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
        if selected['epoch']!=d['best_epoch'] or cp['selection']!='VAL_5D_COMMODITY_MACRO_MAE':raise ValueError('Checkpoint selection audit failed')
        done[control]=d;metrics[control]=read(folder/'metrics.json')['val'];histories+=history
        with np.load(folder/'val_predictions.npz',allow_pickle=False) as z:arrays[control]={k:z[k] for k in z.files}
        r=population(arrays[control]['prediction'],arrays[control]['target'])
        if r['primary']!=metrics[control]['primary']:raise ValueError('Saved VAL prediction metrics differ')
        if control!='C0':
            for key in ('target','origin_dates','commodity_order'):
                if not np.array_equal(arrays[control][key],arrays['C0'][key]):raise ValueError('VAL population differs')
    ranking=[];all_h=[];per=[];routes=[];js=[];competence=[];corr=[]
    for rank,control in enumerate(sorted(KEYS,key=lambda k:metrics[k]['primary']),1):
        d=done[control];r=metrics[control]
        ranking.append(dict(Rank=rank,Control=control,Model=MODES[control],
            **{f'VAL_{h}d':r['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS},
            Delta5d_vs_D0B=r['primary']-metrics['C0']['primary'],Params=d['parameters']['total'],Best_Epoch=d['best_epoch'],
            training_seconds=d['training_seconds'],VAL5_RMSE=r['horizons']['5']['RMSE'],VAL5_Hit_percent=100*r['horizons']['5']['Hit']))
    commodities=cfg['reference_config']['data_audit']['commodity_order']
    for control in KEYS:
        for h,v in metrics[control]['horizons'].items():all_h.append(dict(Control=control,Horizon=int(h),**v))
        for v in metrics[control]['per_commodity']:
            if v['horizon']==5:per.append(dict(Control=control,commodity=commodities[v['commodity_index']],**v))
        if control!='C0':
            rr,jj=routing_rows(control,arrays[control]['routing']);routes+=rr;js+=jj
            cc,pp=expert_competence(control,arrays[control]['expert_prediction'],arrays[control]['target'],commodities)
            competence+=cc;corr+=pp
    c5=arrays['C5'];train=np.load(out/'runs/C5/train_predictions.npz',allow_pickle=False)
    operations=intervention_routes(train['routing'],c5['routing']);interventions=[]
    for name,weights in operations.items():
        # Exact frozen C5 prediction-mixture operator. Candidate forecasts already
        # contain the selected checkpoint's nonlinear shared head, not raw states.
        p=torch.einsum('bhk,bhkc->bhc',torch.from_numpy(np.asarray(weights,dtype=np.float32)),torch.from_numpy(c5['expert_prediction'])).numpy()
        r=population(p,c5['target'])
        for hi,h in enumerate(HORIZONS):
            interventions.append(dict(Intervention=name,Horizon=h,commodity_macro_MAE=r['horizons'][str(h)]['commodity_macro_MAE'],
                DeltaMAE=r['horizons'][str(h)]['commodity_macro_MAE']-metrics['C5']['horizons'][str(h)]['commodity_macro_MAE'],
                mean_absolute_prediction_change=float(np.mean(np.abs(p[:,hi].astype('float64')-c5['prediction'][:,hi]))),
                mean_route_L1=float(np.abs(weights[:,hi]-c5['routing'][:,hi]).sum(-1).mean()),
                interpretation='Frozen checkpoint intervention, not a retrained ablation or financial causal effect'))
    comparisons=[]
    for reference in ('C0','C3','C4','C6'):
        diffs={str(h):metrics['C5']['horizons'][str(h)]['commodity_macro_MAE']-metrics[reference]['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS}
        comparisons.append(dict(Comparison='C5 vs '+reference,VAL_5d_Difference=diffs['5'],
            Other_Horizon_Effects=json.dumps({h:v for h,v in diffs.items() if h!='5'}),
            Interpretation='C5 lower on VAL5' if diffs['5']<0 else 'C5 not lower on VAL5'))
    for filename,rows in [('TRAINING_HISTORY.csv',histories),('VAL_5D_RANKING.csv',ranking),('VAL_ALL_HORIZONS.csv',all_h),
        ('EXPERT_ROUTING_BY_HORIZON.csv',routes),('HORIZON_ROUTING_DIFFERENCE.csv',js),
        ('EXPERT_FORECAST_COMPETENCE.csv',competence),('EXPERT_PREDICTION_CORRELATION.csv',corr),
        ('ROUTER_INTERVENTION_RESULTS.csv',interventions),('PER_COMMODITY_5D_RESULTS.csv',per),('KEY_MECHANISM_COMPARISONS.csv',comparisons)]:csv_file(out/filename,rows)
    if bootstrap:save(out/'BLOCK_BOOTSTRAP.json',{r:moving_block_bootstrap(c5['prediction'],arrays[r]['prediction'],c5['target']) for r in ('C0','C3','C4','C6')})
    else:save(out/'BLOCK_BOOTSTRAP.json',dict(status='NOT_RUN',preregistered=PROTOCOL['bootstrap']))
    save(out/'CHECKPOINT_AUDIT.json',dict(status='PASS',models=done,selection='VAL_5D_COMMODITY_MACRO_MAE',
        strict_load=True,seven_complete=True,test_evaluated=False))
    save(out/'FINAL_RESULTS.json',dict(ranking=ranking,comparisons=comparisons,metrics=metrics,C5_routing=[r for r in routes if r['Model']=='C5'],
        C5_JS=[r for r in js if r['Model']=='C5'],interventions=interventions,test_evaluated=False))
    registry=read(out/'MODEL_REGISTRY.json')
    for key,d in done.items():registry[key].update(status='COMPLETE',best_epoch=d['best_epoch'],checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],VAL5_macro_MAE=d['best_VAL5_macro_MAE'])
    save(out/'MODEL_REGISTRY.json',registry)
    audit=read(out/'AUDIT_RESULTS.json');audit.update(status='MODEL_AND_REPRODUCTION_PASS_LEGACY_DATA_CAUSALITY_FAIL',
        formal_training='COMPLETE',models_complete=7,DATA_CAUSALITY='FAIL_KNOWN_LEGACY_P0',test_evaluated=False)
    save(out/'AUDIT_RESULTS.json',audit)
    report(out,ranking,comparisons,metrics,routes,js,interventions)
    save(out/'STATUS.json',dict(status='COMPLETE_LEGACY_EXPLORATORY',models_complete=7,total=7,test_evaluated=False,DATA_CAUSALITY='FAIL_KNOWN_LEGACY_P0'))


def md_table(rows,fields):
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+
        ['| '+' | '.join(f'{r.get(f):.10g}' if isinstance(r.get(f),float) else str(r.get(f,'')) for f in fields)+' |' for r in rows])


def report(out,ranking,comparisons,metrics,routes,js,interventions):
    profile=[r for r in routes if r['Model']=='C5'];weights=np.array([[r[n] for n in EXPERTS] for r in profile])
    sd=np.array([[r[n+'_std'] for n in EXPERTS] for r in profile])
    byname={r['Comparison']:r for r in comparisons}
    lower=lambda k:metrics['C5']['primary']<metrics[k]['primary']
    answers=[
        '是。独立原始D0B身份strict恢复和源码forward复核通过；训练从fresh state开始，不加载该历史权重。',
        '是。原空间、历史时间加权、图、Transformer、Markov、latent、Balanced Readout与投影保留。',
        '复用了旧四专家类：E1显式加入均值、E4按本轮公式移除旧t残差，E2/E3直接复用；不是旧模型换名。',
        '是。C5先共享head生成候选再加权；C6先混合表示再head，非线性意味着不等价。',
        '7/7，独立seed42，从头训练；best按统一VAL5商品Macro MAE选取。',
        ranking[0]['Control']+'，VAL5='+str(ranking[0]['VAL_5d']),
        str(lower('C0'))+'；差值='+str(byname['C5 vs C0']['VAL_5d_Difference']),
        str(lower('C3'))+'；差值='+str(byname['C5 vs C3']['VAL_5d_Difference']),
        str(lower('C4'))+'；差值='+str(byname['C5 vs C4']['VAL_5d_Difference']),
        str(lower('C6'))+'；差值='+str(byname['C5 vs C6']['VAL_5d_Difference']),
        *[f'{h}日平均权重最大为{EXPERTS[int(weights[i].argmax())]}，完整权重见表2；不是金融因果贡献。' for i,h in enumerate(HORIZONS)],
        '六对期限JS与逐样本JS见HORIZON_ROUTING_DIFFERENCE.csv；权重差异需结合下面交换预测影响解释，不预设短长偏好。',
        '1/20、5/10交换对应误差和预测变化见ROUTER_INTERVENTION_RESULTS.csv，不能据此另选checkpoint。',
        '专家预测分歧/误差相关性与单专家干预已保存。分歧本身不等于可利用互补；不提供可部署Oracle性能。',
        '各期限平均argmax专家为'+str([EXPERTS[i] for i in weights.argmax(-1)])+'；同时提供熵/最大权重，不以argmax同名就断言collapse。',
        '样本路由最大标准差='+str(float(sd.max()))+'；接近常量时应结合均值替换、打乱预测变化判断。',
        ('C5在四项预先指定VAL5比较中均更低，但还需机制干预与跨seed证据。' if all(lower(k) for k in ('C0','C3','C4','C6')) else 'C5未同时通过全部关键VAL5比较，不支持宣布方案整体胜出。')+' Legacy数据已知P0限制阻止无泄漏泛化结论。',
        '预测改善方向、期限路由分化、干预作用和专家互补均需将全部7组在冻结协议下统一做43/44复验；本轮未运行。',
        '否。本轮只读取TRAIN/VAL，未读取TEST数值/标签、未构建TEST loader、未推理TEST。',
    ]
    lines=['# Phase M TRAIN VAL 结果','',answers[19],
        '本轮为用户明确接受已知Legacy数据缺陷后的探索性复验；DATA_CAUSALITY仍FAIL。不能与新TriWeave协议结果混用。','',
        '## 表1 预测性能','',md_table(ranking,['Rank','Control','VAL_1d','VAL_5d','VAL_10d','VAL_20d','Delta5d_vs_D0B','Params','Best_Epoch']),
        '','全部MAE为商品Macro、小数累计收益率；Hit在CSV中为比例。Checkpoint/scheduler/早停均为VAL5 Macro MAE，训练仍四周期Huber+原生KL。',
        '','## 表2 C5不同期限的专家使用','',md_table(profile,['Horizon',*EXPERTS,'mean_sample_entropy_nats']),
        '','## 表3 关键机制对照','',md_table(comparisons,['Comparison','VAL_5d_Difference','Other_Horizon_Effects','Interpretation']),
        '','差值为C5−参照，负值为误差下降。C1/C0同时改变专家、容量和融合，C4/C1及C5/C3同时涉及router容量；不能一概视为单机制因果效应。C5/C6初值和参数量相同，仅融合位置及其随机mask作用对象不同。',
        '','## 22项回答','',*['Q'+str(i)+'. '+a+'\n' for i,a in enumerate(answers,1)],
        '','## 证据边界','',
        '单seed、反复使用过的VAL、Legacy未来bfill与污染标签。四周期变化完整报告，不能以5日最优宣称长期最优。旧Oracle是标签事后选择，不是本方案的可实现性能。冻结单专家干预不是独立训练消融。',
        'Bootstrap仅在显式启用时运行，为40-origin配对移动块，不宣称统计显著或确认性检验。所有历史产物保持不变。完成本轮后停止，无TEST触发、额外seed、结构或超参数搜索。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out,args):
    cfg=read(out/'TRAINING_CONFIG.json')
    if source_hashes()!=cfg['source_hashes'] or cfg['protocol']!=PROTOCOL:raise ValueError('Source/protocol drift; STOP, no silent rerun')
    data,arrays,meta=data_from_cache(Path(cfg['data_cache']),cfg['reference_config']['data_audit'])
    if not (out/'DATA_FINGERPRINT_AUDIT.json').exists():data=audit_data(out)
    authorized_data_gate(out,args.accept_legacy_data_defects)
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Formal seven-run training requires CUDA; no CPU fallback')
    save(out/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,
        device=str(device),device_name=torch.cuda.get_device_name(device),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
    protected={cfg['reference_item']['checkpoint']:cfg['reference_item']['checkpoint_sha256'],cfg['data_cache']:cfg['data_cache_sha256']}
    done={}
    for control in KEYS:
        save(out/'STATUS.json',dict(status='TRAINING_LEGACY_EXPLORATORY',current=control,models_complete=len(done),total=7,test_evaluated=False))
        done[control]=train_one(control,out,cfg,data,device);save(out/'training_index.json',done)
    if any(sha(path)!=digest for path,digest in protected.items()):raise ValueError('Protected original checkpoint/cache changed')
    save(out/'MODELS_FROZEN.json',dict(models=done,config_sha256=sha(out/'TRAINING_CONFIG.json'),test_evaluated=False))
    summarize(out,args.bootstrap)
    print('COMPLETE',out,flush=True)


def main():
    if '--new-data-reference' in sys.argv:
        from cmgm.scripts.phase_m_new_data import main as new_data_main
        return new_data_main()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--audit-data',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--context',type=Path,default=DEFAULT_CONTEXT)
    p.add_argument('--phase-a',type=Path,default=PHASE_A)
    p.add_argument('--accept-legacy-data-defects',action='store_true',help='Only after explicit user acceptance of known-P0 Legacy exploratory scope')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');seed_all(42)
    out=args.resume or prepare(args)
    with (out/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.audit_data:audit_data(out)
            if args.run:run(out,args)
            elif args.summarize:summarize(out,args.bootstrap)
        except Exception:
            save(out/'failure.json',dict(traceback=traceback.format_exc(),time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
            raise


if __name__=='__main__':main()

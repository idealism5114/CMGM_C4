"""Preregistered Phase M on explicitly authorized new data; TRAIN/VAL only."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import fcntl
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from cmgm.models.phase_m_new_data import NewDataPhaseMModel,PhaseMLiveBackbone,matched_new_models
from cmgm.models.phase_m_prediction_moe import KEYS,MODES,EXPERTS,parameter_counts,tensor_state_hash
from cmgm.models.formal_d0b_main_ablation import MainInnovationAblation
from cmgm.scripts.formal_main_innovation_ablation import ROOT,INPUT_KEYS,sha,save,checkpoint,seed_all,state_rng,restore_rng,prediction_loss
from cmgm.scripts.phase_m_analysis import population,routing_rows,expert_competence,intervention_routes
from cmgm.scripts.phase_m_horizon_prediction_moe import csv_file,md_table

HORIZONS=(1,5,10,20)
REFERENCE=ROOT/'experiments/sep17_d0b_triweave/20261022_153322_689200'
PROTOCOL=dict(id='CMGM_PHASE_M_NEW23_SEED42_MACRO5_V1',seed=42,input_nodes=311,feature_dim=26,targets=23,window=20,
 horizons=list(HORIZONS),target_unit='decimal cumulative return',optimizer='Adam',lr=1e-4,weight_decay=1e-5,
 batch_size=64,shuffle=False,drop_last=False,max_epochs=200,patience=10,scheduler_factor=.5,scheduler_patience=5,
 train_loss='SUM_h valid-label Huber(.02) means + native Switch KL ONCE',
 selector='VAL_5D_COMMODITY_MACRO_MAE',selector_formula='mean_c mean_valid_origin abs(pred-target); equal weight over23 commodities',
 epoch0_candidate=False,router_hidden=32,horizon_embedding_dim=8,rank=16,temperature=1.,auxiliary_loss=False,
 training_initialization='Fresh original D0B construction then explicit common/expert/router state copying; no trained weight initialization',
 dropout_note='C5 head body on B,K,64; C6 on B,H,64. K=H=4: same random draw shape/count, distinct operands; eval has no dropout.',
 TEST_policy='No TEST numeric data, labels, arrays or inference. No automatic TEST option.',
 bootstrap=dict(block_origins=40,repeats=2000,seed=42,role='paired date blocks; exploratory single seed',run=True),
 user_data_override='User explicitly chose new311/26/23 dataset over conflicting old21/24 requirement',
 data_limitations='Frozen new-data Research23 is exploratory; historical PIT membership, price release time and continuous futures roll method not certified')
FILES=['cmgm/models/phase_m_new_data.py','cmgm/models/phase_m_prediction_moe.py','cmgm/models/formal_d0b_main_ablation.py',
 'cmgm/models/four_task_moe.py','cmgm/models/horizon_tensor_fusion.py','cmgm/models/frozen_d0b_task_experts.py',
 'cmgm/scripts/phase_m_new_data.py','cmgm/scripts/phase_m_analysis.py','cmgm/scripts/phase_m_horizon_prediction_moe.py',
 'cmgm/scripts/formal_main_innovation_ablation.py','tests/test_phase_m_new_data.py','tests/test_phase_m_prediction_moe.py']

def read(p):return json.loads(Path(p).read_text())
def hashes():return {p:sha(ROOT/p) for p in FILES}

class TrainValDataset(Dataset):
 def __init__(self,path,split):
  if split not in ('train','val'):raise ValueError('Only TRAIN/VAL allowed')
  self.split=split;self.path=Path(path)
  names=['features','input_valid_mask','feature_valid_mask','feature_applicable_mask','reference_dates']
  names += [split+'_'+s for s in ('targets','label_valid_mask','origin_indices','window_indices')]
  self.a={k:np.load(self.path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in names}
  if str(self.a['reference_dates'][-1])>'2024-12-31':raise ValueError('TRAIN/VAL-only timeline gate failed')
 def __len__(self):return len(self.a[self.split+'_origin_indices'])
 def __getitem__(self,k):
  a=self.a;s=self.split;idx=a[s+'_window_indices'][k]
  def win(key):
   v=np.array(a[key][np.maximum(idx,0)],copy=True);v[idx<0]=0;return v
  origin=int(a[s+'_origin_indices'][k])
  return dict(X=win('features'),input_valid_mask=win('input_valid_mask'),feature_valid_mask=win('feature_valid_mask'),
   feature_applicable_mask=np.array(a['feature_applicable_mask'],copy=True),Y=np.array(a[s+'_targets'][k],copy=True),
   label_valid_mask=np.array(a[s+'_label_valid_mask'][k],copy=True),origin_date=str(a['reference_dates'][origin]))

def loader(cfg,split):return DataLoader(TrainValDataset(cfg['data_path'],split),batch_size=64,shuffle=False,drop_last=False,num_workers=0)
def inputs(batch,device):return {k:batch[k].to(device) for k in INPUT_KEYS}
def model_for(key,cfg,device='cpu'):
 return NewDataPhaseMModel(key,cfg['history_root'],**cfg['shape']).to(device)

@torch.inference_mode()
def evaluate(model,batches,device,capture=False):
 model.eval();ps=[];ys=[];ms=[];dates=[];routes=[];candidates=[];norms=[];batch_losses=[]
 for batch in batches:
  p,aux=model(**inputs(batch,device),return_aux=True,with_candidates=capture)
  if p.shape!=batch['Y'].shape or not torch.isfinite(p).all():raise ValueError('Output shape/finite failure')
  ps.append(p.cpu().numpy());ys.append(batch['Y'].numpy());ms.append(batch['label_valid_mask'].numpy());dates+=batch['origin_date']
  batch_losses.append(float(prediction_loss(p,batch['Y'].to(device),batch['label_valid_mask'].to(device))))
  if capture and aux:
   routes.append(aux['routing'].cpu().numpy());candidates.append(aux['expert_prediction'].cpu().numpy())
   norms.append(aux['experts'].norm(dim=-1).cpu().numpy())
 arrays=dict(prediction=np.concatenate(ps),target=np.concatenate(ys),label_valid_mask=np.concatenate(ms),origin_dates=np.array(dates))
 if len(dates)!=len(batches.dataset) or len(set(dates))!=len(dates):raise ValueError('Origin alignment/coverage failure')
 r=population(arrays['prediction'],arrays['target'],arrays['label_valid_mask']);r['Huber_batch_mean_logging_only']=float(np.mean(batch_losses))
 if routes:arrays.update(routing=np.concatenate(routes),expert_prediction=np.concatenate(candidates),expert_norm=np.concatenate(norms))
 return (r,arrays) if capture else r

def temporal_parity():
 a=ast.parse(textwrap.dedent(inspect.getsource(MainInnovationAblation.temporal))).body[0]
 b=ast.parse(textwrap.dedent(inspect.getsource(PhaseMLiveBackbone.temporal))).body[0]
 return ast.dump(ast.Module(body=a.body[:-1],type_ignores=[]))==ast.dump(ast.Module(body=b.body[:-1],type_ignores=[]))

def audit_data(cfg,out):
 ref=Path(cfg['reference']);old=read(ref/'config.json');d=read(ref/'runs/full_d0b/training_complete.json')
 if read(ref/'status.json')['status']!='COMPLETE' or d['name']!='Full D0B' or sha(d['checkpoint'])!=d['checkpoint_sha256']:
  raise ValueError('Original D0B identity gate failed')
 if old['protocol']['full']!='switching_latent_balanced_readout' or old['protocol']['historical_commit']!='9eab8eebea703812652aa609071a4f5bcbce1165':
  raise ValueError('Not the audited original gated D0B')
 for source,identity in read(ref/'historical_source/source_manifest.json').items():
  path=ref/'historical_source'/source.replace('cmgm/','cmgm_sep17_frozen/',1)
  if sha(path)!=identity['namespaced_sha256']:raise ValueError('Sealed original source changed: '+source)
 original_adapter='cmgm/models/formal_d0b_main_ablation.py'
 if sha(ROOT/original_adapter)!=old['source_hashes'][original_adapter]:raise ValueError('Audited mask adapter changed')
 data=Path(cfg['data_path']);manifest=read(data/'dataset_manifest.json')
 if sha(data/'dataset_manifest.json')!=old['dataset_manifest_sha256']:raise ValueError('Frozen data identity changed')
 needed=['features','input_valid_mask','feature_valid_mask','feature_applicable_mask','reference_dates']
 for split in ('train','val'):needed += [split+'_'+s for s in ('targets','label_valid_mask','origin_indices','window_indices')]
 fingerprints={}
 for key in needed:
  path=data/(key+'.npy');expected=manifest['files'][path.name]['sha256']
  if sha(path)!=expected:raise ValueError('Frozen data changed: '+key)
  fingerprints[path.name]=expected
 assets=read(data/'asset_manifest.json');targets=[a['asset_id'] for a in assets if a['market']=='future']
 if targets!=cfg['targets']:raise ValueError('Target ordering differs')
 splits={}
 for split in ('train','val'):
  ds=TrainValDataset(data,split);a=ds.a;origin=a[split+'_origin_indices'];mask=a[split+'_label_valid_mask']
  dates=a['reference_dates'][origin]
  expected=manifest['split_summary'][split.upper()]
  if len(ds)!=expected['origin_count'] or str(dates[0])!=expected['first_origin'] or str(dates[-1])!=expected['last_origin']:
   raise ValueError('Frozen calendar/origin mismatch')
  splits[split]=dict(origins=len(ds),first_origin=str(dates[0]),last_origin=str(dates[-1]),
   valid_labels={str(h):int(mask[:,i].sum()) for i,h in enumerate(HORIZONS)},timeline_last=str(a['reference_dates'][-1]))
 model=model_for('C0',cfg)
 cp=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)
 if cp['name']!='Full D0B' or cp['epoch']!=d['best_epoch']:raise ValueError('Checkpoint metadata failed')
 model.encoder.load_state_dict(cp['model_state_dict'],strict=True);model.set_epoch(cp['epoch'])
 fresh,arrays=evaluate(model,loader(cfg,'val'),'cpu',True)
 # This is a previous VAL artifact only. No test results or TEST arrays opened.
 with np.load(ref/'runs/full_d0b/val_predictions.npz',allow_pickle=False) as saved:
  for k in ('target','label_valid_mask','origin_dates'):
   if not np.array_equal(arrays[k],saved[k]):raise ValueError('VAL origin/label/mask differ: '+k)
  diff=float(np.max(np.abs(arrays['prediction']-saved['prediction'])))
  np.testing.assert_allclose(arrays['prediction'],saved['prediction'],atol=1e-6,rtol=1e-5)
 if not temporal_parity():raise ValueError('Live temporal interface changed computation')
 replay=read(ref/'preprocessing_replay_audit.json')
 if not replay['PASS'] or replay['TEST_numeric_values_parsed']!=0:raise ValueError('Prior bounded replay was not isolated')
 result=dict(status='PASS',dataset_protocol=manifest['protocol_id'],manifest_sha256=sha(data/'dataset_manifest.json'),
  array_hashes=fingerprints,splits=splits,targets=targets,model_shape=cfg['shape'],original_parameters=sum(p.numel() for p in model.parameters()),
  original_checkpoint_sha256=d['checkpoint_sha256'],original_best_epoch=d['best_epoch'],strict_load=True,
  original_VAL_prediction_max_abs_diff=diff,original_VAL5_macro_MAE=fresh['primary'],temporal_math_AST_identical=True,
  previous_bounded_preprocessing_replay=sha(ref/'preprocessing_replay_audit.json'),TEST_arrays_opened=False,
  limitation=PROTOCOL['data_limitations'],historical_TEST_summary_seen_during_version_document_review=True,
  TEST_summary_used_for_selection=False)
 save(out/'DATA_FINGERPRINT_AUDIT.json',result)
 return result

def prepare(out,reference):
 old=read(reference/'config.json');data=Path(old['data_path']);assets=read(data/'asset_manifest.json')
 shape=dict(nodes=len(assets),stocks=sum(a['market']=='stock' for a in assets),bonds=sum(a['market']=='bond' for a in assets),
  targets=len(old['target_order']),features=read(data/'dataset_manifest.json')['feature_count'])
 if shape!=dict(nodes=311,stocks=263,bonds=25,targets=23,features=26):raise ValueError('Unexpected explicitly authorized data shape')
 cfg=dict(protocol=PROTOCOL,reference=str(reference.resolve()),history_root=str(reference.resolve()/'historical_source'),data_path=str(data),
  targets=old['target_order'],shape=shape,checkpoint_root=str(ROOT/'checkpoints/phase_m_horizon_prediction_moe'/out.name),
  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=hashes(),
  created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
 audit=audit_data(cfg,out)
 models,init=matched_new_models(cfg['history_root'],shape)
 save(out/'COMMON_INITIALIZATION_AUDIT.json',init)
 checkpoint(out/'initial_states.pt',{k:{n:v.clone() for n,v in m.state_dict().items()} for k,m in models.items()})
 counts={k:parameter_counts(m) for k,m in models.items()}
 # Actual TRAIN mini-batch backward: count effective parameters before decay/update.
 b=next(iter(loader(cfg,'train')));b={k:v[:2] if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==64 else v for k,v in b.items()}
 active={}
 for key,m in models.items():
  m.eval();p=m(**inputs(b,'cpu'));prediction_loss(p,b['Y'],b['label_valid_mask']).backward()
  missing=[n for n,v in m.named_parameters() if v.grad is None]
  if missing:raise ValueError('Disconnected parameters: '+str(missing))
  if not all(torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('Nonfinite preflight gradient')
  counts[key]['effective_gradient_parameters']=sum(v.numel() for v in m.parameters() if v.grad is not None)
  active[key]=dict(unused_parameter_names=missing,finite=True,PASS=True)
 save(out/'MODEL_REGISTRY.json',{k:dict(model_id=MODES[k],parameters=counts[k]) for k in KEYS})
 csv_file(out/'MODEL_PARAMETER_COUNTS.csv',[dict(Control=k,Model=MODES[k],Params=v['total'],Trainable=v['trainable'],Effective=v['effective_gradient_parameters'],Experts=sum(v['experts'].values()),Router=v['router']) for k,v in counts.items()])
 save(out/'AUDIT_RESULTS.json',dict(status='PREFLIGHT_PASSED',data_identity='PASS',original_D0B='PASS',common_initialization='PASS',
  effective_parameter_checks=active,TEST_arrays_opened=False,training='NOT_RUN'))
 save(out/'TRAINING_CONFIG.json',cfg)
 for relative in cfg['source_hashes']:
  dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
 save(out/'PROVENANCE.json',dict(git_commit=cfg['git_commit'],source_hashes=cfg['source_hashes'],
  prior_workspace=read(out/'WORKSPACE_BEFORE.json'),historical_source_manifest_sha256=sha(reference/'historical_source/source_manifest.json'),
  initial_states_sha256=sha(out/'initial_states.pt'),independent_trained_D0B_used_only_for_audit=True,
  historical_TEST_summary_seen_during_version_document_review=True,TEST_numeric_arrays_read=False,TEST_used_for_selection=False))
 write_docs(out,cfg,audit,counts)
 save(out/'STATUS.json',dict(status='PREPARED',models_complete=0,total=7,test_evaluated=False))
 print('PREPARED',out,flush=True)


def train_one(key,out,cfg,device):
 folder=out/'runs'/key;folder.mkdir(parents=True,exist_ok=True)
 cpdir=Path(cfg['checkpoint_root'])/key;cpdir.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.json').exists():
  d=read(folder/'COMPLETE.json')
  if sha(d['checkpoint'])!=d['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
  return d
 model=model_for(key,cfg,device)
 initial=torch.load(out/'initial_states.pt',map_location='cpu',weights_only=True)[key]
 model.load_state_dict(initial,strict=True);seed_all(42)
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
 train=loader(cfg,'train');val=loader(cfg,'val');history=[];best=float('inf');stale=0;epoch_start=1;best_epoch=None;seconds=0.
 best_path,latest=cpdir/'best.pt',cpdir/'latest.pt'
 if latest.exists():
  cp=torch.load(latest,map_location='cpu',weights_only=False)
  if cp['control']!=key or cp['config_sha256']!=sha(out/'TRAINING_CONFIG.json'):raise ValueError('Resume identity differs')
  model.load_state_dict(cp['model_state_dict'],strict=True);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
  restore_rng(cp['rng']);model.set_epoch(cp['epoch']);history=cp['history'];best=cp['best'];best_epoch=cp['best_epoch'];stale=cp['stale'];epoch_start=cp['epoch']+1;seconds=cp['seconds']
 for epoch in range(epoch_start,201):
  if stale>=10:break
  tic=time.perf_counter();model.train();model.set_epoch(epoch);losses=[];kls=[];dates=[]
  for batch in train:
   optimizer.zero_grad(set_to_none=True);p=model(**inputs(batch,device))
   pred=prediction_loss(p,batch['Y'].to(device),batch['label_valid_mask'].to(device));kl=model.switch_loss();loss=pred+kl
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward()
   if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError('Nonfinite gradient')
   optimizer.step();losses.append(float(pred.detach()));kls.append(float(kl.detach()));dates+=batch['origin_date']
  if len(dates)!=len(train.dataset) or len(set(dates))!=len(dates):raise ValueError('TRAIN coverage differs')
  r=evaluate(model,val,device);score=r['primary'];scheduler.step(score);improved=score<best
  if improved:best=score;best_epoch=epoch;stale=0
  else:stale+=1
  elapsed=time.perf_counter()-tic;seconds+=elapsed
  row=dict(Control=key,epoch=epoch,train_Huber_batch_mean=float(np.mean(losses)),train_KL_batch_mean=float(np.mean(kls)),
   VAL_5D_COMMODITY_MACRO_MAE=score,VAL_MAE={h:v['commodity_macro_MAE'] for h,v in r['horizons'].items()},
   VAL_Huber_valid_label_sum=r['four_horizon_prediction_Huber_sum'],VAL_Huber_batch_mean_logging=r['Huber_batch_mean_logging_only'],
   optimizer_steps=len(losses),train_origins_used=len(dates),seconds=elapsed,lr=optimizer.param_groups[0]['lr'],switch_beta=model.switching_latent_transformer.regime_filter.current_beta)
  history.append(row);save(folder/'history.json',history)
  state=dict(control=key,model_id=MODES[key],epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,seconds=seconds,
   model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=state_rng(),history=history,
   config_sha256=sha(out/'TRAINING_CONFIG.json'),selection=PROTOCOL['selector'],data_audit_sha256=sha(out/'DATA_FINGERPRINT_AUDIT.json'))
  if improved:checkpoint(best_path,state)
  checkpoint(latest,state)
  print(f'[{key}] epoch={epoch} VAL5_macro_MAE={score:.10g} best={best_epoch} seconds={elapsed:.1f}',flush=True)
 cp=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True);model.set_epoch(cp['epoch'])
 results={}
 for split in ('train','val'):
  r,a=evaluate(model,loader(cfg,split),device,True);a['commodity_order']=np.array(cfg['targets']);np.savez_compressed(folder/(split+'_predictions.npz'),**a);results[split]=r
 if not np.isclose(results['val']['primary'],best,atol=1e-10,rtol=1e-7):raise ValueError('Best checkpoint reproduction mismatch')
 save(folder/'metrics.json',results)
 done=dict(control=key,model_id=MODES[key],best_epoch=cp['epoch'],best_VAL5_macro_MAE=best,parameters=parameter_counts(model),training_seconds=seconds,
  checkpoint=str(best_path),checkpoint_sha256=sha(best_path),latest_checkpoint=str(latest),strict_load=True,status='COMPLETE',test_evaluated=False)
 save(folder/'COMPLETE.json',done)
 del model,optimizer,scheduler,cp,initial;gc.collect()
 if device.type=='cuda':torch.cuda.empty_cache()
 return done


def run(out,device):
 cfg=read(out/'TRAINING_CONFIG.json')
 if cfg['source_hashes']!=hashes():raise ValueError('Source drift after preregistration')
 for name,digest in read(out/'DATA_FINGERPRINT_AUDIT.json')['array_hashes'].items():
  if sha(Path(cfg['data_path'])/name)!=digest:raise ValueError('Data drift')
 if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Formal training requires GPU; no CPU fallback')
 save(out/'ENVIRONMENT.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(device),
  physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
 done={}
 for key in KEYS:
  save(out/'STATUS.json',dict(status='TRAINING',current=key,models_complete=len(done),total=7,test_evaluated=False))
  done[key]=train_one(key,out,cfg,device);save(out/'training_index.json',done)
 save(out/'models_frozen.json',done)
 summarize(out)
 save(out/'STATUS.json',dict(status='COMPLETE',models_complete=7,total=7,test_evaluated=False))
 print('COMPLETE',out,flush=True)


def write_docs(out,cfg,audit,counts):
 descriptions={
  'PREREGISTERED_PROTOCOL.md':'# Phase M 新数据预注册协议\n\n'+json.dumps(PROTOCOL,ensure_ascii=False,indent=2)+'\n\n固定按C0至C6顺序各训练seed42一次。只按VAL5商品macro MAE选模，训练仍为四周期Huber加原生KL。C0必须重新训练，不能复用按Huber选出的历史最佳。不得根据结果更换专家、增加seed或读取TEST。',
  'CMGM_D0B_SOURCE_AUDIT.md':'# 原始D0B身份\n\n架构来源9eab8eebea703812652aa609071a4f5bcbce1165。当前HEAD '+cfg['git_commit']+'。封存的原始AdaptiveGate D0B，新数据mask adapter不增加参数。原Full参数量 '+str(audit['original_parameters'])+'；checkpoint SHA256 '+audit['original_checkpoint_sha256']+'；原best epoch '+str(audit['original_best_epoch'])+'。strict=True恢复，仅用于身份/VAL复现，不作为训练初始化。VAL预测最大差 '+str(audit['original_VAL_prediction_max_abs_diff'])+'。\n\n新增temporal子类方法在最终return以前与原mask adapter AST完全一致。默认调用值不变，只暴露已计算的live H/Z。原空间和时间文件未修改。输入311/26，输出23，目标/日期来源冻结manifest，用户明确选择该新协议。',
  'EXPERT_ARCHITECTURE.md':'# 共享四专家\n\nE1=ReLU(W[Z_last,mean5Z,Z_last-mean5Z])，12352参数。它复用旧RecentChange的投影机制并显式增加均值输入，比旧128→64增加4096参数。Z包含递推历史，不是仅最近五日原始数据。\n\nE2=ReLU(W[H_last,mean20H])，16448参数，直接复用LongMemoryExpert。\n\nE3=s+ReLU(Ws)，4160参数，直接复用SpatialStructureExpert。\n\nE4=Wo[(Us s)⊙(Ut t)]，rank16，3072参数；复用旧LowRankInteraction矩阵，按本轮公式去掉旧t残差。全部MoE共享同一专家定义及初值。\n\n四专家每次各计算一次。Prediction MoE：一次共享Head body作用于B,K,64，然后输出权重view(H,C,64)取对应行，得到B,H,K,C，再乘路由。C6只改变融合位置；初始state_dict与C5完全相同。没有复制专家头、输出归一化或辅助损失。',
  'ROUTER_ARCHITECTURE.md':'# 六个路由对照\n\nC1单组4logits；C2两组4logits按[short,short,long,long]；C3四组4logits。C4 context=[LN(s),LN(t),LN(Zlast-mean5Z),LN(Linear128→64(mean20H))]，256→32→4，GELU。C5/C6在context后拼接4×8周期embedding，264→32→4。温度1，四专家全部可访问。最后层Normal(0,.001)、bias0；embedding Normal(0,.02)。C4与C5的共同context变换、第一层context列及最后层显式复制，C5/C6全参数相同；容量差288参数记录为限制。\n\n静态路由不读取样本，动态路由不读取标签，无固定专家掩码、warmup、entropy或balance loss。C5 head Dropout作用于expert，C6作用于horizon：同shape/随机数数量但作用对象不同，不能宣称训练轨迹相同。',
 }
 for name,text in descriptions.items():(out/name).write_text(text+'\n')


def bootstrap_masked(candidate,reference,target,mask):
 rng=np.random.default_rng(42);n=len(target);block=40;repeats=2000
 difference=np.where(mask,np.abs(candidate.astype('float64')-target)-np.abs(reference.astype('float64')-target),0.)
 values=[]
 for _ in range(repeats):
  starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
  idx=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
  denominator=mask[idx].sum(0)
  if (denominator==0).any():raise ValueError('Bootstrap sample has unevaluable commodity/horizon')
  values.append((difference[idx].sum(0)/denominator).mean(-1))
 return dict(block_origins=block,repeats=repeats,seed=42,method='paired noncircular moving date blocks; all commodities grouped; valid-label commodity-macro',
  CI95={str(h):np.quantile(np.asarray(values)[:,i],[.025,.975]).tolist() for i,h in enumerate(HORIZONS)},
  interpretation='Exploratory uncertainty only, not statistical significance or untouched validation')


def summarize(out):
 cfg=read(out/'TRAINING_CONFIG.json');done={};rs={};arrays={};histories=[];checkpoint_audit={}
 if cfg['source_hashes']!=hashes():raise ValueError('Source drift')
 for key in KEYS:
  folder=out/'runs'/key;d=read(folder/'COMPLETE.json');cp=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)
  if sha(d['checkpoint'])!=d['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
  restored=model_for(key,cfg);restored.load_state_dict(cp['model_state_dict'],strict=True);restored.set_epoch(cp['epoch'])
  history=read(folder/'history.json');best=min(history,key=lambda row:row['VAL_5D_COMMODITY_MACRO_MAE'])
  if cp['epoch']!=best['epoch'] or cp['selection']!=PROTOCOL['selector']:raise ValueError('Selection audit differs')
  if not all(row['train_origins_used']==2726 and row['optimizer_steps']==43 for row in history):raise ValueError('TRAIN coverage changed')
  with np.load(folder/'val_predictions.npz',allow_pickle=False) as a:arrays[key]={k:a[k] for k in a.files}
  a=arrays[key];r=population(a['prediction'],a['target'],a['label_valid_mask'])
  saved=read(folder/'metrics.json')['val']
  if r['primary']!=saved['primary']:raise ValueError('Offline metric recomputation differs')
  if key!='C0':
   for name in ('target','label_valid_mask','origin_dates','commodity_order'):
    if not np.array_equal(a[name],arrays['C0'][name]):raise ValueError('Evaluation population differs')
  checkpoint_audit[key]=dict(strict_load=True,best_epoch=cp['epoch'],checkpoint_sha256=d['checkpoint_sha256'],selection=cp['selection'],origin_coverage=True,metrics_recomputed=True)
  done[key]=d;rs[key]=r;histories+=history
  del restored,cp
 ranking=[];allh=[];per=[];routes=[];js=[];competence=[];correlations=[];norms=[]
 for rank,key in enumerate(sorted(KEYS,key=lambda k:rs[k]['primary']),1):
  r=rs[key];d=done[key]
  ranking.append(dict(Rank=rank,Control=key,Model=MODES[key],**{f'VAL_{h}d':r['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS},
   Delta5d_vs_D0B=r['primary']-rs['C0']['primary'],Params=d['parameters']['total'],Best_Epoch=d['best_epoch'],
   Training_seconds=d['training_seconds'],VAL5_RMSE=r['horizons']['5']['RMSE'],VAL5_Hit_percent=100*r['horizons']['5']['Hit']))
 for key in KEYS:
  a=arrays[key]
  for h,v in rs[key]['horizons'].items():allh.append(dict(Control=key,Horizon=int(h),**v))
  for item in rs[key]['per_commodity']:
   if item['horizon']==5:per.append(dict(Control=key,commodity=cfg['targets'][item['commodity_index']],**item))
  if key=='C0':continue
  rr,jj=routing_rows(key,a['routing']);routes+=rr;js+=jj
  cc,pp=expert_competence(key,a['expert_prediction'],a['target'],cfg['targets'],a['label_valid_mask']);competence+=cc;correlations+=pp
  for i,name in enumerate(EXPERTS):norms.append(dict(Control=key,Expert=name,mean_L2=float(a['expert_norm'][:,i].mean()),std_L2=float(a['expert_norm'][:,i].std())))
 c5=arrays['C5']
 with np.load(out/'runs/C5/train_predictions.npz',allow_pickle=False) as train:train_routes=np.array(train['routing'])
 operations=intervention_routes(train_routes,c5['routing']);interventions=[];saved_interventions={}
 native=np.einsum('nhk,nhkc->nhc',c5['routing'],c5['expert_prediction'])
 np.testing.assert_allclose(native,c5['prediction'],atol=1e-7,rtol=1e-6)
 for name,w in operations.items():
  p=np.einsum('nhk,nhkc->nhc',w,c5['expert_prediction']);r=population(p,c5['target'],c5['label_valid_mask'])
  row=dict(Intervention=name,VAL5_macro_MAE=r['primary'],Delta5d_vs_native=r['primary']-rs['C5']['primary'],
   prediction_mean_absolute_change=float(np.mean(np.abs(p.astype('float64')-c5['prediction']))),
   routing_mean_L1_change=float(np.abs(w-c5['routing']).sum(-1).mean()))
  for h in HORIZONS:row[f'Delta_{h}d_MAE']=r['horizons'][str(h)]['commodity_macro_MAE']-rs['C5']['horizons'][str(h)]['commodity_macro_MAE']
  interventions.append(row);saved_interventions[name]=p
 np.savez_compressed(out/'C5_VAL_frozen_intervention_predictions.npz',**saved_interventions,
  target=c5['target'],label_valid_mask=c5['label_valid_mask'],origin_dates=c5['origin_dates'],commodity_order=c5['commodity_order'])
 pairs=[('C1','C0'),('C2','C1'),('C3','C2'),('C4','C1'),('C5','C3'),('C5','C4'),('C5','C6'),('C5','C0')]
 comparisons=[]
 for candidate,reference in pairs:
  row=dict(Comparison=candidate+' vs '+reference,VAL5_difference=rs[candidate]['primary']-rs[reference]['primary'])
  for h in HORIZONS:row[f'Delta_{h}d']=rs[candidate]['horizons'][str(h)]['commodity_macro_MAE']-rs[reference]['horizons'][str(h)]['commodity_macro_MAE']
  row['Interpretation']='candidate lower' if row['VAL5_difference']<0 else 'candidate not lower'
  comparisons.append(row)
 main_pairs=[next(r for r in comparisons if r['Comparison']=='C5 vs '+k) for k in ('C0','C3','C4','C6')]
 for name,rows in [('TRAINING_HISTORY.csv',histories),('VAL_5D_RANKING.csv',ranking),('VAL_ALL_HORIZONS.csv',allh),
  ('EXPERT_ROUTING_BY_HORIZON.csv',routes),('EXPERT_FORECAST_COMPETENCE.csv',competence),('EXPERT_PREDICTION_CORRELATION.csv',correlations),
  ('ROUTER_INTERVENTION_RESULTS.csv',interventions),('HORIZON_ROUTING_DIFFERENCE.csv',js),('PER_COMMODITY_5D_RESULTS.csv',per),
  ('EXPERT_OUTPUT_NORMS.csv',norms),('KEY_MECHANISM_COMPARISONS.csv',main_pairs),('ALL_CONTROL_COMPARISONS.csv',comparisons)]:csv_file(out/name,rows)
 bootstrap={k:bootstrap_masked(c5['prediction'],arrays[k]['prediction'],c5['target'],c5['label_valid_mask']) for k in ('C0','C3','C4','C6')}
 save(out/'BOOTSTRAP_EXPLORATORY.json',bootstrap);save(out/'CHECKPOINT_AUDIT.json',checkpoint_audit)
 save(out/'VAL_METRICS.json',rs)
 current_audit=read(out/'AUDIT_RESULTS.json');current_audit.update(status='COMPLETE',models_complete=7,checkpoints='PASS',metrics='PASS',test_evaluated=False,bootstrap='COMPLETE_EXPLORATORY')
 save(out/'AUDIT_RESULTS.json',current_audit)
 c5routes=[r for r in routes if r['Model']=='C5'];c5js=[r for r in js if r['Model']=='C5']
 lines=['# Phase M Horizon Conditioned Spatiotemporal Prediction MoE','',
  f"Seven fresh seed42 TRAIN/VAL experiments COMPLETE. Lowest VAL5 commodity-macro MAE: {ranking[0]['Control']} ({ranking[0]['VAL_5d']:.10f}).",
  'All numerical differences below are candidate minus reference: negative means lower error. No TEST inference, dataset or labels.','',
  '## Table 1 Prediction performance','',md_table(ranking,['Rank','Control','VAL_1d','VAL_5d','VAL_10d','VAL_20d','Delta5d_vs_D0B','Params','Best_Epoch']),
  '\n## Table 2 C5 expert use by horizon\n',md_table(c5routes,['Horizon',*EXPERTS,'mean_sample_entropy_nats']),
  '\n## Table 3 Key mechanisms\n',md_table(main_pairs,['Comparison','VAL5_difference','Delta_1d','Delta_10d','Delta_20d','Interpretation']),
  '\n## Frozen C5 interventions\n',md_table(interventions,['Intervention','VAL5_macro_MAE','Delta5d_vs_native','prediction_mean_absolute_change']),
  '\n## Required answers\n']
 answers={1:'Yes: independently identified original gated D0B from sealed Sep17 source. Historical checkpoint is audit-only; every fit starts fresh.',
  2:'Yes. Spatial graph/time weighting, market tokens, causal Transformer, regime/generators and balanced readout are unchanged. The temporal method is AST-identical until its live-tensor return.',
  3:'Reused LongMemory and SpatialStructure directly. RecentChange adds the explicitly requested mean input; LowRankInteraction reuses rank16 matrices but removes the old temporal residual, per this protocol.',
  4:'Yes. C5 predicts each expert before mixing; C6 mixes states before the same nonlinear head. Entire initial state is identical; dropout acts on different operands in training.',
  5:'All seven seed42 fits and strict best checkpoint restores are complete.',6:ranking[0]['Control']+' has lowest VAL5 macro MAE.',
  17:'Candidate forecast correlations/disagreements and per-expert MAE are in competence/correlation CSVs. Jointly trained single-expert inference is not an independently trained model. Differences alone do not establish useful complementarity.',
  18:'All horizon argmax expert names: '+str([max(EXPERTS,key=lambda k:r[k]) for r in c5routes])+'. Same argmax is not proof of collapse; inspect weights and entropy.',
  19:'C5 maximum across-sample expert-weight std: '+str(max(r[k+'_std'] for r in c5routes for k in EXPERTS))+'. Variation alone is not evidence of routing benefit.',
  20:'Prediction evidence: '+str({k:rs['C5']['primary']-rs[k]['primary'] for k in ('C0','C3','C4','C6')})+'. Mechanism evidence must additionally use mean/shuffled/swapped-route interventions. No general causal or significance claim.',
  21:'All prediction and division-of-labor claims require matched seeds43/44 over all seven fixed variants, not only the winners; those runs were not started.',
  22:'No TEST arrays/prices, TEST labels, loader or inference were accessed by Phase M. During initial version-document review, historical saved TEST summary numbers were visible; they were not used to define architecture, hyperparameters, ranking or training. This is not a first-blind-test claim.'}
 for num,other in [(7,'C0'),(8,'C3'),(9,'C4'),(10,'C6')]:
  delta=rs['C5']['primary']-rs[other]['primary'];answers[num]=f'C5 vs {other}: delta5d={delta:+.12g}; '+('lower' if delta<0 else 'not lower')+'. Single-run evidence only.'
 for num,row in zip(range(11,15),c5routes):answers[num]=f'{row["Horizon"]}d mean weights: '+str({k:row[k] for k in EXPERTS})+'. These are routing weights, not economic causal influence.'
 answers[15]='C5 horizon-pair mean-route JS nats: '+str([r['JS_mean_routes_nats'] for r in c5js])+'. Numerical horizon differences are reported without forcing a specialization interpretation.'
 answers[16]='See swap_1d_20d and swap_5d_10d rows; actual prediction/MAE changes are reported. No checkpoint was selected using these interventions.'
 for num in range(1,23):lines.append(f'Q{num}. {answers[num]}\n')
 lines+=['## Scope and limitations\n',
  'This explicitly authorized new-data protocol uses311 nodes,26 features,23 targets; old21/24 data and known Legacy bfill pollution are not used. TRAIN2726, VAL483; frozen dates/scaler/labels/masks retained. First19 TRAIN origins have padding; expert means include the original zero/carry padded live sequence positions, consistently across all controls.',
  'Primary metric is valid-origin MAE per commodity then equal commodity mean. Four-horizon Huber training is retained, but all checkpoint/scheduler/earlystop decisions now use VAL5 macro MAE. Historic Huber-selected checkpoints are never ranked as new C0.',
  'Single seed and previously used VAL are exploratory. Bootstrap block40 origins covers overlapping20-session targets; whole dates and commodity vectors stay paired. Intervals do not establish statistical significance. Historical PIT membership/release-time and continuous-futures roll certification remain unverified.',
  'C1 versus C0 changes capacity and expert architecture. C4/C5 differ by288 router parameters. C5/C6 share expert/router/head initialization and structures but train dropout on different operands. No loss balance, routing warmup, oracle selection, parameter search, additional seeds or TEST follow-up. STOP.']
 (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
 save(out/'file_index.json',dict(output=str(out),models=done,reports={p.name:str(p) for p in out.iterdir() if p.suffix in ('.md','.json','.csv','.txt')},test_evaluated=False))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--new-data-reference',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--run',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--device',default='cuda:0')
 args=p.parse_args();torch.set_num_threads(4);seed_all()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 with (out/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   if not (out/'TRAINING_CONFIG.json').exists():prepare(out,args.new_data_reference)
   if args.run:run(out,torch.device(args.device))
   elif args.summarize:summarize(out)
  except Exception:
   save(out/('failure_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'.json'),dict(traceback=traceback.format_exc()))
   raise

if __name__=='__main__':main()

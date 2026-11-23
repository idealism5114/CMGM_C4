"""One-time byte-verified export. Source repository is strictly read-only.

Only this migration tool knows the old repository. Runtime imports never do.
TEST files are copied/hashed as opaque bytes; no TEST arrays are parsed.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

OLD=Path('/home/yangxiaotong/projects/myresearch/Commedities')
NEW=Path(__file__).resolve().parents[1]
MAIN=OLD/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381'
BASE=OLD/'experiments/baselines_deep_v3_c4_val5/seed42_20261109'
ABL=OLD/'experiments/phase_m_c4_ablation/seed42_20261026_r1'
HIST=OLD/'experiments/sep17_d0b_triweave/20261022_153322_689200/historical_source'
FROZEN=HIST/'cmgm_sep17_frozen'
records=[];extractions=[]


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(rel,s):
    p=NEW/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def cp(src,rel,category):
    src=Path(src);dst=NEW/rel;dst.parent.mkdir(parents=True,exist_ok=True);h=sha(src)
    if dst.exists():
        if sha(dst)!=h:raise ValueError('Conflicting destination: '+str(dst))
    else:shutil.copy2(src,dst)
    if sha(dst)!=h or src.stat().st_ino==dst.stat().st_ino:raise ValueError('Copy failed or hardlinked: '+str(src))
    records.append(dict(source=str(src),destination=str(dst.relative_to(NEW)),sha256=h,bytes=src.stat().st_size,category=category))
def tree(src,rel,category):
    for p in sorted(Path(src).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='runner.lock':cp(p,Path(rel)/p.relative_to(src),category)
def extract(src,names,header='',replace=None):
    text=Path(src).read_text();a=ast.parse(text);parts=[]
    for n in a.body:
        name=getattr(n,'name',None)
        if isinstance(n,ast.Assign):name=getattr(n.targets[0],'id',None)
        if name in names:
            parts.append(ast.get_source_segment(text,n))
            extractions.append(dict(source=str(src),source_sha256=sha(src),symbol=name,first_line=n.lineno,last_line=n.end_lineno))
    missing=set(names)-{x['symbol'] for x in extractions if x['source']==str(src)}
    if missing:raise ValueError('Missing extraction: '+str(missing))
    result=header+'\n\n'+'\n\n\n'.join(parts)+'\n'
    for a,b in (replace or {}).items():result=result.replace(a,b)
    return result


def export_sources():
    for folder in ('models','models/_native','baselines','data','training','evaluation'):
        write('src/cmgm_c4/'+folder+'/__init__.py','')
    write('src/cmgm_c4/__init__.py','"""Stable original C4, Deep V3 and its thirteen registered ablation rows."""\n__version__="1.0.0"\n')
    hdr='"""Extracted original mathematical primitives; provenance in provenance/extractions.json."""\nimport math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nFEATURE_DIM=21\nLSTM_HIDDEN_DIM=64\n'
    write('src/cmgm_c4/models/_native/spatial.py',extract(FROZEN/'models/hetero_mixhop_model.py',
        ['_TypeInputProjection','EdgeAttnMixHop','_TypeMeanPool'],hdr))
    write('src/cmgm_c4/models/_native/market.py',extract(FROZEN/'models/switching_transformer.py',['MarketAwareTemporalEncoder'],hdr))
    write('src/cmgm_c4/models/_native/graph.py',(FROZEN/'graph/adaptive_graph.py').read_text())
    write('src/cmgm_c4/models/_native/mixhop.py',extract(FROZEN/'models/model.py',['MixHopPropagation'],hdr))
    write('src/cmgm_c4/models/_native/temporal.py',(FROZEN/'models/switching_latent_transformer.py').read_text().replace(
        'from cmgm_sep17_frozen.models.switching_transformer import MarketAwareTemporalEncoder','from .market import MarketAwareTemporalEncoder'))
    # Build ONLY the original balanced SLBR modules, in original construction order.
    write('src/cmgm_c4/models/_native/native.py','''"""Original D0B module inventory, no experimental variant registry."""
from torch import nn
from .spatial import _TypeInputProjection,EdgeAttnMixHop,_TypeMeanPool
from .graph import AdaptiveGraphLearner
from .temporal import SwitchingLatentTransformerBranch

class NativeSLBR(nn.Module):
    def __init__(self,num_nodes,n_commodities,n_stock=263,n_bond=25,feat_dim=26):
        super().__init__()
        self.num_nodes=num_nodes;self.n_commodities=n_commodities;self.n_stock=n_stock;self.n_bond=n_bond
        self.feat_dim=feat_dim;self.n_horizons=4;self.variant='switching_latent_balanced_readout'
        self.type_proj=_TypeInputProjection(feat_dim,64)
        self.graph_learner=AdaptiveGraphLearner(num_nodes,embed_dim=10,alpha=.5,top_k=10)
        self.attn_mixhop1=EdgeAttnMixHop(64,64,K=2,beta=.05,n_heads=8,dropout=.1,hard_mask=False,prior_scale=.5,self_heads=0,cross_mask=None,graph_prior_heads=None,qk_norm=False)
        self.attn_mixhop2=EdgeAttnMixHop(64,64,K=2,beta=.05,n_heads=8,dropout=.1,hard_mask=False,prior_scale=.5,self_heads=0,cross_mask=None,graph_prior_heads=None,qk_norm=False)
        self.gcn_norm=nn.LayerNorm(64)
        self.type_pool=_TypeMeanPool(64,n_stock,n_bond)
        # Gate is constructed at the original point to preserve random draws;
        # C4 deletes it after constructing the entire encoder, as before.
        self.gate_fc=nn.Linear(128,64)
        self.gcn_proj=nn.Linear(64,64);self.lstm_proj=nn.Linear(64,64)
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,4*n_commodities))
        self.temporal_score=nn.Linear(64,1)
        self.switching_latent_transformer=SwitchingLatentTransformerBranch(
            feat_dim=feat_dim,n_stock=n_stock,n_bond=n_bond,n_commodity=num_nodes-n_stock-n_bond,
            node_dim=32,d_model=128,n_heads=4,n_layers=2,ffn_dim=256,dropout=.3,max_len=20,
            K=3,z_dim=64,sticky_alpha=.5,tau=1.,beta_max=5e-4,warmup_epochs=20,
            output_dim=64,balanced_readout=True,forecast_horizons=(1,5,10,20))
''')
    s=extract(OLD/'cmgm/models/formal_d0b_main_ablation.py',
        ['NAMES','KEY','DEFINITIONS','masked_softmax','zero','MainInnovationAblation'],hdr+'''
from types import SimpleNamespace
from ._native.native import NativeSLBR
from ._native.mixhop import MixHopPropagation

def native_modules(history_root=None):
    return SimpleNamespace(HeteroMixHopCMGM=lambda nodes,targets,n_stock,n_bond,variant,feat_dim: NativeSLBR(nodes,targets,n_stock,n_bond,feat_dim),MixHopPropagation=MixHopPropagation)
''')
    s=s.replace('def __init__(self, name, history_root, *,','def __init__(self, name, history_root=None, *,')
    write('src/cmgm_c4/models/backbone.py',s)
    write('src/cmgm_c4/models/temporal.py',extract(OLD/'cmgm/models/phase_m_new_data.py',['PhaseMLiveBackbone'],hdr+'\nfrom .backbone import MainInnovationAblation,masked_softmax,zero\n'))
    s=extract(OLD/'cmgm/models/four_task_moe.py',['LongMemoryExpert','SpatialStructureExpert','LowRankInteractionExpert'],hdr)
    s+=extract(OLD/'cmgm/models/phase_m_prediction_moe.py',['MeanAugmentedRecentExpert','PureLowRankInteractionExpert','ExpertBank','expert_forecasts','checked_weights'],
        replace={'class MeanAugmentedRecentExpert(RecentChangeExpert):':'class MeanAugmentedRecentExpert(nn.Module):'})
    write('src/cmgm_c4/models/experts.py',s)
    old=ast.parse((OLD/'cmgm/models/phase_m_prediction_moe.py').read_text());router=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='PhaseMRouter')
    context=ast.get_source_segment((OLD/'cmgm/models/phase_m_prediction_moe.py').read_text(),next(n for n in router.body if getattr(n,'name',None)=='context'))
    write('src/cmgm_c4/models/router.py','''"""The original C4 global dynamic router only; four horizons share its weights."""
import torch
from torch import nn

class C4Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.long_projection=nn.Linear(128,64)
        self.norm_s=nn.LayerNorm(64);self.norm_t=nn.LayerNorm(64)
        self.norm_change=nn.LayerNorm(64);self.norm_long=nn.LayerNorm(64)
        self.network=nn.Sequential(nn.Linear(256,32),nn.GELU(),nn.Linear(32,4))
        nn.init.normal_(self.network[-1].weight,mean=0.,std=1e-3);nn.init.zeros_(self.network[-1].bias)

'''+''.join('    '+l+'\n' for l in context.splitlines())+'''
    def forward(self,s,t,h,z):
        return self.network(self.context(s,t,h,z)).softmax(-1)[:,None].expand(-1,4,-1)
''')
    s=extract(OLD/'cmgm/models/phase_m_new_data.py',['NewDataPhaseMModel'],hdr+'''
from .temporal import PhaseMLiveBackbone
from .experts import ExpertBank,expert_forecasts,checked_weights
from .router import C4Router
KEYS=('C0','C4')
MODES={'C0':'moe_phase_m/d0b_original','C4':'moe_phase_m/pred_global_dynamic'}
''')
    s=s.replace('def __init__(self, control, history_root, **shape):','def __init__(self, control="C4", history_root=None, **shape):')
    s=s.replace('self.router=PhaseMRouter(control)','self.router=C4Router()')
    start=s.index("        if self.control=='C6':")
    end=s.index('        aux=dict(',start)
    s=s[:start]+"        candidates=expert_forecasts(self.base.head,bank,self.n_commodities)\n        prediction=torch.einsum('bhk,bhkc->bhc',pi,candidates)\n"+s[end:]
    s+='\n\nclass C4Model(NewDataPhaseMModel):\n    def __init__(self,**shape):\n        super().__init__("C4",**shape)\n'
    write('src/cmgm_c4/models/c4.py',s)
    s=(OLD/'cmgm/models/phase_m_c4_ablation.py').read_text()
    s=s.replace('from cmgm.models.phase_m_new_data import NewDataPhaseMModel','from .c4 import NewDataPhaseMModel')
    s=s.replace('from cmgm.models.formal_d0b_main_ablation import','from .backbone import').replace('from cmgm.models.phase_m_prediction_moe import expert_forecasts','from .experts import expert_forecasts')
    s=s.replace('def __init__(self, ablation, history_root, *,','def __init__(self, ablation, history_root=None, *,')
    write('src/cmgm_c4/models/ablations.py',s)
    # Eight latest baseline definitions and their exact self-contained cores.
    wanted=['HORIZONS','TRAINABLE','ORDER','BaselineMultimodalAdapter','Base','GRUBaseline','VanillaTransformerBaseline',
        'GraphTemporalBlock','MTGNNBaseline','RNNBaseline','LSTMBaseline','GraphWaveNetBaseline','MSGNetBaseline','CrossGNNBaseline','MODELS','make_model']
    s=extract(OLD/'cmgm/models/comparison_baselines.py',wanted,hdr+'''
from .graph import AdaptiveGraphLearner
from .mixhop import MixHopPropagation
''',replace={'from cmgm.models.baselines.graph_wavenet':'from .graph_wavenet',
             'from cmgm.models.baselines.msgnet':'from .msgnet','from cmgm.models.baselines.crossgnn':'from .crossgnn'})
    write('src/cmgm_c4/baselines/models.py',s)
    for n in ('graph_wavenet.py','msgnet.py','crossgnn.py'):write('src/cmgm_c4/baselines/'+n,(OLD/'cmgm/models/baselines'/n).read_text())
    write('src/cmgm_c4/baselines/graph.py',(OLD/'cmgm/graph/adaptive_graph.py').read_text())
    write('src/cmgm_c4/baselines/mixhop.py',extract(OLD/'cmgm/models/model.py',['MixHopPropagation'],hdr))
    s=(OLD/'cmgm/models/triweave_baseline_adapter.py').read_text().replace('from cmgm.models.comparison_baselines import make_model','from .models import make_model')
    write('src/cmgm_c4/baselines/adapter.py',s)
    s=extract(OLD/'cmgm/scripts/phase_m_analysis.py',['HORIZONS','population'],'''"""Original float64 C4 metric definitions; no TEST or fitting side effects."""\nimport numpy as np''')
    write('src/cmgm_c4/evaluation/metrics.py',s)


def export_artifacts():
    import torch
    main=read(MAIN/'TRAINING_CONFIG.json');base=read(BASE/'config.json');abl=read(ABL/'config.json')
    mainrow=read(MAIN/'runs/C4/COMPLETE.json');ablrows=read(ABL/'training_index.json')
    baserows=read(BASE/'models_frozen.json')
    if 'models' in baserows:baserows=baserows['models']
    # Exact preprocessed data and saved standardization; no split or scaler fit.
    tree(Path(main['data_path']),'data/train_val','TRAIN_VAL_frozen_data')
    test=Path(abl['test_release_path']);tree(test,'data/test_release','TEST_opaque_transport')
    tree(Path(base['train_val_cache']),'data/baseline_train_val','baseline_neutral_TRAIN_VAL')
    # Baseline TEST cache path is read only as metadata, then copied as bytes.
    oldbase=OLD/'experiments/baseline_comparison_deep_v3_triweave/20261022_184258_600773'
    if (oldbase/'neutral_test').exists():tree(oldbase/'neutral_test','data/baseline_test_release','TEST_opaque_transport')
    registry={}
    def model_record(suite,key,row,resultfolder):
        ident=suite+'/'+key;best=Path(row['checkpoint'])
        if sha(best)!=row['checkpoint_sha256']:raise ValueError('Checkpoint SHA mismatch '+ident)
        latest=Path(row.get('latest_checkpoint',best.with_name('latest.pt')))
        for p in (best,latest):cp(p,Path('checkpoints')/suite/key/p.name,'checkpoint')
        tree(resultfolder,Path('results')/suite/key,'archived_results')
        registry[ident]=dict(suite=suite,key=key,checkpoint=f'checkpoints/{suite}/{key}/best.pt',
            latest=f'checkpoints/{suite}/{key}/latest.pt',checkpoint_sha256=sha(best),latest_sha256=sha(latest),
            best_epoch=row['best_epoch'],selection='VAL_5D_COMMODITY_MACRO_MAE',results=f'results/{suite}/{key}',
            original_checkpoint=str(best),original_record=row)
    model_record('c4','full',mainrow,MAIN/'runs/C4')
    for key,row in baserows.items():model_record('baselines',key,row,BASE/'runs'/key)
    for key,row in ablrows.items():model_record('ablations',key,row,ABL/'runs'/key)
    c4test=OLD/'experiments/phase_m_horizon_prediction_moe_test/seed42_20261026_frozen_best/runs/C4'
    for n in ('test_predictions.npz','result.json'):
        if (c4test/n).exists():cp(c4test/n,Path('results/c4/full')/n,'TEST_archived_result_opaque')
    # Original configs/checkpoint metadata remain byte-identical; new paths live
    # in the registry only, never patch an old checkpoint or old run config.
    for folder,name in ((MAIN,'c4'),(BASE,'baselines'),(ABL,'ablations')):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.json','.md','.csv','.txt'):
                cp(p,Path('provenance/original_runs')/name/p.name,'historical_metadata')
        snap=folder/('source_snapshot')
        if snap.exists():tree(snap,Path('provenance/original_sources')/name,'source_snapshot')
    tree(HIST,'provenance/sep17_sealed_source','sealed_reference_only_not_runtime')
    # Store freshly initialized states with original tensor keys, not warm starts.
    maininit=torch.load(MAIN/'initial_states.pt',map_location='cpu',weights_only=True)
    (NEW/'initial_states').mkdir(exist_ok=True)
    torch.save({k:maininit[k] for k in ('C4','C0')},NEW/'initial_states/c4.pt')
    cp(BASE/'initial_states.pt','initial_states/baselines.pt','fresh_initialization')
    cp(ABL/'initial_states.pt','initial_states/ablations.pt','fresh_initialization')
    tree(OLD/'figures/c4_architecture/20261112_v1','figures/c4_architecture','figure')
    for name in ('LICENSE','LICENSE.md','LICENSE.txt'):
        if (OLD/name).exists():cp(OLD/name,name,'license')
    for p in (OLD/'cmgm/models/baselines').iterdir():
        if p.suffix in ('.md','.json'):cp(p,Path('provenance/baseline_adaptations')/p.name,'baseline_adaptation_source')
    common=dict(protocol_id='CMGM_C4_STABLE_V1',shape=main['shape'],targets=main['targets'],
        horizons=[1,5,10,20],data_path='data/train_val',test_path='data/test_release',
        baseline_data_path='data/baseline_train_val',baseline_test_path='data/baseline_test_release',
        train=dict(seed=42,optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,shuffle=False,drop_last=False,
            max_epochs=200,patience=10,scheduler_factor=.5,scheduler_patience=5,Huber_delta=.02,epoch0_candidate=False),
        selector='VAL_5D_COMMODITY_MACRO_MAE',Switch_KL='Native .0005/warmup20 once for C4 and enabled ablations; never add to baselines',
        dataset_manifest_sha256=sha(Path(main['data_path'])/'dataset_manifest.json'),
        inactive_parameters=abl['inactive_parameters'],
        source_archives={'c4':'provenance/original_runs/c4/TRAINING_CONFIG.json','baselines':'provenance/original_runs/baselines/config.json','ablations':'provenance/original_runs/ablations/config.json'},
        limits='Single seed, repeatedly viewed historical TEST. PIT membership, release timing and continuous futures roll remain unverified.')
    save(NEW/'configs/project.json',common);save(NEW/'configs/registry.json',registry)
    for suite,keys in [('c4',['full']),('baselines',list(baserows)),('ablations',list(ablrows))]:
        for key in keys:save(NEW/f'configs/{suite}/{key}.json',dict(suite=suite,key=key,project='../project.json',selector=common['selector']))
    save(NEW/'provenance/initial_states.json',dict(c4_source=str(MAIN/'initial_states.pt'),source_file_sha256=sha(MAIN/'initial_states.pt'),
        selected_keys=['C4','C0'],destination='initial_states/c4.pt',destination_sha256=sha(NEW/'initial_states/c4.pt'),
        statement='Tensor-preserving extraction of untrained states; no trained weights used for initialization'))
    return common,registry


def main():
    if (NEW/'provenance/migration_manifest.json').exists():
        raise ValueError('This export is already complete. Refuse to overwrite the standalone project.')
    (NEW/'provenance').mkdir(parents=True,exist_ok=True)
    save(NEW/'provenance/workspace_before.json',dict(source_root=str(OLD),destination=str(NEW),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=OLD,text=True),
        diff_stat=subprocess.check_output(['git','diff','--stat'],cwd=OLD,text=True),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=OLD)).hexdigest()))
    export_sources();common,registry=export_artifacts()
    save(NEW/'provenance/extractions.json',extractions)
    save(NEW/'provenance/migration_manifest.json',dict(schema=1,source=str(OLD),created='2026-11-12',
        files=records,models=len(registry),TEST_policy='Opaque byte copy and SHA only; zero TEST numeric parsing or inference',
        original_checkpoint_and_config_bytes_preserved=True,link_policy='Independent copies; no symlinks/hardlinks to original project'))
    print('EXPORTED',NEW,'models',len(registry),'files',len(records),'bytes',sum(x['bytes'] for x in records),flush=True)


if __name__=='__main__':main()

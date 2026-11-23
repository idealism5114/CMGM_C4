"""Phase N provenance, TRAIN-only fixed reference, and TEST read guard."""
import ast
import builtins
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from cmgm.models.phase_n_dynamic_graph import PhaseNModel, spatial_history
from cmgm.models.phase_m_prediction_moe import tensor_state_hash
from cmgm.scripts.phase_m_new_data import loader, inputs, evaluate
from cmgm.scripts.formal_main_innovation_ablation import ROOT, sha, save, seed_all

REFERENCE = ROOT/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381'
GROUPS = ('G0', 'G1', 'G2')
FILES = ['cmgm/graph/conditional_graph.py', 'cmgm/models/phase_n_dynamic_graph.py',
    'cmgm/scripts/phase_n_protocol.py', 'cmgm/scripts/phase_n_dynamic_graph.py',
    'cmgm/scripts/phase_n_graph_diagnostics.py', 'tests/test_phase_n_dynamic_graph.py', 'docs/phase_n_dynamic_graph.md']
DESIGN = dict(id='CMGM_PHASE_N_C4_CONDITIONAL_PRIOR_V1', seed=42, rank=8, gamma_init=.1,
    gamma='sigmoid(trainable logit); all parameters use original Adam weight decay',
    relation='tanh(Q@K.T/sqrt8), then row center over the fixed full node set BEFORE original availability mask',
    projections='Linear(64,8,bias=False), PyTorch default initialization',
    prior='log(A0.clamp_min(0)+1e-6) + gamma*centered_delta; passed directly to attention, no second log/A0 term',
    static_reference='SHA256 deterministic asset-ID codes, marketwise weighted mean/std matched to TRAIN pre-graph H from FRESH C4 initial state; buffer, never trainable',
    static_raw_prior='bitwise identical across samples; effective support may vary with original availability, as in G0',
    scale_check='TRAIN-only initial marketwise reference/H RMS ratio must be within [.5,2]; no VAL tuning. Trained scale drift is reported without repairs.',
    bootstrap=dict(block=40, repeats=2000, seed=42, unit='whole prediction dates, all commodities paired'),
    time_robustness='four chronologically contiguous equal-count VAL quarters; descriptive, fixed before results',
    judgement='Compare G1 with BOTH G0 and G2; report absolute/relative changes, paired block CIs and quarters separately. No significance/cross-seed claim; no new TEST.',
    interventions=['base_A0', 'TRAIN_mean_graph_prior_valid_edge_average', 'G2_full_static_prior',
        'TRAIN_mean_router', 'base_A0_AND_TRAIN_mean_router', 'native_graph_WITH_base_A0_router'],
    intervention_notes='G2 full prior includes its own trained A0 and mapping; distribution-shift diagnostic, not a pure conditioning-factor ablation',
    graph_statistics='valid edges; adjacent-date changes on common support, re-centered on that support; raw-logit and effective softmax differences kept distinct',
    TEST='FORBIDDEN: no TEST metrics, numerical arrays, dataset or inference; prompt historical numbers not used to choose design',
    preregistered_models=3, new_fits=2, reuse='G0 only if strict C4 provenance/data/selector/initialization audit passes')


def read(p):
    return json.loads(Path(p).read_text())


def hashes(cfg):
    return {p: sha(ROOT/p) for p in list(cfg['parent_source_hashes'])+FILES}


@contextmanager
def train_val_only_guard():
    """Reject TEST data/metrics paths at file-open boundaries, not after loading."""
    old_open, old_io = builtins.open, io.open
    hits = []
    def check(file):
        if isinstance(file, (str, bytes, Path)):
            p = Path(file.decode() if isinstance(file, bytes) else file)
            parts = [s.lower() for s in p.parts]
            name = p.name.lower()
            numeric = p.suffix.lower() in ('.npy', '.npz', '.json', '.csv', '.parquet', '.pt', '.pth')
            if numeric and (name.startswith('test_') or name == 'test_metrics.json'
                    or 'test_dataset' in parts or any(s.endswith('_test') for s in parts)):
                hits.append(str(p))
                raise RuntimeError('Phase N forbids TEST access: '+str(p))
    def guarded(file, *a, **kw):
        check(file); return old_open(file, *a, **kw)
    def guarded_io(file, *a, **kw):
        check(file); return old_io(file, *a, **kw)
    builtins.open, io.open = guarded, guarded_io
    try:
        yield hits
    finally:
        builtins.open, io.open = old_open, old_io


def validate_reference(reference):
    """Explicit whitelist: never glob prior run data or open TEST reports."""
    cfg = read(reference/'TRAINING_CONFIG.json')
    audit = read(reference/'DATA_FINGERPRINT_AUDIT.json')
    done = read(reference/'runs/C4/COMPLETE.json')
    expected = dict(seed=42, input_nodes=311, feature_dim=26, targets=23, window=20,
        batch_size=64, shuffle=False, drop_last=False, max_epochs=200, patience=10,
        lr=1e-4, weight_decay=1e-5, scheduler_factor=.5, scheduler_patience=5,
        selector='VAL_5D_COMMODITY_MACRO_MAE', epoch0_candidate=False)
    for k,v in expected.items():
        if cfg['protocol'][k] != v:
            raise ValueError('C4 protocol differs: '+k)
    if done['model_id'] != 'moe_phase_m/pred_global_dynamic' or done['control'] != 'C4' or done['status'] != 'COMPLETE':
        raise ValueError('Not the completed original C4')
    protected = {}
    for name,h in cfg['source_hashes'].items():
        if sha(ROOT/name) != h or sha(reference/'source_snapshot'/name) != h:
            raise ValueError('Original C4 source drift: '+name)
        protected[str(ROOT/name)] = h
    sealed = Path(cfg['history_root'])
    for name,record in read(sealed/'source_manifest.json').items():
        p = sealed/name.replace('cmgm/', 'cmgm_sep17_frozen/', 1)
        if sha(p) != record['namespaced_sha256']:
            raise ValueError('Sealed D0B source changed')
        protected[str(p)] = sha(p)
    data = Path(cfg['data_path'])
    if sha(data/'dataset_manifest.json') != audit['manifest_sha256']:
        raise ValueError('Frozen data manifest drift')
    for name,h in audit['array_hashes'].items():
        if name.lower().startswith('test_'):
            raise ValueError('Unexpected TEST array in TRAIN/VAL audit')
        p = data/name
        if sha(p) != h:
            raise ValueError('TRAIN/VAL array drift')
        protected[str(p)] = h
    assets = read(data/'asset_manifest.json')
    order = [a['asset_id'] for a in assets if a['market']=='future']
    if order != cfg['targets'] or [a['market'] for a in assets] != ['stock']*263+['bond']*25+['future']*23:
        raise ValueError('Asset/target order differs')
    for p in [reference/'TRAINING_CONFIG.json', reference/'PROVENANCE.json', reference/'initial_states.pt',
        reference/'DATA_FINGERPRINT_AUDIT.json', reference/'runs/C4/COMPLETE.json', reference/'runs/C4/metrics.json',
        reference/'runs/C4/history.json', reference/'runs/C4/train_predictions.npz', reference/'runs/C4/val_predictions.npz',
        Path(done['checkpoint']), Path(done['latest_checkpoint']), data/'asset_manifest.json', data/'dataset_manifest.json']:
        protected[str(p)] = sha(p)
    if sha(done['checkpoint']) != done['checkpoint_sha256']:
        raise ValueError('C4 best checkpoint hash differs')
    if sha(reference/'initial_states.pt') != read(reference/'PROVENANCE.json')['initial_states_sha256']:
        raise ValueError('Fresh initialization archive differs')
    history = read(reference/'runs/C4/history.json')
    best = min(history, key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
    cp = torch.load(done['checkpoint'], map_location='cpu', weights_only=False)
    if cp['control'] != 'C4' or cp['model_id'] != done['model_id'] or cp['epoch'] != best['epoch'] or cp['epoch'] != done['best_epoch']:
        raise ValueError('C4 best epoch identity mismatch')
    if cp['selection'] != expected['selector'] or cp['config_sha256'] != sha(reference/'TRAINING_CONFIG.json'):
        raise ValueError('C4 selection provenance differs')
    if cp['data_audit_sha256'] != sha(reference/'DATA_FINGERPRINT_AUDIT.json'):
        raise ValueError('C4 data audit identity differs')
    if not all(r['optimizer_steps']==43 and r['train_origins_used']==2726 for r in history):
        raise ValueError('C4 actual TRAIN coverage differs')
    return cfg, audit, done, assets, protected


def masked(batch, device='cpu'):
    b = inputs(batch, device)
    app = b['feature_applicable_mask']
    if app.ndim == 2: app = app[None,None]
    elif app.ndim == 3: app = app[:,None]
    fm = b['feature_valid_mask'] & app & b['input_valid_mask'][...,None]
    return torch.where(fm,b['X'],0.), b['input_valid_mask'] & fm.any(-1)


def identity_codes(assets):
    result = np.empty((len(assets),64), 'float64')
    for i,a in enumerate(assets):
        for d in range(64):
            code = hashlib.sha256(f'PhaseN|42|{a["asset_id"]}|{d}'.encode()).digest()
            result[i,d] = (int.from_bytes(code[:8],'big')+.5)/2**64*2-1
    return result


def match_reference_codes(assets, node_counts, sums, squared, observations):
    codes = identity_codes(assets)
    ref = np.zeros_like(codes); stats = {}
    for market in ('stock','bond','future'):
        ids = np.array([i for i,a in enumerate(assets) if a['market']==market])
        weights = np.asarray(node_counts[ids], 'float64')
        count = observations[market]
        if count <= 0 or weights.sum() <= 0:
            raise ValueError('Market has no valid TRAIN states: '+market)
        mean = sums[market]/count
        std = np.sqrt(np.maximum(squared[market]/count-mean**2,0))
        center = np.average(codes[ids],axis=0,weights=weights)
        scale = np.sqrt(np.average((codes[ids]-center)**2,axis=0,weights=weights))
        if (scale<=1e-12).any():raise ValueError('Degenerate deterministic identity code')
        ref[ids] = mean+(codes[ids]-center)/scale*std
        train_rms = float(np.sqrt(np.mean(squared[market]/count)))
        reference_rms = float(np.sqrt(np.average(ref[ids]**2,axis=0,weights=weights).mean()))
        ratio = reference_rms/train_rms if train_rms>0 else 1.
        if not .5 <= ratio <= 2:
            raise ValueError('Predefined TRAIN input scale gate failed')
        stats[market] = dict(TRAIN_valid_node_windows=int(count), mean=mean.tolist(), std=std.tolist(),
            TRAIN_H_RMS=train_rms, reference_RMS=reference_rms, RMS_ratio=ratio,
            zero_variance_channels=int((std<=1e-8).sum()))
    return ref.astype('float32'), stats


@torch.inference_mode()
def fit_reference(fresh_model, cfg, assets):
    fresh_model.eval()
    sums = {g:np.zeros(64) for g in ('stock','bond','future')}
    squares = {g:np.zeros(64) for g in sums}; counts = {g:0 for g in sums}
    nodes = np.zeros(len(assets),'int64')
    for batch in loader(cfg,'train'):
        x,v = masked(batch)
        h,nv,_ = spatial_history(fresh_model.encoder,x,v)
        a,valid = h.numpy().astype('float64'),nv.numpy()
        nodes += valid.sum(0)
        for market in sums:
            ids = [i for i,z in enumerate(assets) if z['market']==market]
            values = a[:,ids][valid[:,ids]]
            sums[market] += values.sum(0); squares[market] += (values**2).sum(0); counts[market] += len(values)
    ref,stats = match_reference_codes(assets,nodes,sums,squares,counts)
    return torch.from_numpy(ref), dict(markets=stats,node_valid_TRAIN_counts=nodes.tolist(),
        TRAIN_unobserved_assets=[assets[i]['asset_id'] for i,n in enumerate(nodes) if n==0],
        protocol='No labels/VAL/TEST fitting. FRESH C4 pre-graph states; identity pattern fixed by SHA256.',
        limitation='Matches marketwise marginal first/second moments, not node covariance or temporal variation. Buffer remains fixed while trained G1 features can drift.')


def model_for(group, out, cfg, device='cpu', load_initial=True):
    seed_all(42)
    ref = torch.from_numpy(np.load(out/'reference_code.npy',allow_pickle=False))
    m = PhaseNModel(group,cfg['history_root'],ref,**cfg['shape'])
    if load_initial:
        m.load_state_dict(torch.load(out/'initial_states.pt',map_location='cpu',weights_only=True)[group],strict=True)
    return m.to(device)

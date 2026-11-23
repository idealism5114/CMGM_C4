"""Deep V3 seed42 rerun with C4's VAL5 commodity-macro selection protocol.

Default: preflight only. --run fits the fixed eight models, freezes all best
checkpoints, then evaluates the previously authorized frozen TEST release.
The old Huber-selected suite and all original model defaults are read-only.
"""
import argparse
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cmgm.models.comparison_baselines import ORDER
from cmgm.models.triweave_baseline_adapter import prepared_model
from cmgm.models.phase_m_prediction_moe import tensor_state_hash
from cmgm.scripts.formal_main_innovation_ablation import (
    ROOT, sha, save, checkpoint, seed_all, state_rng, restore_rng, prediction_loss,
)
from cmgm.scripts.phase_m_analysis import population
from cmgm.scripts.phase_n_protocol import validate_reference, train_val_only_guard
from cmgm.scripts.triweave_baseline_benchmark import sources as old_sources, synthetic_preflight

SELECTOR = 'VAL_5D_COMMODITY_MACRO_MAE'
HORIZONS = (1, 5, 10, 20)
PROTOCOL_ID = 'CMGM_DEEP_V3_C4_VAL5_MACRO_SEED42_V1'
OLD_RUN = ROOT/'experiments/baseline_comparison_deep_v3_triweave/20261022_184258_600773'
C4_RUN = ROOT/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381'
C4_TEST = ROOT/'experiments/phase_m_horizon_prediction_moe_test/seed42_20261026_frozen_best'


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sources():
    files = ['cmgm/scripts/baselines_val5_macro.py',
             'tests/test_baselines_val5_macro.py', 'docs/baselines_val5_macro.md',
             'cmgm/scripts/phase_m_analysis.py', 'cmgm/scripts/phase_n_protocol.py']
    return {**old_sources(), **{p: sha(ROOT/p) for p in files}}


def csv_file(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


class NeutralSplit(Dataset):
    """Explicit array whitelist; TEST requires a verified all-model freeze."""
    def __init__(self, folder, split, *, test_authorized=False):
        if split not in ('train', 'val') and not (split == 'test' and test_authorized):
            raise ValueError('TRAIN/VAL only until all eight checkpoints are frozen')
        folder = Path(folder)
        names = ['features', 'reference_dates'] + [split+'_'+s for s in (
            'window_indices', 'origin_indices', 'targets', 'label_valid_mask')]
        self.a = {n: np.load(folder/(n+'.npy'), mmap_mode='r', allow_pickle=False) for n in names}
        self.split = split
        self.indices = self.a[split+'_window_indices']
        origins = self.a[split+'_origin_indices']
        expected = origins[:, None]+np.arange(-19, 1)
        # The frozen dataset encodes every pre-history slot as -1, not -19..-1.
        if not np.array_equal(np.maximum(self.indices, -1), np.maximum(expected, -1)):
            raise ValueError('Window/origin alignment differs')
        self.dates = self.a['reference_dates'][origins]
        if split in ('train', 'val') and str(self.a['reference_dates'][-1]) > '2024-12-31':
            raise ValueError('TRAIN/VAL timeline contains TEST dates')
        if len(np.unique(self.dates)) != len(self.dates):
            raise ValueError('Repeated origins')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        x = np.array(self.a['features'][np.maximum(idx, 0)], copy=True)
        x[idx < 0] = 0
        return dict(X=x, Y=np.array(self.a[self.split+'_targets'][k], copy=True),
                    label_valid_mask=np.array(self.a[self.split+'_label_valid_mask'][k], copy=True),
                    origin_date=str(self.dates[k]))


def loader(folder, split, *, test_authorized=False):
    return DataLoader(NeutralSplit(folder, split, test_authorized=test_authorized),
                      batch_size=64, shuffle=False, drop_last=False, num_workers=0)


def scored(prediction, target, mask):
    """The exact float64 evaluator used by C4; never use pooled MAE to select."""
    r = population(prediction, target, mask)
    if not np.isfinite(r['primary']):
        raise ValueError('Nonfinite VAL5 commodity Macro MAE')
    return r


@torch.inference_mode()
def evaluate(model, batches, device, targets):
    model.eval()
    ps, ys, ms, dates = [], [], [], []
    for b in batches:
        p = model(b['X'].to(device))
        if p.shape != b['Y'].shape or p.shape[1:] != (4, len(targets)) or not torch.isfinite(p).all():
            raise ValueError('Prediction shape/finite failure')
        ps.append(p.cpu().numpy()); ys.append(b['Y'].numpy())
        ms.append(b['label_valid_mask'].numpy()); dates.extend(b['origin_date'])
    if dates != list(batches.dataset.dates):
        raise ValueError('Origin coverage/order failure')
    a = dict(prediction=np.concatenate(ps), target=np.concatenate(ys),
             label_valid_mask=np.concatenate(ms), origin_dates=np.array(dates),
             commodity_order=np.array(targets))
    return scored(a['prediction'], a['target'], a['label_valid_mask']), a


def same_population(a, b):
    for k in ('target', 'label_valid_mask', 'origin_dates', 'commodity_order'):
        if not np.array_equal(a[k], b[k]):
            raise ValueError('Population/order mismatch: '+k)


def audit_cache(cache, manifest):
    for name, h in manifest['files'].items():
        if name.startswith('test_') or sha(Path(cache)/name) != h:
            raise ValueError('Unexpected/changed TRAIN/VAL cache: '+name)


def prepare(out, old_run, c4_run):
    out.mkdir(parents=True, exist_ok=False)
    workspace = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        diff_stat=subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True),
        diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git', 'diff'], cwd=ROOT)).hexdigest())
    save(out/'WORKSPACE_BEFORE.json', workspace)
    with train_val_only_guard() as hits:
        old = read(old_run/'config.json')
        if read(old_run/'status.json')['status'] != 'COMPLETE':
            raise ValueError('Original baseline suite not complete')
        # Old architectures, defaults and adapter must still match the sealed run.
        for p, h in old['source_hashes'].items():
            if sha(ROOT/p) != h:
                raise ValueError('Old baseline source changed: '+p)
        c4, audit, done, assets, protected = validate_reference(c4_run)
        if old['dataset_manifest_sha256'] != audit['manifest_sha256'] or old['targets'] != c4['targets']:
            raise ValueError('Baseline/C4 data or commodity order differs')
        cache = old_run/'neutral_train_val'
        cm = read(cache/'cache_manifest.json'); audit_cache(cache, cm)
        for name, h in cm['source_hashes'].items():
            if audit['array_hashes'].get(name) != h:
                raise ValueError('Neutral cache source is not C4 frozen data: '+name)
        for split in ('train', 'val'):
            ds = NeutralSplit(cache, split)
            if len(ds) != audit['splits'][split]['origins']:
                raise ValueError('Split population count changed')
            with np.load(c4_run/'runs/C4'/f'{split}_predictions.npz', allow_pickle=False) as a:
                for key, source in [('target', split+'_targets'), ('label_valid_mask', split+'_label_valid_mask')]:
                    if not np.array_equal(ds.a[source], a[key]):
                        raise ValueError('C4 labels/masks differ from baseline: '+key)
                if not np.array_equal(ds.dates, a['origin_dates']):
                    raise ValueError('C4 origins differ')
        # Fresh C4 replay under its original forward, with no TEST or new fitting.
        from cmgm.scripts.phase_m_new_data import model_for, evaluate as c4_eval, loader as c4_loader
        c4_model = model_for('C4', c4)
        cp = torch.load(done['checkpoint'], map_location='cpu', weights_only=False)
        c4_model.load_state_dict(cp['model_state_dict'], strict=True); c4_model.set_epoch(cp['epoch'])
        fresh, arr = c4_eval(c4_model, c4_loader(c4, 'val'), 'cpu', True)
        with np.load(c4_run/'runs/C4/val_predictions.npz', allow_pickle=False) as saved:
            diff = float(np.max(np.abs(arr['prediction']-saved['prediction'])))
            np.testing.assert_allclose(arr['prediction'], saved['prediction'], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(fresh['primary'], done['best_VAL5_macro_MAE'], atol=1e-9, rtol=1e-7)
        save(out/'C4_REPRODUCTION.json', dict(strict_load=True, prediction_max_abs_diff=diff,
             VAL5_macro_MAE=fresh['primary'], best_epoch=cp['epoch'], checkpoint_sha256=sha(done['checkpoint']), PASS=True))
        del c4_model, cp
        cfg = dict(protocol_id=PROTOCOL_ID, selector=SELECTOR, scheduler_metric=SELECTOR,
            early_stopping_metric=SELECTOR, seed=42, models=list(ORDER),
            train=dict(optimizer='Adam', lr=1e-4, weight_decay=1e-5, batch=64, shuffle=False,
                drop_last=False, max_epochs=200, patience=10, scheduler_factor=.5, scheduler_patience=5,
                prediction_loss='four-horizon masked Huber(.02) means SUM; no baseline auxiliary loss',
                gradient_clip=None, AMP=False, epoch0_candidate=False),
            old_baseline_run=str(old_run), c4_run=str(c4_run), c4_complete=done,
            dataset_manifest_sha256=audit['manifest_sha256'], data_audit=audit, targets=c4['targets'],
            train_val_cache=str(cache), train_val_cache_manifest=cm,
            checkpoint_dir=str(ROOT/'checkpoints/baselines_deep_v3_c4_val5'/out.name),
            source_hashes=sources(), created_at=now(), workspace=workspace,
            changed_vs_old_baselines=['checkpoint selector', 'scheduler monitor', 'early stopping monitor'],
            unchanged_vs_old_baselines=['architecture and adapter', 'data and masks', 'loss', 'seed/init order', 'optimizer/budget'],
            initialization='seed42 before constructing each original V3 model, as in old baseline run; no trained warm start',
            comparison_limits='Baselines use V3 27-token compressed view; C4 full311-node graph/native KL. This is a common selection protocol, not identical architecture/information capacity.',
            test_policy='All eight best checkpoints frozen before TEST access; reuse existing release, no new preprocessing/fitting. Historical TEST already viewed.',
            authorization='User requested all latest V3 baselines rerun using C4 VAL5 Macro MAE; historical baseline/TEST comparison continued; no extra seeds or tuning')
        initial, init_audit = {}, {}
        for name in ORDER:
            seed_all(42); m = prepared_model(name)
            initial[name] = {k:v.clone() for k,v in m.state_dict().items()}
            init_audit[name] = dict(state_sha256=tensor_state_hash(initial[name]),
                parameters=sum(p.numel() for p in m.parameters()), trained_checkpoint_loaded=False)
        checkpoint(out/'initial_states.pt', initial)
        cfg['initial_states_sha256'] = sha(out/'initial_states.pt')
        checks = synthetic_preflight()
        for name in ORDER:
            checks[name]['initial_state_sha256'] = init_audit[name]['state_sha256']
        for p in [old_run/'config.json', old_run/'models_frozen.json', old_run/'training_index.json',
                  cache/'cache_manifest.json']:
            protected[str(p)] = sha(p)
        for name, d in read(old_run/'training_index.json').items():
            for p in [Path(d['checkpoint']), old_run/'runs'/name/'training_history.json',
                      old_run/'runs'/name/'train_val_metrics.json']:
                protected[str(p)] = sha(p)
        cfg['protected'] = protected
        save(out/'config.json', cfg)
        save(out/'INITIALIZATION_AUDIT.json', init_audit)
        save(out/'preflight.json', dict(PASS=True, checks=checks, TEST_access_attempts=hits,
             TRAIN_origins=2726, VAL_origins=483, selector=SELECTOR))
        for p in cfg['source_hashes']:
            dest = out/'source_snapshot'/p
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/p, dest)
    save(out/'status.json', dict(status='PREPARED', complete=0, total=8, test_evaluated=False))
    (out/'PREREGISTERED_PROTOCOL.md').write_text(
        '# Deep V3 baseline selection repair\n\n'+json.dumps({k:v for k,v in cfg.items()
        if k in ('protocol_id','selector','scheduler_metric','early_stopping_metric','seed','models','train',
                 'changed_vs_old_baselines','unchanged_vs_old_baselines','comparison_limits','test_policy')}, indent=2)+'\n')
    print('PREPARED', out, flush=True)


def assert_inputs_unchanged(out, cfg):
    if cfg['source_hashes'] != sources() or sha(out/'initial_states.pt') != cfg['initial_states_sha256']:
        raise ValueError('Source or fresh initialization changed')
    for p, h in cfg['protected'].items():
        if sha(p) != h:
            raise ValueError('Historical source/data/result/checkpoint changed: '+p)
    audit_cache(cfg['train_val_cache'], cfg['train_val_cache_manifest'])


def train_one(name, out, cfg, device):
    folder = out/'runs'/name; folder.mkdir(parents=True, exist_ok=True)
    cpdir = Path(cfg['checkpoint_dir'])/name; cpdir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = cpdir/'best.pt', cpdir/'latest.pt'
    if (folder/'training_complete.json').exists():
        d = read(folder/'training_complete.json')
        if sha(d['checkpoint']) != d['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        return d
    seed_all(42); model = prepared_model(name).to(device)
    initial = torch.load(out/'initial_states.pt', map_location='cpu', weights_only=True)[name]
    if tensor_state_hash(model.state_dict()) != tensor_state_hash(initial):
        raise ValueError('Fresh shared model initialization differs')
    model.load_state_dict(initial, strict=True)
    # Keep old baseline RNG order, including construction; no new train-time RNG draws.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    batches = loader(cfg['train_val_cache'], 'train'); val_batches = loader(cfg['train_val_cache'], 'val')
    history=[]; best=float('inf'); best_epoch=None; stale=0; start_epoch=1; seconds=0.
    if latest_path.exists():
        cp = torch.load(latest_path, map_location='cpu', weights_only=False)
        if cp['config_sha256'] != sha(out/'config.json') or cp['name'] != name or cp['selection'] != SELECTOR:
            raise ValueError('Resume identity differs')
        model.load_state_dict(cp['model_state_dict'], strict=True); optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']); restore_rng(cp['rng'])
        history=cp['history']; best=cp['best_val']; best_epoch=cp['best_epoch']; stale=cp['stale']
        start_epoch=cp['epoch']+1; seconds=cp['seconds']
    for epoch in range(start_epoch, 201):
        if stale >= 10: break
        tic=time.perf_counter(); model.train(); losses=[]; origins=[]
        for b in batches:
            optimizer.zero_grad(set_to_none=True)
            p=model(b['X'].to(device))
            loss=prediction_loss(p, b['Y'].to(device), b['label_valid_mask'].to(device))
            if not torch.isfinite(p).all() or not torch.isfinite(loss): raise ValueError('Nonfinite training')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError('Nonfinite gradient')
            optimizer.step(); losses.append(float(loss.detach())); origins.extend(b['origin_date'])
        if origins != list(batches.dataset.dates): raise ValueError('TRAIN coverage/order differs')
        val, _ = evaluate(model, val_batches, device, cfg['targets'])
        score=val['primary']; scheduler.step(score)
        improved=score<best
        if improved: best=score; best_epoch=epoch; stale=0
        else: stale+=1
        elapsed=time.perf_counter()-tic; seconds+=elapsed
        history.append(dict(epoch=epoch, train_prediction_batch_mean=float(np.mean(losses)),
            VAL_5D_COMMODITY_MACRO_MAE=score, VAL_prediction_Huber_sum=val['four_horizon_prediction_Huber_sum'],
            VAL_macro_MAE={h:v['commodity_macro_MAE'] for h,v in val['horizons'].items()},
            optimizer_steps=len(losses), train_origins_used=len(origins), coverage=1.,
            LR=optimizer.param_groups[0]['lr'], seconds=elapsed))
        save(folder/'training_history.json', history)
        state=dict(name=name, epoch=epoch, best_epoch=best_epoch, best_val=best, stale=stale, seconds=seconds,
            model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=state_rng(), history=history, config_sha256=sha(out/'config.json'), selection=SELECTOR,
            data_manifest_sha256=cfg['dataset_manifest_sha256'], initial_state_sha256=tensor_state_hash(initial))
        if improved: checkpoint(best_path, state)
        checkpoint(latest_path, state)
        print(f'[{name}] epoch={epoch} VAL5_macro_MAE={score:.10g} best={best_epoch} seconds={elapsed:.1f}', flush=True)
    cp=torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    results={}
    for split in ('train','val'):
        r,a=evaluate(model, loader(cfg['train_val_cache'],split), device, cfg['targets'])
        np.savez_compressed(folder/(split+'_predictions.npz'), **a); results[split]=r
    np.testing.assert_allclose(results['val']['primary'], best, atol=1e-10, rtol=1e-7)
    save(folder/'train_val_metrics.json', results)
    done=dict(name=name, status='COMPLETE', selection=SELECTOR, best_epoch=cp['epoch'], last_epoch=history[-1]['epoch'],
        best_VAL5_macro_MAE=best, parameters=sum(v.numel() for v in model.parameters()),
        training_seconds=seconds, checkpoint=str(best_path), checkpoint_sha256=sha(best_path),
        latest_checkpoint=str(latest_path), strict_restore=True, test_evaluated=False)
    save(folder/'training_complete.json', done)
    del model,optimizer,scheduler,cp,initial; gc.collect(); torch.cuda.empty_cache()
    return done


def freeze_models(out, cfg, done):
    if set(done) != set(ORDER): raise ValueError('Cannot open TEST before all eight models finish')
    for name,d in done.items():
        hist=read(out/'runs'/name/'training_history.json')
        best=min(hist, key=lambda r:r[SELECTOR])
        if d['best_epoch'] != best['epoch'] or sha(d['checkpoint']) != d['checkpoint_sha256']:
            raise ValueError('Checkpoint selection/hash failed')
    frozen=dict(config_sha256=sha(out/'config.json'), selection=SELECTOR,
                dataset_manifest_sha256=cfg['dataset_manifest_sha256'], models=done)
    dest=out/'models_frozen.json'
    if dest.exists() and read(dest) != frozen: raise ValueError('Frozen model set changed')
    if not dest.exists(): save(dest,frozen)
    return sha(dest)


def evaluate_test(out, cfg, done, device):
    freeze_hash=freeze_models(out,cfg,done)
    source=Path(cfg['old_baseline_run'])/'neutral_test'
    manifest=read(source/'cache_manifest.json')
    for name,h in manifest['files'].items():
        if sha(source/name) != h: raise ValueError('Previously frozen TEST cache changed')
    c4_dir=C4_TEST/'runs/C4'
    c4_result=read(c4_dir/'result.json')
    if c4_result['checkpoint_sha256'] != cfg['c4_complete']['checkpoint_sha256']:
        raise ValueError('Wrong C4 TEST checkpoint')
    with np.load(c4_dir/'test_predictions.npz',allow_pickle=False) as ref:
        ref_arrays={k:ref[k] for k in ('prediction','target','label_valid_mask','origin_dates','commodity_order')}
    c4_metrics=scored(ref_arrays['prediction'],ref_arrays['target'],ref_arrays['label_valid_mask'])
    np.testing.assert_allclose(c4_metrics['primary'],c4_result['metrics']['primary'],atol=1e-15,rtol=0)
    results={}
    for name in ORDER:
        folder=out/'runs'/name
        if (folder/'test_metrics.json').exists():
            results[name]=read(folder/'test_metrics.json'); continue
        save(out/'status.json',dict(status='TEST_EVALUATION',current=name,complete=len(results),total=8))
        if sha(out/'models_frozen.json') != freeze_hash: raise ValueError('Frozen model set changed')
        seed_all(42); model=prepared_model(name).to(device)
        cp=torch.load(done[name]['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model_state_dict'],strict=True)
        r,a=evaluate(model,loader(source,'test',test_authorized=True),device,cfg['targets'])
        same_population(a,ref_arrays)
        r['primary_name']='TEST_5D_COMMODITY_MACRO_MAE'
        r['selection_role']='reporting only; all eight checkpoints frozen before any TEST access'
        np.savez_compressed(folder/'test_predictions.npz',**a); save(folder/'test_metrics.json',r)
        results[name]=r
        print('[TEST COMPLETE]',name,'Macro MAE@5d',r['primary'],flush=True)
        del model,cp; gc.collect(); torch.cuda.empty_cache()
    save(out/'TEST_RELEASE_AUDIT.json',dict(frozen_models_sha256=freeze_hash,
        source=str(source),manifest_sha256=sha(source/'cache_manifest.json'),
        common_C4_population=True,raw_data_read=False,scaler_refitted=False,models_evaluated=8,
        C4_checkpoint_unchanged=sha(cfg['c4_complete']['checkpoint'])==cfg['c4_complete']['checkpoint_sha256']))
    save(out/'test_metrics.json',results)
    return results,c4_metrics


def summarize(out,cfg,done,tests,c4_test):
    val_c4=read(Path(cfg['c4_run'])/'runs/C4/metrics.json')['val']
    vals={n:read(out/'runs'/n/'train_val_metrics.json')['val'] for n in ORDER}
    tables={}
    for split, rs, ref in [('VAL',vals,val_c4),('TEST',tests,c4_test)]:
        combined={**rs,'C4':ref}; rows=[]; multi=[]; per=[]
        for rank,name in enumerate(sorted(combined,key=lambda n:combined[n]['primary']),1):
            r=combined[name]; d=cfg['c4_complete'] if name=='C4' else done[name]
            params=d['parameters']['total'] if isinstance(d['parameters'],dict) else d['parameters']
            rows.append(dict(Rank=rank,Model=name,**{f'{split}_{h}d_Macro_MAE':r['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS},
                Delta5d_vs_C4=r['primary']-ref['primary'],RMSE5=r['horizons']['5']['RMSE'],
                Hit5_percent=100*r['horizons']['5']['Hit'],Parameters=params,Best_Epoch=d['best_epoch']))
            multi.extend(dict(Model=name,horizon=int(h),**v) for h,v in r['horizons'].items())
            per.extend(dict(Model=name,Commodity=cfg['targets'][v['commodity_index']],**v) for v in r['per_commodity'])
        csv_file(out/f'{split}_5D_RANKING.csv',rows); csv_file(out/f'{split}_ALL_HORIZONS.csv',multi)
        csv_file(out/f'{split}_PER_COMMODITY.csv',per); tables[split]=rows
    all_history=[dict(Model=n,**r) for n in ORDER for r in read(out/'runs'/n/'training_history.json')]
    csv_file(out/'TRAINING_HISTORY.csv',all_history)
    changes=[]
    old=Path(cfg['old_baseline_run'])
    for name in ORDER:
        previous=read(old/'runs'/name/'training_complete.json')
        old_test=read(old/'test_metrics.json')[name]
        changes.append(dict(Model=name,Old_Best_Epoch=previous['best_epoch'],New_Best_Epoch=done[name]['best_epoch'],
            Old_TEST5_Macro_MAE=old_test['horizons']['5']['commodity_macro_MAE'],
            New_TEST5_Macro_MAE=tests[name]['primary'],
            Delta_NEW_minus_OLD=tests[name]['primary']-old_test['horizons']['5']['commodity_macro_MAE'],
            Old_Selector='four-horizon VAL Huber',New_Selector=SELECTOR))
    csv_file(out/'OLD_VS_NEW_SELECTOR.csv',changes)
    lines=['# Deep V3 baselines: C4-matched VAL5 Macro MAE selection','',
        '8/8 seed42 models retrained from scratch. Checkpoint, scheduler and early stopping all monitor VAL5 commodity Macro MAE.',
        'Training remains four-horizon masked Huber(.02); all old architectures, adapter, optimizer, batch/order and 200-epoch cap retained. Epoch0 is not a candidate, matching original C4.',
        'C4 is the unchanged Phase M global dynamic prediction MoE, reused after strict checkpoint and VAL replay audits. No trained checkpoint initialized any new baseline.','']
    for split in ('VAL','TEST'):
        lines += [f'## {split}', '', '| Rank | Model | 1d | 5d | 10d | 20d | Delta5 vs C4 | Best epoch |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|']
        for r in tables[split]:
            values=' | '.join(f'{r[f"{split}_{h}d_Macro_MAE"]:.10f}' for h in HORIZONS)
            lines.append(f'| {r["Rank"]} | {r["Model"]} | {values} | {r["Delta5d_vs_C4"]:+.10f} | {r["Best_Epoch"]} |')
        lines.append('')
    lines += ['## Interpretation boundaries','',
        'The selection mismatch is repaired. The V3 27-token compressed input view, fixed-slot core masking limits, architecture capacities and absence of native Switch KL still differ from C4; this is not a fully identical information representation control.',
        'Changing the scheduler/early-stop monitor can alter the training trajectory as well as the selected epoch. New results cannot be obtained by merely choosing a different old checkpoint.',
        'No model/seed/structure was changed after results. TEST was evaluated only after all eight checkpoint identities were frozen. The project has repeatedly viewed TEST; these are exploratory historical TEST results, not a blind confirmation set.',
        'Complete per-commodity metrics, all four horizons, prediction/target/mask/date caches, source/init hashes and best/latest optimizer/scheduler/RNG states are indexed in file_index.json.',
        'GraphWaveNet retains its pre-existing unused final graph projection (3104 parameters); total/effective-gradient counts are in preflight.json. No baseline architecture was altered to fix this during the selector rerun.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def verify(out,cfg,done):
    checks={}
    for name,d in done.items():
        folder=out/'runs'/name
        history=read(folder/'training_history.json'); best=min(history,key=lambda r:r[SELECTOR])
        if best['epoch']!=d['best_epoch']: raise ValueError('Best epoch audit failed')
        # Replay scheduler independently from the recorded validation sequence.
        dummy=torch.nn.Parameter(torch.zeros(()))
        opt=torch.optim.Adam([dummy],lr=1e-4,weight_decay=1e-5)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
        for r in history:
            sch.step(r[SELECTOR])
            if opt.param_groups[0]['lr']!=r['LR'] or r['optimizer_steps']!=43 or r['train_origins_used']!=2726:
                raise ValueError('Scheduler/coverage audit failed')
        m=prepared_model(name)
        for path in (d['checkpoint'],d['latest_checkpoint']):
            cp=torch.load(path,map_location='cpu',weights_only=False); m.load_state_dict(cp['model_state_dict'],strict=True)
            if cp['selection']!=SELECTOR or cp['config_sha256']!=sha(out/'config.json'):
                raise ValueError('Checkpoint strict identity audit failed')
            if len(cp['optimizer']['param_groups'][0]['params'])!=len(list(m.parameters())):
                raise ValueError('Optimizer coverage differs')
            if cp['scheduler']['last_epoch']!=cp['epoch'] or not {'python','numpy','torch','cuda'}<=set(cp['rng']):
                raise ValueError('Scheduler/RNG state incomplete')
        recomputed={}
        for split in ('train','val','test'):
            with np.load(folder/(split+'_predictions.npz'),allow_pickle=False) as a:
                # Explicit float64 per-commodity independent implementation of primary MAE.
                errors=np.abs(a['prediction'][:,1].astype('float64')-a['target'][:,1].astype('float64'))
                mask=a['label_valid_mask'][:,1]
                macro=float(np.mean([errors[:,c][mask[:,c]].mean() for c in range(23)]))
                r=scored(a['prediction'],a['target'],a['label_valid_mask'])
                np.testing.assert_allclose(r['primary'],macro,atol=1e-15,rtol=0)
            saved=read(folder/'test_metrics.json') if split=='test' else read(folder/'train_val_metrics.json')[split]
            np.testing.assert_allclose(macro,saved['primary'],atol=1e-15,rtol=0)
            recomputed[split]=macro
        if sha(d['checkpoint'])!=d['checkpoint_sha256']: raise ValueError('Checkpoint changed')
        checks[name]=dict(PASS=True,strict_best_latest=True,best_epoch=d['best_epoch'],
            last_epoch=history[-1]['epoch'],full_TRAIN_coverage=True,epochs=len(history),
            scheduler_replayed=True,selection=SELECTOR,metric_recomputation=recomputed)
    assert_inputs_unchanged(out,cfg)
    diff=__import__('hashlib').sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest()
    if diff!=cfg['workspace']['diff_sha256']: raise ValueError('Existing user tracked modifications changed')
    save(out/'FINAL_VERIFICATION.json',dict(status='PASS',models=checks,baselines_complete=8,
        historical_sources_results_checkpoints_unchanged=True,existing_tracked_user_diff_unchanged=True,
        all_models_frozen_before_TEST=True,all_three_selection_monitors=SELECTOR,
        total_epochs=sum(v['epochs'] for v in checks.values()),
        total_training_seconds=sum(d['training_seconds'] for d in done.values())))


def run(out,device):
    cfg=read(out/'config.json')
    if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent fallback')
    if not read(out/'preflight.json')['PASS']: raise ValueError('Preflight failed')
    done={}
    with train_val_only_guard() as hits:
        assert_inputs_unchanged(out,cfg)
        save(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            device=str(device),device_name=torch.cuda.get_device_name(device),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES')))
        for name in ORDER:
            save(out/'status.json',dict(status='TRAINING',current=name,complete=len(done),total=8,test_evaluated=False))
            done[name]=train_one(name,out,cfg,device); save(out/'training_index.json',done)
        freeze_models(out,cfg,done)
        save(out/'TRAIN_VAL_ISOLATION.json',dict(PASS=True,TEST_access_attempts=hits,models_frozen_before_exit=True))
    tests,c4_test=evaluate_test(out,cfg,done,device)
    summarize(out,cfg,done,tests,c4_test); verify(out,cfg,done)
    save(out/'PROVENANCE.json',dict(config_sha256=sha(out/'config.json'),source_hashes=cfg['source_hashes'],
        initialization_archive_sha256=cfg['initial_states_sha256'],data_manifest_sha256=cfg['dataset_manifest_sha256'],
        models_frozen_sha256=sha(out/'models_frozen.json'),C4_checkpoint=cfg['c4_complete'],
        old_baseline_results_unchanged=True,completed_at=now()))
    save(out/'file_index.json',dict(output=str(out),report=str(out/'REPORT.md'),
        VAL_ranking=str(out/'VAL_5D_RANKING.csv'),TEST_ranking=str(out/'TEST_5D_RANKING.csv'),
        models=done,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
                          if p.is_file() and p.suffix in ('.json','.csv','.md','.npz','.pt') and 'source_snapshot' not in p.parts}))
    save(out/'status.json',dict(status='COMPLETE',complete=8,total=8,test_evaluated=True,verification='PASS'))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-run',type=Path,default=OLD_RUN)
    p.add_argument('--c4-run',type=Path,default=C4_RUN)
    p.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); torch.set_num_threads(4)
    if not a.resume: prepare(a.output.resolve(),a.old_run.resolve(),a.c4_run.resolve())
    with (a.output/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.run: run(a.output.resolve(),torch.device(a.device))
        except Exception:
            save(a.output/'failure.json',dict(traceback=traceback.format_exc(),time=now())); raise


if __name__=='__main__':
    main()

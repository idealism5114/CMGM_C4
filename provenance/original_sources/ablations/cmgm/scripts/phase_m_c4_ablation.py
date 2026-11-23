"""Matched seed42 C4 mechanism ablations; default preflight, explicit --run.

Full C4 and no-MoE=C0 are reused, provenance checked. Eleven fresh fits follow
the frozen Phase M training/VAL5 selector, then all best identities freeze before
TEST. Historical runs/checkpoints and original model defaults are never edited.
"""
import argparse
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from cmgm.models.phase_m_c4_ablation import C4Ablation, KEYS, LABELS, DEFS
from cmgm.models.phase_m_prediction_moe import tensor_state_hash
from cmgm.scripts.phase_m_new_data import loader, inputs, evaluate, PROTOCOL as PARENT_PROTOCOL
from cmgm.scripts.phase_m_test_evaluation import (
    inventory, TestReleaseDataset, infer, test_metrics,
)
from cmgm.scripts.formal_main_innovation_ablation import (
    ROOT, sha, save, checkpoint, seed_all, state_rng, restore_rng, prediction_loss,
)
from cmgm.scripts.phase_m_horizon_prediction_moe import csv_file

REFERENCE = ROOT/'experiments/phase_m_horizon_prediction_moe/newdata_20261026_135222_918381'
TEST_REFERENCE = ROOT/'experiments/phase_m_horizon_prediction_moe_test/seed42_20261026_frozen_best'
REUSE = dict(full_c4='C4', no_moe='C0')
HORIZONS = (1, 5, 10, 20)
NEW_FILES = ['cmgm/models/phase_m_c4_ablation.py', 'cmgm/scripts/phase_m_c4_ablation.py',
             'tests/test_phase_m_c4_ablation.py', 'docs/phase_m_c4_ablation.md']


def read(p):
    return json.loads(Path(p).read_text())


def source_hashes(cfg):
    paths = list(cfg['parent_source_hashes']) + NEW_FILES
    return {p: sha(ROOT/p) for p in paths}


def model_for(key, out, cfg, device='cpu'):
    seed_all(42)
    m = C4Ablation(key, cfg['history_root'], **cfg['shape'])
    initial = torch.load(out/'initial_states.pt', map_location='cpu', weights_only=True)[key]
    m.load_state_dict(initial, strict=True)
    m.freeze_inactive(cfg['inactive_parameters'][key])
    return m.to(device)


def prepare(out, reference, test_reference):
    out.mkdir(parents=True, exist_ok=False)
    save(out/'WORKSPACE_BEFORE.json', dict(
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        diff_stat=subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True)))
    parent, rows, protected = inventory(reference)
    test_index = read(test_reference/'EVALUATION_MANIFEST.json')
    if read(test_reference/'STATUS.json')['status'] != 'COMPLETE':
        raise ValueError('Original frozen TEST evaluation incomplete')
    for key in REUSE.values():
        if test_index['models'][key]['checkpoint_sha256'] != rows[key]['checkpoint_sha256']:
            raise ValueError('Reused TEST checkpoint identity differs')
    for p in test_reference.rglob('*'):
        if p.is_file():
            protected[str(p)] = sha(p)
    initial = torch.load(reference/'initial_states.pt', map_location='cpu', weights_only=True)
    first = next(iter(loader(parent, 'train')))
    b = {k: v[:2] for k, v in first.items() if torch.is_tensor(v)}
    states, audits, counts, inactive = {}, {}, [], {}
    for key in KEYS:
        seed_all(42)
        source = 'C0' if key == 'no_moe' else 'C4'
        m = C4Ablation(key, parent['history_root'], initial_state=initial[source], **parent['shape']).eval()
        shared = {n: float((v-m.state_dict()[n]).abs().max()) for n, v in initial[source].items()
                  if not torch.equal(v, m.state_dict()[n])}
        if shared:
            raise ValueError('Shared initialization drift: '+str(shared))
        states[key] = {n: v.clone() for n, v in m.state_dict().items()}
        p = m(**inputs(b, 'cpu'))
        loss = prediction_loss(p, b['Y'], b['label_valid_mask']) + m.switch_loss()
        loss.backward()
        unused = [n for n, v in m.named_parameters() if v.grad is None]
        m.freeze_inactive(unused)
        if not torch.isfinite(p).all() or not torch.isfinite(loss):
            raise ValueError('Nonfinite preflight')
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):
            raise ValueError('Nonfinite gradients')
        inactive[key] = unused
        audits[key] = dict(max_abs_diff=0., mismatch_count=0, PASS=True,
            initialization_source=source+' fresh INITIAL state, not trained best checkpoint',
            shared_keys=list(initial[source]), inactive_frozen_parameters=unused,
            state_sha256=tensor_state_hash(states[key]))
        counts.append(dict(Key=key, Variant=LABELS[key],
            Total_registered=sum(v.numel() for v in m.parameters()),
            Trainable=sum(v.numel() for v in m.parameters() if v.requires_grad),
            Inactive_frozen=sum(v.numel() for v in m.parameters() if not v.requires_grad),
            Experts=sum(v.numel() for v in m.experts.parameters()) if hasattr(m, 'experts') else 0,
            Router=sum(v.numel() for v in m.router.parameters()) if hasattr(m, 'router') else 0))
        del m
    checkpoint(out/'initial_states.pt', states)
    save(out/'INITIALIZATION_AUDIT.json', audits)
    csv_file(out/'PARAMETER_COUNTS.csv', counts)
    protocol = dict(PARENT_PROTOCOL)
    protocol.update(id='CMGM_C4_MECHANISM_ABLATION_NEW23_SEED42_V1', configurations=list(KEYS),
        TEST_policy='User authorized full C4 ablations and TEST; all checkpoints freeze before TEST; no TEST selection',
        bootstrap=dict(run=False, reason='Not requested for this ablation task'),
        reuse=REUSE, additional_training_runs=11,
        former_adaptive_fusion='User approved replacement with uniform four-expert routing; gate ablation not applicable to C4',
        status='Exploratory post-hoc study; C4 selected for follow-up after observing prior TEST results',
        inactive_parameters='Structurally unused whole tensors frozen after synthetic/actual TRAIN preflight; optimizer excludes them. Partial masked router coordinates may have zero gradients.')
    cfg = dict(protocol=protocol, reference=str(reference), test_reference=str(test_reference),
        history_root=parent['history_root'], data_path=parent['data_path'], shape=parent['shape'], targets=parent['targets'],
        parent_source_hashes=parent['source_hashes'], checkpoint_root=str(ROOT/'checkpoints/phase_m_c4_ablation'/out.name),
        inactive_parameters=inactive, initial_states_sha256=sha(out/'initial_states.pt'),
        data_audit_sha256=sha(reference/'DATA_FINGERPRINT_AUDIT.json'), counts={r['Key']: r for r in counts},
        definitions=DEFS, test_release_path=test_index['test_release_path'],
        test_release_manifest_sha256=test_index['test_manifest_sha256'],
        TEST_device='CPU float32, threads2, matching the reused Phase M TEST evaluation',
        user_scope='13 rows: Full C4 + 10 backbone ablations + uniform routing + no MoE; reuse C4/C0, train other11 once seed42')
    cfg['source_hashes'] = source_hashes(cfg)
    for key in REUSE:
        source = REUSE[key]
        cfg.setdefault('reused_runs', {})[key] = dict(**rows[source],
            original_complete=str(reference/'runs'/source/'COMPLETE.json'),
            original_test_result=str(test_reference/'runs'/source/'result.json'))
    save(out/'config.json', cfg)
    save(out/'PROTECTED_FILES.json', protected)
    save(out/'DATA_FINGERPRINT_AUDIT.json', read(reference/'DATA_FINGERPRINT_AUDIT.json'))
    for name in cfg['source_hashes']:
        p = out/'source_snapshot'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, p)
    (out/'PREREGISTERED_PROTOCOL.md').write_text('# C4 ablations, scope frozen before fits\n\n'
        +json.dumps(protocol, ensure_ascii=False, indent=2)+'\n\nDefinitions:\n'
        +'\n'.join(f'- {k}: {DEFS[k]}' for k in KEYS)+'\n')
    save(out/'AUDIT_RESULTS.json', dict(status='PREPARED', shared_initialization='PASS',
        data_identity='PASS', additional_training='NOT_RUN', test_evaluated=False))
    save(out/'STATUS.json', dict(status='PREPARED', new_models_complete=0, new_total=11, reused=2, test_evaluated=False))
    print('PREPARED', out, flush=True)


def reused(key, out, cfg):
    original = cfg['reused_runs'][key]
    d = read(original['original_complete'])
    if sha(d['checkpoint']) != d['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    m = model_for(key, out, cfg)
    cp = torch.load(d['checkpoint'], map_location='cpu', weights_only=False)
    m.load_state_dict(cp['model_state_dict'], strict=True)
    m.set_epoch(cp['epoch'])
    source = Path(original['original_complete']).parent
    folder = out/'runs'/key
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('history.json', 'metrics.json', 'train_predictions.npz', 'val_predictions.npz'):
        shutil.copyfile(source/name, folder/name)
    result = dict(key=key, label=LABELS[key], status='COMPLETE', best_epoch=d['best_epoch'],
        best_VAL5_macro_MAE=d['best_VAL5_macro_MAE'], checkpoint=d['checkpoint'],
        checkpoint_sha256=d['checkpoint_sha256'], training_seconds=d['training_seconds'],
        reused=True, original_control=REUSE[key], source_run=str(source), strict_load=True,
        parameters=cfg['counts'][key], test_evaluated=False)
    save(folder/'COMPLETE.json', result)
    return result


def train_one(key, out, cfg, device):
    folder = out/'runs'/key
    if (folder/'COMPLETE.json').exists():
        d = read(folder/'COMPLETE.json')
        if sha(d['checkpoint']) != d['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        return d
    if key in REUSE:
        return reused(key, out, cfg)
    folder.mkdir(parents=True, exist_ok=True)
    cpdir = Path(cfg['checkpoint_root'])/key
    cpdir.mkdir(parents=True, exist_ok=True)
    m = model_for(key, out, cfg, device)
    seed_all(42)  # Same post-construction policy as the original Phase M fits.
    optimizer = torch.optim.Adam([v for v in m.parameters() if v.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    train, val = loader(cfg, 'train'), loader(cfg, 'val')
    best, best_epoch, stale, seconds, history, start_epoch = float('inf'), None, 0, 0., [], 1
    latest, best_path = cpdir/'latest.pt', cpdir/'best.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=False)
        if cp['key'] != key or cp['config_sha256'] != sha(out/'config.json'):
            raise ValueError('Resume identity mismatch')
        m.load_state_dict(cp['model_state_dict'], strict=True)
        optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        m.set_epoch(cp['epoch']); restore_rng(cp['rng'])
        best, best_epoch, stale = cp['best'], cp['best_epoch'], cp['stale']
        seconds, history, start_epoch = cp['seconds'], cp['history'], cp['epoch']+1
    for epoch in range(start_epoch, 201):
        if stale >= 10:
            break
        tic = time.perf_counter()
        m.train(); m.set_epoch(epoch)
        losses, kls, used = [], [], []
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            p = m(**inputs(batch, device))
            pred = prediction_loss(p, batch['Y'].to(device), batch['label_valid_mask'].to(device))
            kl = m.switch_loss()
            loss = pred+kl
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            losses.append(float(pred.detach())); kls.append(float(kl.detach())); used += batch['origin_date']
        if len(used) != 2726 or len(set(used)) != 2726:
            raise ValueError('Training coverage mismatch')
        r = evaluate(m, val, device)
        score = r['primary']
        scheduler.step(score)
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        elapsed = time.perf_counter()-tic
        seconds += elapsed
        history.append(dict(key=key, epoch=epoch, train_prediction_Huber_batch_mean=float(np.mean(losses)),
            train_KL_batch_mean=float(np.mean(kls)), VAL5_macro_MAE=score,
            VAL_MAE={h: v['commodity_macro_MAE'] for h, v in r['horizons'].items()},
            VAL_Huber=r['four_horizon_prediction_Huber_sum'], optimizer_steps=len(losses),
            train_origins_used=len(used), lr=optimizer.param_groups[0]['lr'], seconds=elapsed,
            switch_beta=m.switching_latent_transformer.regime_filter.current_beta))
        save(folder/'history.json', history)
        state = dict(key=key, epoch=epoch, model_state_dict=m.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), rng=state_rng(), best=best, best_epoch=best_epoch, stale=stale,
            seconds=seconds, history=history, selection='VAL_5D_COMMODITY_MACRO_MAE',
            config_sha256=sha(out/'config.json'), data_audit_sha256=cfg['data_audit_sha256'])
        if improved:
            checkpoint(best_path, state)
        checkpoint(latest, state)
        print(f'[{key}] epoch={epoch} VAL5_macro={score:.10f} best_epoch={best_epoch} stale={stale} seconds={elapsed:.1f}', flush=True)
    cp = torch.load(best_path, map_location='cpu', weights_only=False)
    m.load_state_dict(cp['model_state_dict'], strict=True); m.set_epoch(cp['epoch'])
    metrics = {}
    for split in ('train', 'val'):
        r, a = evaluate(m, loader(cfg, split), device, True)
        a['commodity_order'] = np.array(cfg['targets'])
        np.savez_compressed(folder/(split+'_predictions.npz'), **a)
        metrics[split] = r
    if not np.isclose(metrics['val']['primary'], best, atol=1e-10, rtol=1e-7):
        raise ValueError('Selected checkpoint VAL mismatch')
    save(folder/'metrics.json', metrics)
    d = dict(key=key, label=LABELS[key], status='COMPLETE', best_epoch=cp['epoch'],
        best_VAL5_macro_MAE=best, checkpoint=str(best_path), checkpoint_sha256=sha(best_path),
        latest_checkpoint=str(latest), training_seconds=seconds, parameters=cfg['counts'][key],
        reused=False, strict_load=True, test_evaluated=False)
    save(folder/'COMPLETE.json', d)
    del m, optimizer, scheduler, cp
    gc.collect(); torch.cuda.empty_cache()
    return d


def summarize(out, cfg, done, test):
    rows, multi, val_rows = [], [], []
    full = test['full_c4']['metrics']['primary'] if 'full_c4' in test else None
    for key in KEYS:
        if key not in done:
            continue
        v = read(out/'runs'/key/'metrics.json')['val']
        val_rows.append(dict(Key=key, Variant=LABELS[key], VAL5_macro_MAE=v['primary'],
            **{f'VAL_MAE_{h}d': v['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS},
            best_epoch=done[key]['best_epoch'], trainable_params=cfg['counts'][key]['Trainable']))
        if key not in test:
            continue
        r = test[key]['metrics']; a = r['horizons']['5']
        delta = r['primary']-full if full is not None else None
        rows.append(dict(Key=key, Variant=LABELS[key], TEST_MAE_5d=r['primary'],
            Delta_MAE=delta, Relative_Delta_percent=100*delta/full if full is not None else None,
            TEST_MSE_5d=a['MSE'], TEST_RMSE_5d=a['RMSE'], TEST_Hit_5d_percent=100*a['Hit'],
            **{f'TEST_MAE_{h}d': r['horizons'][str(h)]['commodity_macro_MAE'] for h in HORIZONS},
            best_epoch=done[key]['best_epoch'], Trainable_params=cfg['counts'][key]['Trainable'], Reused=key in REUSE))
        multi += [dict(Key=key, Variant=LABELS[key], Horizon=int(h), **a) for h, a in r['horizons'].items()]
    # Before TEST release, do not fabricate empty/fill-zero result tables.
    if rows:
        csv_file(out/'main_TEST5_ablation_table.csv', rows)
    if multi:
        csv_file(out/'multi_horizon_TEST_ablation.csv', multi)
    csv_file(out/'VAL_ablation_table.csv', val_rows)
    lines = ['# C4 Global Dynamic Prediction MoE — 正式机制消融（探索性）', '',
        f'完成训练/合法复用 {len(done)}/13，完成 TEST {len(test)}/13。Full C4 与 no-MoE D0B 来自本轮 Phase M 匹配协议，未重复训练。',
        '所有 Δ=消融−Full C4；正值表示移除后误差变大。主口径为23商品等权Macro MAE，收益率小数。',
        'TEST423 origins：2025-01-02—2026-09-29。训练只用TRAIN/VAL；所有最佳checkpoint锁定后统一TEST。', '',
        '| Variant | TEST MAE@5d | ΔMAE | Relative Δ% | RMSE@5d | Hit% | best epoch |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        dd = f"{r['Delta_MAE']:+.10f}" if r['Delta_MAE'] is not None else 'PENDING'
        rr = f"{r['Relative_Delta_percent']:+.4f}" if r['Relative_Delta_percent'] is not None else 'PENDING'
        lines.append(f"| {r['Variant']} | {r['TEST_MAE_5d']:.10f} | {dd} | {rr} | {r['TEST_RMSE_5d']:.10f} | {r['TEST_Hit_5d_percent']:.3f} | {r['best_epoch']} |")
    lines += ['', '## Definitions and interpretation', '',
        '原w/o Adaptive Fusion针对D0B gate，不适用C4，经用户明确确认换成四专家固定等权。no-MoE恢复原D0B gate，恰为已完成C0。',
        '移除空间分支保留近期/长期两个专家，禁用空间/交互专家并在剩余候选归一化。移除时间分支仅剩空间专家，不调用时间编码器或KL。',
        'w/o Microstate沿用原定义：仅归一化后最终micro读出为零，Z递推和近期Z专家仍保留；它不是取消所有latent信息。',
        '不可用模块保留state_dict以核验共享初始化，但全量无梯度的参数冻结并排除optimizer；详见INITIALIZATION_AUDIT与PARAMETER_COUNTS。',
        '全部新训练seed42、原四周期masked Huber(.02)+native KL一次；Adam1e-4、WD1e-5、batch64、max200、patience10、scheduler.5/patience5、shuffle=False、drop_last=False。',
        'Checkpoint、scheduler、earlystop均按VAL5 Macro MAE，非旧9月17日Huber选模。epoch0不是候选。',
        'C4是在先前TEST观察后选定用于本研究，故该消融为post-hoc探索性；不能视为盲测、统计显著或严格经济因果识别。',
        '历史PIT/发布时间/连续期货换月仍未认证。未更改旧模型默认值、数据、报告或checkpoint；未增加其他seed/搜索。', '',
        '逐项机制定义：', *[f'- {LABELS[k]}: {DEFS[k]}' for k in KEYS]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'file_index.json', dict(output=str(out), models=done,
        results={k: dict(metrics=str(out/'runs'/k/'TEST_result.json'), predictions=str(out/'runs'/k/'test_predictions.npz')) for k in test},
        reports={p.name: str(p) for p in out.iterdir() if p.suffix in ('.json', '.csv', '.md', '.txt')}))


def run(out, device):
    cfg = read(out/'config.json')
    if cfg['source_hashes'] != source_hashes(cfg) or sha(out/'initial_states.pt') != cfg['initial_states_sha256']:
        raise ValueError('Source or random initialization drift')
    for n, digest in read(out/'DATA_FINGERPRINT_AUDIT.json')['array_hashes'].items():
        if sha(Path(cfg['data_path'])/n) != digest:
            raise ValueError('Training data changed')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('GPU required; no silent CPU training fallback')
    save(out/'ENVIRONMENT.json', dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(device), physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark))
    done = {}
    for key in KEYS:
        save(out/'STATUS.json', dict(status='TRAINING', current=key, rows_complete=len(done), total=13,
            new_complete=sum(k not in REUSE for k in done), new_total=11, test_evaluated=False))
        done[key] = train_one(key, out, cfg, device)
        save(out/'training_index.json', done)
        summarize(out, cfg, done, {})
    # Strictly verify original selection before releasing TEST.
    audit = {}
    for key, row in done.items():
        cp = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        hist = read(out/'runs'/key/'history.json')
        metric = 'VAL_5D_COMMODITY_MACRO_MAE' if key in REUSE else 'VAL5_macro_MAE'
        if min(hist, key=lambda r: r[metric])['epoch'] != cp['epoch'] or cp['epoch'] != row['best_epoch']:
            raise ValueError('Best epoch mismatch')
        m = model_for(key, out, cfg)
        m.load_state_dict(cp['model_state_dict'], strict=True)
        if sha(row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        audit[key] = dict(strict_load=True, selection_verified=True, best_epoch=cp['epoch'], checkpoint_sha256=row['checkpoint_sha256'])
        del m, cp
    save(out/'CHECKPOINT_AUDIT.json', audit)
    save(out/'models_frozen.json', done)
    frozen_sha = sha(out/'models_frozen.json')
    release = Path(cfg['test_release_path'])
    if sha(release/'dataset_manifest.json') != cfg['test_release_manifest_sha256']:
        raise ValueError('Different TEST release')
    manifest = read(release/'dataset_manifest.json')
    for name, digest in manifest['files'].items():
        if sha(release/name) != digest:
            raise ValueError('TEST cache changed')
    ds = TestReleaseDataset(release)
    # Match the already frozen C4/C0 TEST inference environment; no training
    # remains, so this does not change optimizer/Dropout RNG trajectories.
    test_device = torch.device('cpu')
    torch.set_num_threads(2)
    save(out/'TEST_ENVIRONMENT.json', dict(device='cpu', torch=torch.__version__, numpy=np.__version__,
        torch_threads=2, dtype='float32', autocast=False, matched_reused_reference=True))
    test = {}
    for key in KEYS:
        folder = out/'runs'/key
        if (folder/'TEST_result.json').exists():
            row = read(folder/'TEST_result.json')
            if row['checkpoint_sha256'] != done[key]['checkpoint_sha256']:
                raise ValueError('Existing TEST result wrong checkpoint')
            test[key] = row
            continue
        save(out/'STATUS.json', dict(status='EVALUATING_TEST', complete=len(test), current=key, total=13))
        if key in REUSE:
            source = Path(cfg['reused_runs'][key]['original_test_result'])
            original = read(source)
            if original['checkpoint_sha256'] != done[key]['checkpoint_sha256']:
                raise ValueError('Reused TEST identity mismatch')
            shutil.copyfile(source.parent/'test_predictions.npz', folder/'test_predictions.npz')
            metrics = original['metrics']
            seconds = 0.
        else:
            m = model_for(key, out, cfg, test_device)
            cp = torch.load(done[key]['checkpoint'], map_location='cpu', weights_only=False)
            m.load_state_dict(cp['model_state_dict'], strict=True); m.set_epoch(cp['epoch'])
            m.requires_grad_(False).eval()
            before = tensor_state_hash(m.state_dict())
            tic = time.perf_counter()
            a = infer(m, ds, test_device, cfg['targets'])
            np.savez_compressed(folder/'test_predictions.npz', **a)
            metrics = test_metrics(a['prediction'], a['target'], a['label_valid_mask'])
            seconds = time.perf_counter()-tic
            if tensor_state_hash(m.state_dict()) != before:
                raise ValueError('Frozen model changed')
            del m, cp
            gc.collect(); torch.cuda.empty_cache()
        with np.load(folder/'test_predictions.npz', allow_pickle=False) as a:
            if test_metrics(a['prediction'], a['target'], a['label_valid_mask']) != metrics:
                raise ValueError('Saved TEST metrics differ')
        row = dict(key=key, status='COMPLETE', test_evaluated=True, metrics=metrics,
            checkpoint=done[key]['checkpoint'], checkpoint_sha256=done[key]['checkpoint_sha256'],
            best_epoch=done[key]['best_epoch'], models_frozen_sha256=frozen_sha,
            predictions_sha256=sha(folder/'test_predictions.npz'), inference_seconds=seconds,
            reused=key in REUSE, original_test_result=cfg['reused_runs'][key]['original_test_result'] if key in REUSE else None)
        save(folder/'TEST_result.json', row)
        test[key] = row
        save(out/'TEST_METRICS.json', test)
        summarize(out, cfg, done, test)
        print('[TEST]', key, metrics['primary'], flush=True)
    for name, digest in read(out/'PROTECTED_FILES.json').items():
        if sha(name) != digest:
            raise ValueError('Historical artifact changed: '+name)
    save(out/'AUDIT_RESULTS.json', dict(status='PASS', rows_complete=13, new_fits_complete=11, reused=2,
        shared_initialization=True, historical_artifacts_unchanged=True, strict_checkpoints=True,
        checkpoint_selection='VAL5 macro MAE', test_evaluated=True, no_additional_search=True))
    save(out/'STATUS.json', dict(status='COMPLETE', rows_complete=13, new_fits_complete=11, reused=2, test_evaluated=True))
    print('COMPLETE', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--test-reference', type=Path, default=TEST_REFERENCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(4)
    out = a.output.resolve()
    if not (out/'config.json').exists():
        prepare(out, a.reference.resolve(), a.test_reference.resolve())
    with (out/'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.run:
                run(out, torch.device(a.device))
            else:
                print('PREPARED (training NOT_RUN)', out)
        except Exception:
            save(out/('failure_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'.json'),
                 dict(traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()

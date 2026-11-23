"""Latest CMGM Deep V3 architectures on the frozen new-data D0B protocol.

Reached through baseline_comparison --triweave-reference. Default: preflight.
--run explicitly authorizes the eight fixed seed42 fits and frozen TEST inference.
"""
import argparse
import copy
import csv
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
from torch.utils.data import Dataset, DataLoader
from cmgm.models.comparison_baselines import ORDER, MODEL_CONFIGS
from cmgm.models.triweave_baseline_adapter import MaskedNeutralAdapter, prepared_model
from cmgm.scripts.formal_main_innovation_ablation import (
    ROOT, sha, save, checkpoint, seed_all, state_rng, restore_rng,
    prediction_loss, metrics,
)

PROTOCOL_ID = 'CMGM_DEEP_V3_TRIWEAVE23_MASKED_V1'
HORIZONS = (1, 5, 10, 20)


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    paths = [ROOT/'cmgm/models/comparison_baselines.py',
             ROOT/'cmgm/models/triweave_baseline_adapter.py',
             ROOT/'cmgm/scripts/baseline_comparison.py', Path(__file__),
             ROOT/'cmgm/scripts/formal_main_innovation_ablation.py',
             ROOT/'cmgm/models/model.py', ROOT/'cmgm/graph/adaptive_graph.py',
             ROOT/'tests/test_triweave_baselines.py']
    paths += sorted(p for p in (ROOT/'cmgm/models/baselines').iterdir() if p.suffix in ('.py','.md','.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


class NeutralDataset(Dataset):
    def __init__(self, folder, split):
        self.split = split
        self.a = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
                  for p in Path(folder).glob('*.npy')}
        self.indices = self.a[split+'_window_indices']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        x = np.array(self.a['features'][np.maximum(idx, 0)], copy=True)
        x[idx < 0] = 0
        return dict(X=x, Y=np.array(self.a[self.split+'_targets'][k], copy=True),
                    label_valid_mask=np.array(self.a[self.split+'_label_valid_mask'][k], copy=True),
                    origin_date=str(self.a['reference_dates'][int(self.a[self.split+'_origin_indices'][k])]))


def loader(folder, split):
    return DataLoader(NeutralDataset(folder, split), batch_size=64, shuffle=False,
                      drop_last=False, num_workers=0)


def cache_inputs(source, dest, market_indices, splits):
    """Deterministic current-window view; no refit, no raw CSV read, no labels used."""
    dest.mkdir(parents=True, exist_ok=False)
    names = ['features', 'input_valid_mask', 'feature_valid_mask', 'feature_applicable_mask']
    arrays = {k: np.load(source/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in names}
    adapter = MaskedNeutralAdapter(market_indices, arrays['features'].shape[-1])
    neutral = []
    with torch.inference_mode():
        for i in range(0, len(arrays['features']), 128):
            batch = {k: torch.from_numpy(np.array(v if k=='feature_applicable_mask' else v[i:i+128], copy=True))
                     for k, v in arrays.items()}
            # Here the batch dimension holds dates, and time has length one.
            neutral.append(adapter(batch['features'][:, None], batch['input_valid_mask'][:, None],
                                   batch['feature_valid_mask'][:, None],
                                   batch['feature_applicable_mask']).squeeze(1).numpy())
    np.save(dest/'features.npy', np.concatenate(neutral), allow_pickle=False)
    selected = ['reference_dates']
    for split in splits:
        selected += [split+'_'+k for k in ('window_indices', 'origin_indices', 'targets', 'label_valid_mask')]
    for key in selected:
        shutil.copyfile(source/(key+'.npy'), dest/(key+'.npy'))
    manifest = dict(source=str(source), source_hashes={p.name: sha(p) for p in source.glob('*.npy')
        if p.stem in names+selected}, files={p.name: sha(p) for p in dest.glob('*.npy')},
        input_view='stock mean/std + bond mean/std + 23 commodity tokens; 27 x 26',
        fitted_statistics=False, scalar_type='float32', feature_order_unchanged=True)
    save(dest/'cache_manifest.json', manifest)
    return manifest


@torch.inference_mode()
def evaluate(model, batches, device, targets, return_arrays=False):
    model.eval()
    ps, ys, ms, dates = [], [], [], []
    for batch in batches:
        p = model(batch['X'].to(device))
        if p.shape != batch['Y'].shape or p.shape[1:] != (4, len(targets)) or not torch.isfinite(p).all():
            raise ValueError('Prediction shape/finite failure')
        ps.append(p.cpu().numpy()); ys.append(batch['Y'].numpy())
        ms.append(batch['label_valid_mask'].numpy()); dates += batch['origin_date']
    arrays = dict(prediction=np.concatenate(ps), target=np.concatenate(ys),
                  label_valid_mask=np.concatenate(ms), origin_dates=np.array(dates),
                  commodity_order=np.array(targets))
    if len(dates) != len(batches.dataset) or len(set(dates)) != len(dates):
        raise ValueError('Origin coverage failure')
    result = metrics(arrays['prediction'], arrays['target'], arrays['label_valid_mask'], targets)
    result.update(origins=len(dates), first_origin=dates[0], last_origin=dates[-1])
    return (result, arrays) if return_arrays else result


def synthetic_preflight():
    """No real TEST access and no optimizer steps; independent samples and gradient sanity."""
    checks = {}
    for name in ORDER:
        seed_all()
        model = prepared_model(name).eval()
        x = torch.randn(3, 20, 27, 26)
        p = model(x)
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            perm_diff = float((model(x[permutation])-p[permutation]).abs().max())
            single_diff = float((model(x[:1])-p[:1]).abs().max())
        loss = prediction_loss(p, torch.randn_like(p)*.02, torch.ones_like(p, dtype=torch.bool))
        loss.backward()
        finite = all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        if p.shape != (3, 4, 23) or not finite or not torch.isfinite(p).all() or max(perm_diff, single_diff)>2e-6:
            raise ValueError(f'{name} synthetic preflight failed: {perm_diff}, {single_diff}')
        restored = prepared_model(name).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(x), p, atol=0, rtol=0)
        checks[name] = dict(parameters=sum(v.numel() for v in model.parameters()),
                            active_gradient_parameters=sum(v.numel() for v in model.parameters() if v.grad is not None),
                            unused_parameter_names=[n for n,v in model.named_parameters() if v.grad is None],
                            batch_permutation_max_diff=perm_diff, single_sample_max_diff=single_diff,
                            shape=list(p.shape), finite_gradient=True, strict_roundtrip=True, PASS=True)
    return checks


def prepare(reference, output_root):
    ref = read(reference/'config.json')
    done = read(reference/'runs/full_d0b/training_complete.json')
    if read(reference/'status.json')['status'] != 'COMPLETE' or done['name'] != 'Full D0B':
        raise ValueError('Expected the completed new-data original D0B experiment')
    if sha(done['checkpoint']) != done['checkpoint_sha256']:
        raise ValueError('Reference checkpoint SHA mismatch')
    if ref['protocol']['id'] != 'CMGM_SEP17_TRIWEAVE23_MASKED_V1':
        raise ValueError('Unexpected new-data protocol')
    data = Path(ref['data_path'])
    if sha(data/'dataset_manifest.json') != ref['dataset_manifest_sha256']:
        raise ValueError('Reference data manifest changed')
    assets = read(data/'asset_manifest.json')
    mi = {}
    for market in ('stock', 'bond', 'commodity'):
        source_market = 'future' if market=='commodity' else market
        indices = [a['index'] for a in assets if a['market']==source_market]
        if indices != list(range(min(indices), max(indices)+1)):
            raise ValueError('Noncontiguous market mapping')
        mi[market] = [min(indices), max(indices)+1]
    if [a['asset_id'] for a in assets if a['market']=='future'] != ref['target_order']:
        raise ValueError('Commodity output mapping differs')
    out = output_root/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    protocol = copy.deepcopy(ref['protocol'])
    protocol.update(id=PROTOCOL_ID, models=list(ORDER), reference_model='D0B original gate; frozen/reused',
        architecture_source='current CMGM working tree Deep V3, not Sep17 baselines',
        baseline_auxiliary_loss='NONE; no KL for baseline architectures',
        input_adapter='masked stock mean/std, bond mean/std, individual commodity tokens; 27x26',
        baseline_missingness='invalid raw cells excluded before summaries; missing normalized feature = neutral zero. Cores have fixed token slots and no dynamic graph masks; no claim of full-node equivalence to D0B.',
        train_budget_comparison='same new-data train and VAL selector as reference; differs from legacy V3 drop_last/batch-mean',
        test='all eight fixed selected checkpoints frozen before loading TEST; same reference release/population')
    protocol.pop('configurations'); protocol.pop('full')
    protocol['train']['Switch_KL'] = 'NONE for the eight baselines; reused D0B retains native KL'
    cpdir = ROOT/'checkpoints/baselines_deep_v3_triweave'/out.name
    config = dict(protocol=protocol, reference=str(reference.resolve()), reference_config_sha256=sha(reference/'config.json'),
        reference_checkpoint=done, data_path=str(data), market_indices=mi, targets=ref['target_order'],
        dataset_manifest_sha256=sha(data/'dataset_manifest.json'), checkpoint_dir=str(cpdir),
        source_hashes=sources(), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        git_diff_stat=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True),
        legacy_model_configs=MODEL_CONFIGS,
        dimension_substitution=dict(commodities=[24,23],features=[21,26],neutral_nodes=[28,27],sequence_width=[588,702],outputs=[96,92]),
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        authorization='User requested latest CMGM baselines trained/evaluated on imported new data, 2026-10-22')
    save(out/'config.json', config)
    for relative in config['source_hashes']:
        dest = out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/relative, dest)
    cache_inputs(data, out/'neutral_train_val', mi, ('train', 'val'))
    from cmgm.scripts.formal_main_innovation_ablation import model_for as d0b_model, loaders as d0b_loaders, evaluate as d0b_evaluate
    frozen = torch.load(done['checkpoint'], map_location='cpu', weights_only=False)
    if frozen['name'] != 'Full D0B' or frozen['epoch'] != done['best_epoch'] or frozen['config_sha256'] != sha(reference/'config.json'):
        raise ValueError('Reference checkpoint identity/epoch/config differs')
    reference_model = d0b_model('Full D0B', reference, torch.device('cpu'))
    reference_model.load_state_dict(frozen['model_state_dict'], strict=True)
    reference_model.set_epoch(frozen['epoch'])
    reproduced = d0b_evaluate(reference_model, d0b_loaders(data,'val'), torch.device('cpu'), config['targets'])
    old_val = read(reference/'runs/full_d0b/train_val_metrics.json')['val']
    if not np.isclose(reproduced['selection'], old_val['selection'], atol=1e-9, rtol=1e-5):
        raise ValueError('Frozen D0B VAL reproduction failed')
    save(out/'reference_verification.json', dict(strict_load=True, checkpoint_sha256=sha(done['checkpoint']),
        best_epoch=frozen['epoch'], recorded_VAL_Huber=old_val['selection'], reproduced_VAL_Huber=reproduced['selection'],
        reproduced_VAL_MAE={h:v['MAE'] for h,v in reproduced['horizons'].items()}, checkpoint_unchanged=True, PASS=True))
    del reference_model, frozen
    checks = synthetic_preflight()
    # Real TRAIN/VAL shapes/loss, without fitting; TEST cache is not opened here.
    real = {}
    first = next(iter(loader(out/'neutral_train_val', 'train')))
    for name in ORDER:
        seed_all(); model = prepared_model(name).eval()
        with torch.no_grad():
            p = model(first['X'][:3])
            loss = prediction_loss(p, first['Y'][:3], first['label_valid_mask'][:3])
        if not torch.isfinite(loss):raise ValueError('Real data loss is nonfinite')
        real[name] = dict(shape=list(p.shape), initial_loss=float(loss))
    save(out/'preflight.json', dict(synthetic=checks, actual_train_val=real,
        coverage={s:len(loader(out/'neutral_train_val', s).dataset) for s in ('train','val')},
        TEST_values_read=False, PASS=True))
    save(out/'status.json', dict(status='PREPARED', complete=0, total=8))
    print('PREPARED', out, flush=True)
    return out


def train_one(name, out, cfg, device):
    folder = out/'runs'/name; folder.mkdir(parents=True, exist_ok=True)
    cpdir = Path(cfg['checkpoint_dir'])/name; cpdir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = cpdir/'best.pt', cpdir/'latest.pt'
    if (folder/'training_complete.json').exists():
        done = read(folder/'training_complete.json')
        if sha(done['checkpoint']) != done['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return done
    seed_all(); model = prepared_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    batches = loader(out/'neutral_train_val', 'train'); val_batches = loader(out/'neutral_train_val', 'val')
    history=[]; best=float('inf'); best_epoch=None; stale=0; start_epoch=1; seconds=0.
    if latest_path.exists():
        cp=torch.load(latest_path, map_location='cpu', weights_only=False)
        if cp['config_sha256'] != sha(out/'config.json') or cp['name'] != name:raise ValueError('Resume identity differs')
        model.load_state_dict(cp['model_state_dict'], strict=True);optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']);restore_rng(cp['rng'])
        history=cp['history'];best=cp['best_val'];best_epoch=cp['best_epoch'];stale=cp['stale']
        start_epoch=cp['epoch']+1;seconds=cp['seconds']
    for epoch in range(start_epoch, 201):
        if stale>=10:break
        tic=time.perf_counter();model.train();losses=[];origins=[]
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            p=model(batch['X'].to(device))
            loss=prediction_loss(p,batch['Y'].to(device),batch['label_valid_mask'].to(device))
            if not torch.isfinite(p).all() or not torch.isfinite(loss):raise ValueError('Nonfinite training')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError('Nonfinite gradient')
            optimizer.step();losses.append(float(loss.detach()));origins+=batch['origin_date']
        if len(origins)!=len(batches.dataset) or len(set(origins))!=len(origins):raise ValueError('TRAIN coverage failure')
        val=evaluate(model,val_batches,device,cfg['targets']);score=val['selection'];scheduler.step(score)
        improved=score<best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        elapsed=time.perf_counter()-tic;seconds+=elapsed
        row=dict(epoch=epoch,train_prediction_batch_mean=float(np.mean(losses)),
            VAL_prediction_global_valid_label_Huber=score,VAL_MAE={h:v['MAE'] for h,v in val['horizons'].items()},
            optimizer_steps=len(losses),train_origins_used=len(origins),coverage=1.,LR=optimizer.param_groups[0]['lr'],seconds=elapsed)
        history.append(row);save(folder/'training_history.json',history)
        state=dict(name=name,epoch=epoch,best_epoch=best_epoch,best_val=best,stale=stale,seconds=seconds,
            model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=state_rng(),history=history,config_sha256=sha(out/'config.json'))
        if improved:checkpoint(best_path,state)
        checkpoint(latest_path,state)
        print(f'[{name}] epoch={epoch} VAL_Huber={score:.9g} VAL5_MAE={val["horizons"]["5"]["MAE"]:.9g} best={best_epoch} seconds={elapsed:.1f}',flush=True)
    cp=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True)
    val,arrays=evaluate(model,val_batches,device,cfg['targets'],True)
    if not np.isclose(val['selection'],best,atol=1e-10,rtol=1e-6):raise ValueError('Best VAL reproduction mismatch')
    np.savez_compressed(folder/'val_predictions.npz',**arrays)
    train=evaluate(model,batches,device,cfg['targets'])
    save(folder/'train_val_metrics.json',dict(train=train,val=val))
    done=dict(name=name,status='COMPLETE',best_epoch=cp['epoch'],best_val_Huber=best,
        parameters=sum(v.numel() for v in model.parameters()),training_seconds=seconds,
        checkpoint=str(best_path),checkpoint_sha256=sha(best_path),strict_restore=True,test_evaluated=False)
    save(folder/'training_complete.json',done)
    del model,optimizer,scheduler,cp;gc.collect();torch.cuda.empty_cache()
    return done


def write_report(out, cfg, done, results):
    reference=Path(cfg['reference'])
    ref_test=read(reference/'test_metrics.json')['Full D0B']
    # The reference container is inspected at preparation/run; no numeric constants.
    if 'metrics' in ref_test:ref_test=ref_test['metrics']
    if 'test' in ref_test:ref_test=ref_test['test']
    all_results={**results,'D0B':ref_test}
    rows=[]; multi=[]
    ref5=ref_test['horizons']['5']['MAE']
    for name in (*ORDER,'D0B'):
        d=cfg['reference_checkpoint'] if name=='D0B' else done.get(name)
        r=all_results.get(name)
        if not r:continue
        value=r['horizons']['5']
        rows.append(dict(Model=name,TEST_MAE_5d=value['MAE'],TEST_MSE_5d=value['MSE'],TEST_RMSE_5d=value['RMSE'],
            TEST_Hit_5d_percent=100*value['Hit'],DeltaMAE_vs_D0B=value['MAE']-ref5,
            Relative_Delta_percent=100*(value['MAE']/ref5-1),best_epoch=d['best_epoch'],parameters=d['parameters']))
        for h,v in r['horizons'].items():multi.append(dict(Model=name,horizon=int(h),**v))
    for filename, data in [('main_TEST5_table.csv',rows),('multi_horizon_TEST_table.csv',multi)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['# Latest CMGM Deep V3 — TriWeave TEST comparison','',
        f'Completed baseline TEST evaluations: {len(results)}/8. D0B is read-only reused from {reference}.',
        'Fixed seed42, 23 commodities, decimal cumulative returns; masks and split population identical. Hit shown as percent.',
        'VAL-only global valid-label four-horizon Huber selects checkpoints; TEST does not select models or seeds.','',
        '| Model | TEST MAE@5d | TEST RMSE@5d | Hit % | Delta % vs D0B | best epoch | parameters |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["Model"]} | {r["TEST_MAE_5d"]:.10f} | {r["TEST_RMSE_5d"]:.10f} | {r["TEST_Hit_5d_percent"]:.3f} | {r["Relative_Delta_percent"]:+.3f} | {r["best_epoch"]} | {r["parameters"]} |')
    lines+=['','Baselines use the existing V3 compressed neutral view (4 market summaries + commodity tokens), not D0B full-node input. Economic-type features are normalized by the frozen TRAIN scaler; invalid cells are excluded from summaries and encoded as neutral zeros. The fixed-slot V3 cores do not implement dynamic availability graph masks. No claim of identical internal information representation.',
        'Dimensions change from 28x21/24 targets to 27x26/23 targets; hidden widths, depths, dropout, graph/FFT mechanisms remain V3. MSGNet latent graph width remains32; its latent-to-node convolution changes kernel5 to6 as implied by 27 nodes.',
        'Single run; no significance or cross-period generalization claim. Dataset is exploratory: historical PIT membership and continuous futures roll certification remain unverified. Historical TEST has already been viewed.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out, device):
    cfg=read(out/'config.json')
    if cfg['source_hashes']!=sources():raise ValueError('Working source differs from frozen configuration; STOP')
    reference=Path(cfg['reference']);cpref=cfg['reference_checkpoint']
    if sha(cpref['checkpoint'])!=cpref['checkpoint_sha256']:raise ValueError('D0B checkpoint changed')
    for path in (out/'neutral_train_val').glob('*.npy'):
        if sha(path)!=read(out/'neutral_train_val/cache_manifest.json')['files'][path.name]:raise ValueError('Input cache changed')
    if not device.type=='cuda':raise ValueError('Formal training requires CUDA; no fallback')
    save(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        device=str(device),device_name=torch.cuda.get_device_name(device),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
    done={}
    for name in ORDER:
        save(out/'status.json',dict(status='TRAINING',current=name,complete=len(done),total=8))
        done[name]=train_one(name,out,cfg,device);save(out/'training_index.json',done)
    freeze=out/'models_frozen.json'
    frozen=dict(config_sha256=sha(out/'config.json'),models=done,selection=cfg['protocol']['selection'])
    if freeze.exists():
        if read(freeze)!=frozen:raise ValueError('Frozen model list changed')
    else:save(freeze,frozen)
    # TEST is first opened here, after all eight model identities are locked.
    test_source=reference/'test_dataset'
    manifest=read(test_source/'dataset_manifest.json')
    if not manifest['audit_pass'] or manifest['origin_count']!=423:raise ValueError('Unexpected TEST release')
    for filename, expected in manifest['files'].items():
        if sha(test_source/filename)!=expected:raise ValueError('Reference TEST release changed: '+filename)
    if not (out/'neutral_test').exists():cache_inputs(test_source,out/'neutral_test',cfg['market_indices'],('test',))
    else:
        for filename, expected in read(out/'neutral_test/cache_manifest.json')['files'].items():
            if sha(out/'neutral_test'/filename)!=expected:raise ValueError('TEST cache changed')
    reference_arrays=np.load(reference/'runs/full_d0b/test_predictions.npz',allow_pickle=False)
    results=read(out/'test_metrics.json') if (out/'test_metrics.json').exists() else {}
    for name in ORDER:
        if name in results:continue
        save(out/'status.json',dict(status='TEST_EVALUATION',current=name,complete=len(results),total=8))
        seed_all();model=prepared_model(name).to(device)
        cp=torch.load(done[name]['checkpoint'],map_location='cpu',weights_only=False)
        if sha(done[name]['checkpoint'])!=done[name]['checkpoint_sha256']:raise ValueError('Frozen best checkpoint changed')
        model.load_state_dict(cp['model_state_dict'],strict=True)
        score,arrays=evaluate(model,loader(out/'neutral_test','test'),device,cfg['targets'],True)
        for key in ('target','label_valid_mask','origin_dates','commodity_order'):
            if not np.array_equal(arrays[key],reference_arrays[key]):raise ValueError('TEST population mismatch: '+key)
        np.savez_compressed(out/'runs'/name/'test_predictions.npz',**arrays)
        results[name]=score;save(out/'test_metrics.json',results);write_report(out,cfg,done,results)
        print('[TEST COMPLETE]',name,'MAE@5d',score['horizons']['5']['MAE'],flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()
    if sha(cpref['checkpoint'])!=cpref['checkpoint_sha256']:raise ValueError('D0B changed')
    save(out/'file_index.json',dict(config=str(out/'config.json'),report=str(out/'FINAL_REPORT.md'),
        table=str(out/'main_TEST5_table.csv'),all_horizons=str(out/'multi_horizon_TEST_table.csv'),
        models={name:dict(**d,predictions=str(out/'runs'/name/'test_predictions.npz'),
            history=str(out/'runs'/name/'training_history.json'),train_val=str(out/'runs'/name/'train_val_metrics.json')) for name,d in done.items()},
        D0B_reference=str(reference)))
    save(out/'status.json',dict(status='COMPLETE',complete=8,total=8,test_evaluated=True,reference_checkpoint_unchanged=True))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--triweave-reference',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/baseline_comparison_deep_v3_triweave')
    p.add_argument('--resume',type=Path)
    p.add_argument('--run',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);seed_all()
    out=args.resume or prepare(args.triweave_reference,args.output_dir)
    with (out/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.run:run(out,torch.device(args.device))
        except Exception:
            save(out/'failure.json',dict(traceback=traceback.format_exc(),time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
            raise


if __name__=='__main__':main()

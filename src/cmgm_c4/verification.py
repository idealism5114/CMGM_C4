"""Full TRAIN/VAL replay, strict checkpoint recovery and no-old-repo audit."""
import builtins
from contextlib import contextmanager
import io
from pathlib import Path
import sys
import time
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,sha,tensor_hash
from cmgm_c4.factory import build,restore
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import evaluate
from cmgm_c4.evaluation.metrics import population


@contextmanager
def train_val_guard():
    original,old_io=builtins.open,io.open;hits=[]
    def check(p):
        if isinstance(p,(str,bytes,Path)):
            p=Path(p.decode() if isinstance(p,bytes) else p)
            if p.suffix.lower() in ('.npy','.npz','.csv','.json','.pt') and (p.name.lower().startswith('test_') or any('test_release' in s for s in p.parts)):
                hits.append(str(p));raise RuntimeError('Verification forbids TEST numeric access: '+str(p))
    def guard(p,*a,**kw):check(p);return original(p,*a,**kw)
    def guard_io(p,*a,**kw):check(p);return old_io(p,*a,**kw)
    builtins.open,io.open=guard,guard_io
    try:yield hits
    finally:builtins.open,io.open=original,old_io


def verify(device='cpu'):
    root=project_root();out=root/'verification';out.mkdir(exist_ok=True)
    registry=read(root/'configs/registry.json');transport=read(root/'provenance/migration_manifest.json')
    # Opaque hash validation is deliberately outside numeric-array guard.
    for r in transport['files']:
        p=root/r['destination']
        if p.is_symlink() or sha(p)!=r['sha256']:raise ValueError('Copied archive drift: '+str(p))
    save(out/'TRANSPORT_AUDIT.json',dict(status='PASS',files=len(transport['files']),all_hashes_identical=True,
        TEST_operation='opaque bytes SHA only',TEST_numeric_arrays_parsed=0))
    rows={};start=time.perf_counter();used_modules=[]
    with train_val_guard() as hits:
        for ident,row in registry.items():
            m,cp=restore(ident,device=device);_,latest=restore(ident,which='latest',device='cpu')
            record=dict(checkpoint_sha256=sha(root/row['checkpoint']),strict_best_latest=True,best_epoch=cp['epoch'],parameters=sum(p.numel() for p in m.parameters()),splits={})
            if cp['epoch']!=row['best_epoch']:raise ValueError('Wrong selected epoch')
            for split in ('train','val'):
                tic=time.perf_counter();scores,a=evaluate(m,loader(split,ident.startswith('baselines/')),device,ident.startswith('baselines/'),True)
                with np.load(root/row['results']/(split+'_predictions.npz'),allow_pickle=False) as old:
                    for k in ('target','label_valid_mask','origin_dates'):np.testing.assert_array_equal(a[k],old[k])
                    d=float(np.max(np.abs(a['prediction']-old['prediction'])))
                    np.testing.assert_allclose(a['prediction'],old['prediction'],atol=1e-6,rtol=1e-5)
                    for k in ('routing','expert_prediction'):
                        if k in a and k in old:np.testing.assert_allclose(a[k],old[k],atol=1e-6,rtol=1e-5)
                    # Historical suites use different JSON filenames. Recompute
                    # from their exact archived TRAIN/VAL arrays, without any
                    # new denominator or dependency on reporting filenames.
                    expected=population(old['prediction'],old['target'],old['label_valid_mask'])['primary']
                diff=abs(scores['primary']-expected)
                if diff>1e-8:raise ValueError('Metric replay drift '+ident+' '+split)
                record['splits'][split]=dict(origins=len(a['prediction']),max_prediction_abs_diff=d,MAE5=scores['primary'],metric_difference=diff,seconds=time.perf_counter()-tic)
                dest=out/'replayed'/row['suite']/row['key'];dest.mkdir(parents=True,exist_ok=True)
                save(dest/(split+'_metrics.json'),scores)
                print('[REPLAY PASS]',ident,split,'MAE5',scores['primary'],'maxdiff',d,flush=True)
            rows[ident]=record;save(out/'REPLAY_PROGRESS.json',rows)
            del m,cp,latest
        for name,mod in sys.modules.items():
            path=getattr(mod,'__file__',None)
            if path and ('Commedities' in path or name=='cmgm' or name.startswith(('cmgm.','cmgm_sep17_frozen'))):
                raise ValueError('External old-repository runtime import: '+name+' '+path)
            if name.startswith('cmgm_c4') and path:used_modules.append(str(Path(path).resolve().relative_to(root)))
    result=dict(status='PASS',models=len(rows),replays=len(rows)*2,records=rows,TEST_guard_hits=hits,
        TEST_numeric_arrays_parsed=0,TEST_inference=0,formal_training=0,seconds=time.perf_counter()-start,
        runtime_modules=sorted(set(used_modules)),old_repository_runtime_imports=0)
    save(out/'TRAIN_VAL_REPLAY.json',result);print('ALL TRAIN/VAL REPLAYS PASS',flush=True)
    return result

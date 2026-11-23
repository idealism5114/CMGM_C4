"""Small stable CLI. Training and TEST each require an explicit switch."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from cmgm_c4.common import project_root,read,save,sha,runtime_hashes,audit_train_val_files
from cmgm_c4.factory import restore,restore_run
from cmgm_c4.data.dataset import loader
from cmgm_c4.evaluation.evaluate import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command')
    sub.add_parser('list');sub.add_parser('preflight')
    v=sub.add_parser('verify');v.add_argument('--device',default='cpu')
    e=sub.add_parser('evaluate');e.add_argument('--model',default='c4/full');e.add_argument('--split',choices=['train','val','test'],default='val')
    e.add_argument('--run-dir',type=Path,help='Completed new run; otherwise use migrated frozen checkpoint')
    e.add_argument('--allow-test',action='store_true');e.add_argument('--device',default='cpu');e.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('summarize');s.add_argument('--output',type=Path,default=None);s.add_argument('--runs-dir',type=Path)
    t=sub.add_parser('train');t.add_argument('--model',default='c4/full');t.add_argument('--device',default='cuda:0')
    t.add_argument('--output',type=Path,required=True);t.add_argument('--execute',action='store_true');t.add_argument('--resume',action='store_true')
    t.add_argument('--lr',type=float);t.add_argument('--weight-decay',type=float)
    a=p.parse_args();root=project_root();torch.set_num_threads(4)
    if a.command in (None,'list'):
        for k,r in read(root/'configs/registry.json').items():print(k,'epoch',r['best_epoch'],r['selection'])
    elif a.command=='preflight' or (a.command=='train' and not a.execute):
        return subprocess.call([sys.executable,'-m','pytest','-q',str(root/'tests')],cwd=root)
    elif a.command=='verify':
        from cmgm_c4.verification import verify
        verify(a.device)
    elif a.command=='evaluate':
        if a.split=='test' and not a.allow_test:raise ValueError('TEST requires --allow-test; never enabled by training')
        if a.output.exists():raise ValueError('Evaluation output must be new; do not overwrite archives')
        if a.split in ('train','val'):audit_train_val_files()
        if a.run_dir:
            m,cp,frozen,cp_path=restore_run(a.run_dir,a.device);a.model=frozen['identifier']
        else:
            m,cp=restore(a.model,device=a.device);cp_path=root/read(root/'configs/registry.json')[a.model]['checkpoint']
        before=sha(cp_path);m.requires_grad_(False)
        r,arrays=evaluate(m,loader(a.split,a.model.startswith('baselines/'),a.allow_test),a.device,a.model.startswith('baselines/'),True)
        a.output.mkdir(parents=True);save(a.output/'metrics.json',r);np.savez_compressed(a.output/'predictions.npz',**arrays)
        if sha(cp_path)!=before:raise ValueError('Checkpoint changed during evaluation')
        save(a.output/'provenance.json',dict(model=a.model,best_epoch=cp['epoch'],selection='VAL5 Commodity Macro MAE',split=a.split,
            checkpoint_sha256=before,source_hashes=runtime_hashes(),checkpoint_unchanged=True,TEST_exploratory=a.split=='test'))
    elif a.command=='train':
        from cmgm_c4.training.train import train
        train(a.model,a.output,a.device,a.resume,lr=a.lr,weight_decay=a.weight_decay)
    else:
        from cmgm_c4.reporting import summarize
        summarize(a.output,a.runs_dir)


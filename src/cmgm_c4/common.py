"""Portable project paths and small checkpoint utilities."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def project_root():
    if os.getenv('CMGM_C4_ROOT'):
        p=Path(os.environ['CMGM_C4_ROOT']).resolve()
    else:p=Path(__file__).resolve().parents[2]
    if not (p/'configs/project.json').exists():raise ValueError('Set CMGM_C4_ROOT to the standalone project folder')
    return p

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
def checkpoint(p,state):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(state,tmp);tmp.replace(p)
def prediction_loss(p,y,mask):
    terms=[]
    for h in range(4):
        valid=mask[:,h]
        if not valid.any():raise ValueError('Empty horizon; no silent zero loss')
        terms.append(torch.nn.functional.huber_loss(p[:,h][valid],y[:,h][valid],delta=.02))
    return sum(terms)

def runtime_hashes():
    root=project_root()
    return {str(p.relative_to(root)):sha(p) for p in sorted((root/'src').rglob('*.py'))}

def audit_train_val_files():
    """Validate the copied frozen inputs, never opening a TEST file."""
    root=project_root();files=[]
    for row in read(root/'provenance/migration_manifest.json')['files']:
        if row['category'] not in ('TRAIN_VAL_frozen_data','baseline_neutral_TRAIN_VAL'):continue
        path=root/row['destination']
        if path.is_symlink() or sha(path)!=row['sha256']:raise ValueError('Frozen data changed: '+row['destination'])
        files.append(row['destination'])
    if not files:raise ValueError('Missing TRAIN/VAL fingerprint records')
    return dict(files=len(files),status='PASS',TEST_files_opened=0)


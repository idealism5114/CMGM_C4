"""Explicit nine-trial C4 run; default invocation performs CPU preflight only."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--allow-test',action='store_true')
    a=p.parse_args()
    if not a.execute:return subprocess.call([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=ROOT)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    checked=subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=ROOT,capture_output=True,text=True)
    log=a.output.parent/(a.output.name+'_preflight.txt')
    with log.open('a' if a.resume else 'x') as f:f.write(checked.stdout+checked.stderr)
    print(checked.stdout,flush=True)
    if checked.returncode:return checked.returncode
    import torch
    torch.set_num_threads(4)
    from cmgm_c4.training.tuning import run_search
    run_search(a.output,a.device,a.resume,a.allow_test)


if __name__=='__main__':raise SystemExit(main())

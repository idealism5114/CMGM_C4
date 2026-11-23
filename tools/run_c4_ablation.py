"""Fresh Full+12 C4 ablations; default preflight, explicit training and TEST."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def preflight():
    import torch
    import pytest
    from cmgm_c4.verification import train_val_guard
    torch.set_num_threads(2)
    # Exclude the old report-format test that intentionally reads historical
    # TEST metric JSONs. All other tests run inside the TEST access sentinel.
    with train_val_guard():
        return pytest.main(['-q',str(ROOT/'tests'),'-k','not test_summary_uses_historical_json_not_test_arrays'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.preflight_child or not args.execute:return preflight()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    check=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),'--preflight-child'],cwd=ROOT,capture_output=True,text=True)
    with (args.output.parent/(args.output.name+'_preflight.txt')).open('a' if args.resume else 'x') as f:f.write(check.stdout+check.stderr)
    print(check.stdout+check.stderr,flush=True)
    if check.returncode:return check.returncode
    import torch
    torch.set_num_threads(4)
    from cmgm_c4.training.ablation_suite import run_suite
    run_suite(args.output,args.device,args.resume,args.allow_test)


if __name__=='__main__':raise SystemExit(main())

"""Paired progressive A–E ablations. Use --legacy for the historical Full+12 suite."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def preflight(legacy):
    import torch
    import pytest
    from cmgm_c4.verification import train_val_guard
    torch.set_num_threads(2)
    # Exclude the old report-format test that intentionally reads historical
    # TEST metric JSONs. All other tests run inside the TEST access sentinel.
    with train_val_guard():
        if legacy:
            return pytest.main(['-q',str(ROOT/'tests'),'-k','not test_summary_uses_historical_json_not_test_arrays and not test_report_uses_split_matched_c4_and_baseline_denominator'])
        return pytest.main(['-q',str(ROOT/'tests/test_progressive_ablation.py')])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    parser.add_argument('--legacy',action='store_true',help='Run the original 13 rows with seed 42')
    parser.add_argument('--seeds',nargs='+',type=int,help='Paired seeds (default 42)')
    parser.add_argument('--sensitivity',action='store_true',help='Also compare D and E with the original optimizer settings')
    parser.add_argument('--diagnose-full',type=Path,metavar='MODEL_DIR',help='Read existing Full TRAIN/VAL prediction caches')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.diagnose_full:
        import json
        from cmgm_c4.training.progressive import routing_diagnostic
        print(json.dumps(routing_diagnostic(args.diagnose_full),indent=2))
        return
    if args.legacy and (args.seeds or args.sensitivity):
        parser.error('--seeds and --sensitivity apply to A–E')
    if args.preflight_child or not args.execute:return preflight(args.legacy)
    if args.output is None:parser.error('--execute requires --output')
    if not args.legacy:
        import torch
        from cmgm_c4.training.progressive import run
        status=preflight(False)
        if status:return status
        torch.set_num_threads(4)
        run(args.output,args.device,args.resume,args.allow_test,args.seeds or [42],args.sensitivity)
        return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    check=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--preflight-child']+(['--legacy'] if args.legacy else []),cwd=ROOT,capture_output=True,text=True)
    with (args.output.parent/(args.output.name+'_preflight.txt')).open('a' if args.resume else 'x') as f:f.write(check.stdout+check.stderr)
    print(check.stdout+check.stderr,flush=True)
    if check.returncode:return check.returncode
    import torch
    torch.set_num_threads(4)
    from cmgm_c4.training.ablation_suite import run_suite
    run_suite(args.output,args.device,args.resume,args.allow_test)


if __name__=='__main__':raise SystemExit(main())

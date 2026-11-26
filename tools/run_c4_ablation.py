"""C4 branch ablations. Use --progressive for A–E or --legacy for the historical Full+12 suite."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def diagnose_run(out):
    """Compare the startup protocol, per-model training records and current checkout."""
    from cmgm_c4.common import project_root, read, runtime_hashes, sha

    protocol = read(out/'PROTOCOL.json')
    differences = []

    def compare(label, expected, actual):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(expected.keys() | actual.keys()):
                compare(f'{label}.{key}', expected.get(key, '<missing>'), actual.get(key, '<missing>'))
        elif expected != actual:
            differences.append(f'{label}: expected {expected!r}, found {actual!r}')

    compare('current.source_hashes', protocol['source_hashes'], runtime_hashes())
    compare('current.project_sha256', protocol['project_sha256'], sha(project_root()/'configs/project.json'))
    for profile, settings in protocol['profiles'].items():
        for seed in protocol['seeds']:
            for key in settings['models']:
                rel = Path(profile)/f'seed{seed}'/'models'/key
                folder = out/rel
                if (folder/'COMPLETE.json').is_file():
                    signature = read(folder/'COMPLETE.json')['signature']
                    status = 'complete'
                elif (folder/'config.json').is_file():
                    signature = read(folder/'config.json')
                    status = 'incomplete'
                else:
                    print(rel, 'no training record')
                    continue
                expected = dict(identifier=protocol.get('family', 'progressive')+'/'+key, protocol=dict(protocol['training'], seed=seed,
                                lr=settings['lr'], weight_decay=settings['weight_decay']),
                                source_hashes=protocol['source_hashes'], dataset_sha256=protocol['dataset_sha256'],
                                selector=protocol['selector'])
                count = len(differences)
                for field, value in expected.items():
                    compare(f'{rel}.{field}', value, signature.get(field, '<missing>'))
                print(rel, status, 'MISMATCH' if len(differences) > count else 'OK')
    for line in differences:
        print(line)
    if not differences:
        print('Recorded signature fields and current source/config match the run protocol.')
    return int(bool(differences))


def preflight(legacy, branches=False):
    import torch
    from cmgm_c4.verification import train_val_guard
    torch.set_num_threads(2)
    # Exclude the old report-format test that intentionally reads historical
    # TEST metric JSONs. All other tests run inside the TEST access sentinel.
    with train_val_guard():
        if branches:
            import unittest
            tests = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_branch_ablation.py')
            return int(not unittest.TextTestRunner().run(tests).wasSuccessful())
        import pytest
        if legacy:
            return pytest.main(['-q',str(ROOT/'tests'),'-k','not test_summary_uses_historical_json_not_test_arrays and not test_report_uses_split_matched_c4_and_baseline_denominator'])
        return pytest.main(['-q',str(ROOT/'tests/test_progressive_ablation.py')])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    suites = parser.add_mutually_exclusive_group()
    suites.add_argument('--progressive', action='store_true', help='Run the historical A–E suite')
    suites.add_argument('--legacy',action='store_true',help='Run the original 13 rows with seed 42')
    parser.add_argument('--seeds',nargs='+',type=int,help='Paired seeds (default 42)')
    parser.add_argument('--sensitivity',action='store_true',help='Also compare D and E with the original optimizer settings')
    parser.add_argument('--diagnose-full',type=Path,metavar='MODEL_DIR',help='Read existing Full TRAIN/VAL prediction caches')
    parser.add_argument('--diagnose-run',type=Path,metavar='RUN_DIR',help='Compare saved training signatures without starting training')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.diagnose_run:
        if not (args.diagnose_run/'PROTOCOL.json').is_file():
            parser.error('--diagnose-run requires a run directory containing PROTOCOL.json')
        return diagnose_run(args.diagnose_run)
    if args.diagnose_full:
        import json
        from cmgm_c4.training.progressive import routing_diagnostic
        print(json.dumps(routing_diagnostic(args.diagnose_full),indent=2))
        return
    if args.legacy and (args.seeds or args.sensitivity):
        parser.error('--seeds and --sensitivity apply to A–E')
    branches = not (args.legacy or args.progressive or args.sensitivity)
    if args.resume and args.output and (args.output/'PROTOCOL.json').is_file():
        import json
        previous = json.loads((args.output/'PROTOCOL.json').read_text())
        branches = previous.get('family') == 'c4v2'
        if branches and (args.progressive or args.sensitivity or args.legacy):
            parser.error('The existing directory contains the four-row C4 branch experiment')
    if args.preflight_child or not args.execute:return preflight(args.legacy, branches)
    if args.output is None:parser.error('--execute requires --output')
    if not args.legacy:
        import torch
        from cmgm_c4.training.progressive import run
        status=preflight(False, branches)
        if status:return status
        torch.set_num_threads(4)
        options = {'branches': True} if branches else {}
        run(args.output,args.device,args.resume,args.allow_test,args.seeds or [42],args.sensitivity,**options)
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

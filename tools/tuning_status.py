"""Read-only progress view; does not import PyTorch or load any dataset."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    root=args.folder
    progress=root/'PROGRESS.json'
    completed=json.loads(progress.read_text()) if progress.exists() else dict(completed=0,total=9,rows=[])
    records=[]
    for folder in sorted((root/'trials').glob('T*')):
        path=folder/'history.json'
        if not path.exists():continue
        history=json.loads(path.read_text());best=min(history,key=lambda r:r['VAL_5D_COMMODITY_MACRO_MAE'])
        records.append(dict(trial=folder.name,epochs=len(history),last_epoch=history[-1]['epoch'],best_epoch=best['epoch'],
            VAL5=best['VAL_5D_COMMODITY_MACRO_MAE'],last_seconds=history[-1]['seconds'],
            status='COMPLETE' if (folder/'COMPLETE.json').exists() else 'RUNNING'))
    print(json.dumps(dict(completed=completed['completed'],total=9,trials=records,
        TEST_opened=(root/'TEST_OPEN_AUTHORIZATION.json').exists(),done=(root/'COMPLETE.json').exists()),indent=2))


if __name__=='__main__':main()

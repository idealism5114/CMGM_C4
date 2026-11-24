"""Summarize existing metric JSONs, without loading any TEST numeric arrays."""
import csv
from pathlib import Path
from cmgm_c4.common import project_root,read,save


def c4_improvement_pct(baseline, c4, *, higher_is_better=False):
    """Relative gain with the baseline metric as denominator."""
    if baseline == 0:
        return None
    gain = c4 - baseline if higher_is_better else baseline - c4
    return 100 * (gain / baseline)


def horizon_improvements(baseline, c4):
    return {
        f'c4_improvement_pct_{metric}': c4_improvement_pct(
            baseline[metric], c4[metric], higher_is_better=metric == 'Hit')
        for metric in ('commodity_macro_MAE', 'pooled_MAE', 'MSE', 'RMSE', 'Hit', 'Huber')
    }


def summarize(output=None,runs_dir=None):
    root=project_root();out=Path(output) if output else root/'reports/existing_experiments'
    if out.exists():raise ValueError('Summary output must be new; do not overwrite old reports')
    registry=read(root/'configs/registry.json');rows=[]
    def display_path(p):return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    if runs_dir:
        runs_dir=Path(runs_dir).resolve();registry={}
        for frozen in sorted(runs_dir.rglob('models_frozen.json')):
            if not (frozen.parent/'COMPLETE.json').exists():continue
            meta=read(frozen);identifier=meta['identifier'];key=identifier+'@'+str(frozen.parent.relative_to(runs_dir))
            registry[key]=dict(suite=identifier.split('/')[0],results=str(frozen.parent),best_epoch=meta['best_epoch'],
                selection=meta['selection'],checkpoint_sha256=meta['checkpoint_sha256'])
        if not registry:raise ValueError('No completed runs found')
    for identifier,r in registry.items():
        folder=root/r['results'];base=folder/('train_val_metrics.json' if runs_dir or r['suite']=='baselines' else 'metrics.json')
        scores=read(base)
        # These are historical saved JSON results, never a TEST inference call.
        candidates=('test_metrics.json','TEST_result.json','result.json')
        for name in candidates:
            if (folder/name).exists():
                value=read(folder/name);scores['test']=value.get('metrics',value);break
        for split,metrics in scores.items():
            if split not in ('train','val','test'):continue
            row=dict(model=identifier,suite=r['suite'],split=split,best_epoch=r['best_epoch'],
                selector=r['selection'],checkpoint_sha256=r['checkpoint_sha256'],
                metrics_source=display_path(base) if split!='test' else display_path(folder/name),
                role='historical_saved_metric; no new inference' if split=='test' else 'historical_saved_metric')
            for h in ('1','5','10','20'):
                row[f'MAE_{h}d']=metrics['horizons'][h]['commodity_macro_MAE']
            row['RMSE_5d']=metrics['horizons']['5']['RMSE'];row['Hit_5d']=metrics['horizons']['5']['Hit']
            rows.append(row)
    references = [identifier for identifier in registry if identifier.split('@', 1)[0] == 'c4/full']
    if runs_dir and not references:
        references = [identifier for identifier in registry if identifier.startswith('ablations/full_c4@')]
    reference = references[0] if len(references) == 1 else None
    reference_by_split = {row['split']: row for row in rows if row['model'] == reference}
    for row in rows:
        c4 = reference_by_split.get(row['split'])
        row['c4_reference'] = c4['model'] if c4 and row['suite'] == 'baselines' else None
        for metric in ('MAE_1d', 'MAE_5d', 'MAE_10d', 'MAE_20d', 'RMSE_5d', 'Hit_5d'):
            row[f'c4_improvement_pct_{metric}'] = (
                c4_improvement_pct(row[metric], c4[metric], higher_is_better=metric == 'Hit_5d')
                if row['c4_reference'] else None
            )
    out.mkdir(parents=True)
    for name,subset in [('ALL_RESULTS.csv',rows)]+[(split.upper()+'_RANKING.csv',sorted([r for r in rows if r['split']==split],key=lambda r:r['MAE_5d'])) for split in ('val','test')]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(subset)
    save(out/'provenance.json',dict(source='Completed new-run metric JSON' if runs_dir else 'Copied frozen historical metric JSON only',models=len(registry),
        TEST_numeric_arrays_loaded=0,TEST_inference=0,training=0,
        c4_reference=reference, c4_improvement_denominator='baseline metric for the same split and horizon; zero gives null',
        warning='Historical TEST was repeatedly viewed. Baselines use compressed 27-token input, C4 uses 311 nodes. Full C4 is also an ablation row.'))
    print('Existing results summarized:',out)
    return rows

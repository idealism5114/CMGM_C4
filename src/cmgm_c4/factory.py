"""Only original C4, eight latest Deep V3 baselines and thirteen ablation rows."""
import torch
from cmgm_c4.common import project_root,read,sha,seed_all
from cmgm_c4.models.c4 import C4Model
from cmgm_c4.models.ablations import C4Ablation
from cmgm_c4.models.horizon import V2_MODELS, HorizonC4
from cmgm_c4.models.progressive import ROWS, configure
from cmgm_c4.baselines.adapter import prepared_model


def build(identifier='c4v2/full',initial=False,device='cpu',*,seed=42):
    root=project_root();cfg=read(root/'configs/project.json');registry=read(root/'configs/registry.json')
    progressive = identifier in {'progressive/'+row for row in ROWS}
    if identifier in {'c4v2/'+row for row in V2_MODELS}:
        seed_all(seed)
        initial_state = build('c4/full', initial=True).state_dict() if initial and seed == 42 else None
        seed_all(seed)
        return HorizonC4(identifier.split('/')[1], cfg['shape'], cfg['inactive_parameters'], initial_state).to(device)
    if identifier not in registry and not progressive:raise ValueError('Unknown stable experiment: '+identifier)
    suite,key=identifier.split('/');seed_all(seed)
    if progressive:
        model = C4Ablation('no_temporal' if key=='A' else 'full_c4', **cfg['shape'])
        if initial and seed==42:
            model.load_state_dict(build('c4/full',initial=True).state_dict(),strict=True)
        return configure(model,key,cfg['inactive_parameters']).to(device)
    if suite=='c4':m=C4Model(**cfg['shape'])
    elif suite=='baselines':m=prepared_model(key,commodities=23,features=26)
    else:m=C4Ablation(key,**cfg['shape'])
    if initial:
        initial_path=root/'initial_states'/('c4.pt' if suite=='c4' else suite+'.pt')
        if suite=='c4':expected=read(root/'provenance/initial_states.json')['destination_sha256']
        else:
            rows=read(root/'provenance/migration_manifest.json')['files']
            expected=next(r['sha256'] for r in rows if r['destination']==str(initial_path.relative_to(root)))
        if sha(initial_path)!=expected:raise ValueError('Archived fresh initialization changed')
        a=torch.load(initial_path,map_location='cpu',weights_only=True)
        m.load_state_dict(a['C4' if suite=='c4' else key],strict=True)
    if suite=='ablations':m.freeze_inactive(cfg['inactive_parameters'][key])
    return m.to(device)


def restore(identifier='c4/full',which='best',device='cpu'):
    root=project_root();row=read(root/'configs/registry.json')[identifier]
    path=root/row['checkpoint' if which=='best' else 'latest']
    expected=row['checkpoint_sha256' if which=='best' else 'latest_sha256']
    if sha(path)!=expected:raise ValueError('Archived checkpoint changed')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp.get('selection',cp.get('selector'))!='VAL_5D_COMMODITY_MACRO_MAE':raise ValueError('Unexpected checkpoint selection')
    m=build(identifier,device=device);m.load_state_dict(cp['model_state_dict'],strict=True)
    if hasattr(m,'set_epoch'):m.set_epoch(cp['epoch'])
    return m,cp


def restore_run(folder,device='cpu'):
    """Restore only a completed, explicitly trained new run; no epoch selection."""
    from pathlib import Path
    folder=Path(folder);r=read(folder/'models_frozen.json');path=folder/r['checkpoint']
    if r['selection']!='VAL_5D_COMMODITY_MACRO_MAE' or sha(path)!=r['checkpoint_sha256']:
        raise ValueError('New-run checkpoint identity mismatch')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['epoch']!=r['best_epoch'] or cp['identifier']!=r['identifier']:raise ValueError('New-run epoch/model mismatch')
    m=build(r['identifier'],device=device);m.load_state_dict(cp['model_state_dict'],strict=True)
    if hasattr(m,'set_epoch'):m.set_epoch(cp['epoch'])
    return m,cp,r,path


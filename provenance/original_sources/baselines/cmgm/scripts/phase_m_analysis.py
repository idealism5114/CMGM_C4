"""Offline, float64 TRAIN/VAL diagnostics for Phase M. No fitting or TEST API."""
from itertools import combinations
import numpy as np

HORIZONS=(1,5,10,20)
EXPERTS=('recent_change','long_memory','spatial_structure','st_interaction')


def population(prediction,target,mask=None):
    p,y=np.asarray(prediction),np.asarray(target)
    if p.shape!=y.shape or p.ndim!=3 or p.shape[1]!=4:raise ValueError('Expected matching N,H,C')
    valid=np.ones_like(p,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    result={};commodities=[]
    for hi,h in enumerate(HORIZONS):
        per=[];per_h=[]
        for c in range(p.shape[2]):
            m=valid[:,hi,c]
            if not m.any():raise ValueError(f'No valid VAL labels for commodity {c}, horizon{h}')
            a,b=p[:,hi,c][m],y[:,hi,c][m]
            e=a.astype('float64')-b.astype('float64')
            if not np.isfinite(e).all():raise ValueError('Nonfinite valid predictions/labels')
            value=dict(horizon=h,commodity_index=c,MAE=float(np.abs(e).mean()),MSE=float(np.mean(e*e)),
                RMSE=float(np.sqrt(np.mean(e*e))),Hit=float(np.mean(np.sign(a)==np.sign(b))),valid_labels=int(m.sum()))
            per.append(value['MAE']);per_h.append(value);commodities.append(value)
        m=valid[:,hi];a,b=p[:,hi][m],y[:,hi][m]
        e=a.astype('float64')-b.astype('float64');ab=np.abs(e);mse=float(np.mean(e*e))
        result[str(h)]=dict(commodity_macro_MAE=float(np.mean(per)),pooled_MAE=float(ab.mean()),
            MSE=mse,RMSE=float(np.sqrt(mse)),Hit=float(np.mean(np.sign(a)==np.sign(b))),
            Huber=float(np.where(ab<=.02,.5*e*e,.02*(ab-.01)).mean()),valid_labels=int(m.sum()))
    return dict(horizons=result,per_commodity=commodities,
                four_horizon_prediction_Huber_sum=sum(v['Huber'] for v in result.values()),
                primary=result['5']['commodity_macro_MAE'],primary_name='VAL_5D_COMMODITY_MACRO_MAE',
                target_unit='decimal cumulative return',Hit_unit='fraction; display100x percent')


def entropy(weights):
    w=np.asarray(weights,dtype='float64')
    return -(w*np.log(np.maximum(w,1e-12))).sum(-1)


def js_divergence(a,b):
    a,b=np.asarray(a,dtype='float64'),np.asarray(b,dtype='float64');m=.5*(a+b)
    return .5*((a*np.log(np.maximum(a,1e-12)/np.maximum(m,1e-12))).sum(-1)
               +(b*np.log(np.maximum(b,1e-12)/np.maximum(m,1e-12))).sum(-1))


def routing_rows(model,weights):
    w=np.asarray(weights,dtype='float64')
    if w.ndim!=3 or w.shape[1:]!=(4,4):raise ValueError('Expected N,H,K routes')
    rows=[];differences=[]
    for hi,h in enumerate(HORIZONS):
        mean=w[:,hi].mean(0);sd=w[:,hi].std(0)
        row=dict(Model=model,Horizon=h,mean_sample_entropy_nats=float(entropy(w[:,hi]).mean()),
            entropy_of_mean_weights_nats=float(entropy(mean)),mean_max_expert_weight=float(w[:,hi].max(-1).mean()))
        for k,name in enumerate(EXPERTS):
            row[name]=float(mean[k]);row[name+'_std']=float(sd[k])
            for q in (.1,.5,.9):row[name+f'_q{q}']=float(np.quantile(w[:,hi,k],q))
        rows.append(row)
    for i,j in combinations(range(4),2):
        differences.append(dict(Model=model,Horizon_A=HORIZONS[i],Horizon_B=HORIZONS[j],
            JS_mean_routes_nats=float(js_divergence(w[:,i].mean(0),w[:,j].mean(0))),
            mean_per_sample_JS_nats=float(js_divergence(w[:,i],w[:,j]).mean()),
            mean_route_L1=float(np.abs(w[:,i]-w[:,j]).sum(-1).mean())))
    return rows,differences


def correlation(a,b):
    a,b=np.asarray(a,dtype='float64').ravel(),np.asarray(b,dtype='float64').ravel()
    return None if min(a.std(),b.std())<=1e-12 else float(np.corrcoef(a,b)[0,1])


def expert_competence(model,candidates,target,commodities,mask=None):
    candidates=np.asarray(candidates)
    if candidates.shape!=(len(target),4,4,len(commodities)):raise ValueError('Expected N,H,K,C candidates')
    rows=[];pairs=[]
    for k,name in enumerate(EXPERTS):
        r=population(candidates[:,:,k],target,mask)
        for h,v in r['horizons'].items():
            rows.append(dict(Model=model,Expert=name,commodity='ALL_MACRO',horizon=int(h),
                MAE=v['commodity_macro_MAE'],MSE=v['MSE'],RMSE=v['RMSE'],Hit=v['Hit'],valid_labels=v['valid_labels']))
        for item in r['per_commodity']:
            rows.append(dict(Model=model,Expert=name,commodity=commodities[item['commodity_index']],**item))
    for hidx,h in enumerate(HORIZONS):
        for i,j in combinations(range(4),2):
            valid=np.ones_like(target[:,hidx],bool) if mask is None else mask[:,hidx]
            a,b=candidates[:,hidx,i][valid],candidates[:,hidx,j][valid];y=target[:,hidx][valid]
            pairs.append(dict(Model=model,Horizon=h,Expert_A=EXPERTS[i],Expert_B=EXPERTS[j],
                prediction_correlation=correlation(a,b),error_correlation=correlation(a-y,b-y),
                mean_absolute_prediction_disagreement=float(np.mean(np.abs(a.astype('float64')-b.astype('float64'))))))
    return rows,pairs


def intervention_routes(train_routes,val_routes,seed=42):
    train=np.asarray(train_routes);val=np.asarray(val_routes)
    operations={'train_mean':np.broadcast_to(train.mean(0),val.shape).copy(),
        'paired_sample_shuffle':val[np.random.default_rng(seed).permutation(len(val))].copy(),
        'uniform':np.full_like(val,.25)}
    for first,second in ((0,3),(1,2)):
        order=list(range(4));order[first],order[second]=order[second],order[first]
        operations[f'swap_{HORIZONS[first]}d_{HORIZONS[second]}d']=val[:,order].copy()
    # Force exactly one horizon; all other horizon routes stay native.
    for hi,h in enumerate(HORIZONS):
        for k,name in enumerate(EXPERTS):
            w=val.copy();w[:,hi]=0;w[:,hi,k]=1
            operations[f'force_{h}d_{name}']=w
    return operations


def moving_block_bootstrap(candidate,reference,target,block=40,repeats=2000,seed=42):
    """Paired origins; whole same-day commodity vector, no independent resampling."""
    if block<20:raise ValueError('Block must cover at least the longest20-session horizon')
    ca=np.abs(np.asarray(candidate,dtype='float64')-target)
    re=np.abs(np.asarray(reference,dtype='float64')-target)
    n=len(ca)
    if n<block:raise ValueError('Not enough origins for the preregistered block')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        indices=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        values.append((ca[indices].mean(0)-re[indices].mean(0)).mean(-1))
    values=np.asarray(values)
    return dict(block_origins=block,repeats=repeats,seed=seed,method='paired noncircular moving blocks; all commodities grouped by origin',
        delta_candidate_minus_reference_CI95={str(h):np.quantile(values[:,hi],[.025,.975]).tolist() for hi,h in enumerate(HORIZONS)},
        interpretation='Exploratory uncertainty on one seed and previously used VAL, not significance or confirmatory evidence')

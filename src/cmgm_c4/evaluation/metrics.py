"""Original float64 C4 metric definitions; no TEST or fitting side effects."""
import numpy as np

HORIZONS=(1,5,10,20)


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

"""Standalone figures from completed saved metrics; never fits a model."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    if not (folder/'COMPLETE.json').exists():raise ValueError('Wait for all preregistered trials')
    lock=json.loads((folder/'SELECTED_MODEL.json').read_text());rows=lock['grid_results'];dest=folder/'figures';dest.mkdir(exist_ok=True)
    lrs=[3e-5,1e-4,3e-4];wds=[1e-6,1e-5,1e-4]
    matrix=np.empty((3,3));labels={}
    for r in rows:
        i,j=lrs.index(r['lr']),wds.index(r['weight_decay']);matrix[i,j]=r['VAL5_macro_MAE'];labels[i,j]=r['trial']
    fig,ax=plt.subplots(figsize=(7.5,5.2));im=ax.imshow(matrix,cmap='viridis_r')
    ax.set_xticks(range(3),[f'{v:.0e}' for v in wds]);ax.set_yticks(range(3),[f'{v:.0e}' for v in lrs])
    ax.set(xlabel='Adam weight decay',ylabel='Initial learning rate',title='C4 Seed42: VAL5 Commodity Macro MAE')
    for (i,j),v in np.ndenumerate(matrix):
        ax.text(j,i,f'{labels[i,j]}\n{v:.8f}'+('\nVAL selected' if labels[i,j]==lock['winner'] else ''),ha='center',va='center',
            color='black' if (v-matrix.min())/(matrix.max()-matrix.min()+1e-12)<.55 else 'white',fontsize=10)
    fig.colorbar(im,ax=ax,label='MAE (decimal return; lower is better)');fig.tight_layout()
    for suffix in ('png','pdf','svg'):fig.savefig(dest/('VAL_grid.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for r in sorted(rows,key=lambda x:x['trial']):
        history=json.loads((folder/'trials'/r['trial']/'history.json').read_text());x=[v['epoch'] for v in history];y=[v['VAL_5D_COMMODITY_MACRO_MAE'] for v in history]
        for ax in axes:
            line,=ax.plot(x,y,label=f"{r['trial']} lr={r['lr']:.0e}, wd={r['weight_decay']:.0e}",lw=1.1)
            ax.plot(r['best_epoch'],r['VAL5_macro_MAE'],'.',ms=5,color=line.get_color())
            ax.set(xlabel='Epoch',ylabel='VAL5 Commodity Macro MAE');ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].set_title('Complete training histories')
    axes[1].set_title('Zoom around all trials\' best scores')
    axes[1].set_ylim(matrix.min()-.00003,matrix.max()+.00003)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.18,1,1))
    for suffix in ('png','pdf','svg'):fig.savefig(dest/('VAL_learning_curves.'+suffix),dpi=180)
    plt.close(fig)
    print('Figures saved to',dest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);plot(p.parse_args().folder)

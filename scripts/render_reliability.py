"""Render model-level ten-seed statistics without changing frozen evaluations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'dqn':'#4c78a8','cql':'#f58518','bc':'#54a24b'}


def read(path):
    return json.loads(path.read_text())


def render(root):
    for study in ['baseline','ablation','navigation_validation','navigation']:
        folder=root/study
        if not (folder/'summary.json').exists():
            continue
        result=read(folder/'summary.json')
        rows=result['aggregate'];runs=result['runs']
        keys={'baseline':['dataset','method'],'ablation':['dataset','method','alpha'],
              'navigation':['candidate'],'navigation_validation':['candidate']}[study]
        metrics=['success_rate','collision_rate','timeout_rate','return_mean',
                 'successful_steps_mean','stall_rate','max_stationary_steps_mean']
        fig,axes=plt.subplots(3,3,figsize=(15,13),layout='constrained')
        for ax,metric in zip(axes.flat,metrics):
            for i,row in enumerate(rows):
                stat=row[metric]
                if stat['mean'] is None:continue
                scale=100 if metric.endswith('_rate') else 1
                color=COLORS.get(row.get('method'),'#79504B')
                mean=stat['mean']*scale;ci=np.array(stat['ci95'])*scale
                ax.errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',capsize=3,color=color)
                values=[r['metrics'][metric]*scale for r in runs
                        if all(r[k]==row[k] for k in keys) and r['metrics'][metric] is not None]
                ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,color=color,alpha=.65)
            labels=['/'.join(str(r[k]) for k in keys) for r in rows]
            ax.set_xticks(range(len(rows)),labels,rotation=65,ha='right',fontsize=7)
            ax.set_title(metric.replace('_',' '));ax.grid(axis='y',alpha=.2)
            if metric.endswith('_rate'):ax.set_ylim(-15,115)
            if metric=='return_mean':ax.set_ylim(-1.2,1.2)
        for ax in list(axes.flat)[len(metrics):]:ax.set_visible(False)
        fig.suptitle('10 trained models per condition; mean and 95% Student t CI; dots = models')
        fig.savefig(folder/'metrics.png',dpi=160);plt.close(fig)
        # Per-seed curves expose failures and pairing, instead of only aggregate bars.
        groups=list(dict.fromkeys(r.get('dataset','navigation') for r in runs))
        fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),4.5),squeeze=False,layout='constrained')
        for ax,group in zip(axes.flat,groups):
            subsets={}
            for r in runs:
                if r.get('dataset','navigation')==group:
                    name=r.get('candidate',r.get('method',''))
                    if study=='ablation':name+=f" alpha={r['alpha']}" if r['alpha'] is not None else ''
                    subsets.setdefault(name,[]).append(r)
            for name,members in subsets.items():
                members=sorted(members,key=lambda r:r['seed'])
                ax.plot([r['seed'] for r in members],[100*r['metrics']['success_rate'] for r in members],
                        marker='o',markersize=4,label=name,color=COLORS.get(name))
            ax.set(xlabel='Independent training seed',ylabel='Success (%)',title=group,ylim=(-5,105),xticks=range(10))
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(folder/'per_seed.png',dpi=160);plt.close(fig)
        # Machine-readable paired effects and a readable confidence-interval table.
        import csv
        contrasts=result['paired_differences']
        with (folder/'paired_differences.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['dataset','alpha','contrast','metric','n','mean','std','ci95_low','ci95_high'])
            for c in contrasts:
                for metric,stat in c['statistics'].items():
                    w.writerow([c.get('dataset'),c.get('alpha'),c['contrast'],metric,stat['n'],stat['mean'],stat['std'],*(stat['ci95'] or [None,None])])
        lines=['| Dataset | Alpha | Contrast | Mean success difference (pp) | 95% t CI (pp) |',
               '|---|---|---|---:|---|']
        for c in contrasts:
            s=c['statistics']['success_rate'];lo,hi=s['ci95']
            lines.append(f"| {c.get('dataset','navigation')} | {c.get('alpha','—')} | {c['contrast']} | {100*s['mean']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
        (folder/'paired_differences.md').write_text('\n'.join(lines)+'\n')
    # A fixed alpha axis makes the three data compositions easy to compare.
    p=root/'ablation/summary.json'
    if p.exists():
        r=read(p);fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,metric,scale in zip(axes,['success_rate','return_mean'],[100,1]):
            for dataset,color in zip(['strong','mixed_episode','mixed_balanced'],['#4c78a8','#f58518','#54a24b']):
                rows=[next(x for x in r['aggregate'] if x['dataset']==dataset and x['method']=='cql' and x['alpha']==a) for a in [0.,.1,1.,5.]]
                means=np.array([x[metric]['mean'] for x in rows])*scale
                ci=np.array([x[metric]['ci95'] for x in rows])*scale
                ax.errorbar(range(4),means,yerr=[means-ci[:,0],ci[:,1]-means],marker='o',capsize=4,color=color,label=dataset)
            ax.set_xticks(range(4),['0 (DQN)','0.1','1','5']);ax.set_xlabel('CQL alpha')
            ax.set_title(metric.replace('_',' '));ax.grid(alpha=.2)
            ax.set_ylim((-15,115) if scale==100 else (-1.2,1.2));ax.legend(fontsize=8)
        fig.suptitle('Mean and 95% Student t CI across 10 independent models')
        fig.savefig(root/'ablation/alpha_metrics.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,default=Path('reports/reliability'))
    render(parser.parse_args().reports)

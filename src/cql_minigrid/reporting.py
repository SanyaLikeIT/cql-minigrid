"""Aggregate over training seeds and publish compact, measured artifacts."""
import csv
from pathlib import Path
import shutil

import numpy as np

from .common import sha256, write_json

METRICS = ['success_rate','collision_rate','timeout_rate','return_mean','successful_steps_mean']


def aggregate(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run['dataset'],run['method']),[]).append(run)
    result = []
    for (dataset,method), members in sorted(groups.items()):
        seeds = [r['seed'] for r in members]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate training seed in an aggregate group')
        row = {'dataset':dataset,'method':method,'training_seeds':sorted(seeds),'n':len(seeds)}
        for metric in METRICS:
            values = [r['metrics'][metric] for r in members if r['metrics'][metric] is not None]
            row[metric] = {'mean':float(np.mean(values)) if values else None,
                           'std':float(np.std(values,ddof=1 if len(values)>1 else 0)) if values else None,
                           'valid_seeds':len(values)}
        result.append(row)
    return result


def results_table(rows):
    lines = ['| Dataset | Method | Success % | Collision % | Return |',
             '|---|---|---:|---:|---:|']
    order = {'strong':0,'mixed':1,'dqn':0,'cql':1,'bc':2}
    for row in sorted(rows,key=lambda x:(order[x['dataset']],order[x['method']])):
        fields=[]
        for metric,scale,precision in [('success_rate',100,1),('collision_rate',100,1),('return_mean',1,3)]:
            stat=row[metric]
            fields.append(f"{scale*stat['mean']:.{precision}f} ± {scale*stat['std']:.{precision}f}")
        lines.append(f"| {row['dataset']} | {row['method'].upper()} | " + ' | '.join(fields) + ' |')
    return '\n'.join(lines)


def write_reports(result,root,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    reports.mkdir(parents=True,exist_ok=True)
    write_json(reports/'summary.json',result)
    fields=['dataset','method','seed','updates',*METRICS,'training_seconds','checkpoint_sha256','dataset_sha256']
    with (reports/'per_run.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for run in result['runs']:
            writer.writerow({key:run['metrics'][key] if key in METRICS else run[key] for key in fields})
    colors={'dqn':'#4c78a8','cql':'#f58518','bc':'#54a24b'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    datasets=list(result['datasets'])
    for ax,metric,title in zip(axes,['success_rate','collision_rate','return_mean'],
                                ['Success rate','Collision rate','Return'],strict=True):
        for i,method in enumerate(['dqn','cql','bc']):
            rows=[next(r for r in result['aggregate'] if r['dataset']==d and r['method']==method) for d in datasets]
            x=np.arange(len(datasets))+(i-1)*.24
            ax.bar(x,[r[metric]['mean'] for r in rows],width=.22,
                   yerr=[r[metric]['std'] for r in rows],capsize=3,label=method.upper(),color=colors[method])
            for center,dataset in zip(x,datasets,strict=True):
                values=[r['metrics'][metric] for r in result['runs'] if r['dataset']==dataset and r['method']==method]
                ax.scatter(center+np.linspace(-.04,.04,len(values)),values,s=12,color='#202020',zorder=3)
        ax.set_xticks(range(len(datasets)),datasets);ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if metric!='return_mean':ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    fig.suptitle('Mean ± sample SD over training seeds; dots show individual runs',fontsize=10)
    fig.savefig(reports/'metrics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,unit,title in zip(axes,['episodes','transitions'],['Episode composition','Transition composition'],strict=True):
        bottom=np.zeros(len(datasets))
        for source,color in [('strong','#4c78a8'),('early','#f58518'),('random','#bab0ac')]:
            values=[result['datasets'][d]['sources'].get(source,{}).get(unit,0)/result['datasets'][d][unit] for d in datasets]
            label={'strong':'final PPO','early':'early PPO','random':'random'}[source]
            ax.bar(datasets,values,bottom=bottom,label=label,color=color,width=.6)
            bottom+=values
        ax.set_title(title);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1.05)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    fig.savefig(reports/'dataset_sources.png',dpi=180);plt.close(fig)
    demo_dir=reports/'demos';demo_dir.mkdir(exist_ok=True)
    for path in (root/'replays').glob('*'):
        if path.suffix in ['.gif','.json']:
            shutil.copy2(path,demo_dir/path.name)
    (reports/'results_table.txt').write_text(results_table(result['aggregate'])+'\n')
    write_json(reports/'rendering.json', {'reporting_source_sha256':sha256(Path(__file__)),
               'input_results_sha256':sha256(root/'results.json'),
               'matplotlib_version':matplotlib.__version__})

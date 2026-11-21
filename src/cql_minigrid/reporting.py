"""Aggregate the proposal metrics and publish compact reports and gameplay."""
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
    lines = ['| Dataset | Method | Success % | Collision % | Timeout % | Return | Successful steps |',
             '|---|---|---:|---:|---:|---:|---:|']
    order = {'strong':0,'mixed':1,'dqn':0,'cql':1,'bc':2}
    for row in sorted(rows,key=lambda x:(order[x['dataset']],order[x['method']])):
        fields=[]
        for metric,scale,precision in [('success_rate',100,1),('collision_rate',100,1),
                ('timeout_rate',100,1),('return_mean',1,3),('successful_steps_mean',1,1)]:
            stat=row[metric]
            fields.append('—' if stat['mean'] is None else f"{scale*stat['mean']:.{precision}f} ± {scale*stat['std']:.{precision}f}")
        lines.append(f"| {row['dataset']} | {row['method'].upper()} | " + ' | '.join(fields) + ' |')
    return '\n'.join(lines)


def reference_table(references):
    lines=['| Collector reference | Episodes | Success % | Collision % | Timeout % | Return | Successful steps |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('final','PPO final'),('early','PPO early'),('random','Random')]:
        row=references[name]
        steps='—' if row['successful_steps_mean'] is None else f"{row['successful_steps_mean']:.1f}"
        lines.append(f"| {label} | {row['episodes']} | {100*row['success_rate']:.1f} | {100*row['collision_rate']:.1f} | {100*row['timeout_rate']:.1f} | {row['return_mean']:.3f} | {steps} |")
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
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for run in result['runs']:
            writer.writerow({key:run['metrics'][key] if key in METRICS else run[key] for key in fields})
    with (reports/'collector_reference.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['source','episodes',*METRICS],lineterminator='\n');writer.writeheader()
        for name,row in result['references'].items():writer.writerow({'source':name,**row})
    colors={'dqn':'#4c78a8','cql':'#f58518','bc':'#54a24b'}
    datasets=list(result['datasets'])

    def compare(metrics,titles,filename):
        fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),3.8),layout='constrained')
        for ax,metric,title in zip(np.atleast_1d(axes),metrics,titles,strict=True):
            any_points=False
            for i,method in enumerate(['dqn','cql','bc']):
                rows=[next(r for r in result['aggregate'] if r['dataset']==d and r['method']==method) for d in datasets]
                x=np.arange(len(datasets))+(i-1)*.24
                valid=[j for j,row in enumerate(rows) if row[metric]['mean'] is not None]
                if valid:
                    ax.bar(x[valid],[rows[j][metric]['mean'] for j in valid],width=.22,
                           yerr=[rows[j][metric]['std'] for j in valid],capsize=3,label=method.upper(),color=colors[method])
                    any_points=True
                for center,dataset in zip(x,datasets,strict=True):
                    values=[r['metrics'][metric] for r in result['runs'] if r['dataset']==dataset and r['method']==method and r['metrics'][metric] is not None]
                    jitter=np.linspace(-.04,.04,len(values)) if len(values)>1 else np.zeros(len(values))
                    ax.scatter(center+jitter,values,s=12,color='#202020',zorder=3)
            if not any_points:ax.text(.5,.5,'No successful episodes',ha='center',transform=ax.transAxes)
            ax.set_xticks(range(len(datasets)),datasets);ax.set_title(title)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if metric.endswith('_rate'):ax.yaxis.set_major_formatter(PercentFormatter(1))
        handles,labels=np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
        count=len(result['seeds']['offline_training'])
        fig.suptitle(f"Mean ± sample SD over {count} training seeds; dots show runs",fontsize=10)
        fig.savefig(reports/filename,dpi=180);plt.close(fig)

    compare(['success_rate','collision_rate','return_mean'],['Success rate','Collision rate','Return'],'metrics.png')
    compare(['timeout_rate','successful_steps_mean'],['Timeout rate','Mean steps on successful episodes'],'ending_metrics.png')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,unit,title in zip(axes,['episodes','transitions'],['Episode composition','Transition composition'],strict=True):
        bottom=np.zeros(len(datasets))
        for source,color in [('strong','#4c78a8'),('early','#f58518'),('random','#bab0ac')]:
            values=[result['datasets'][d]['sources'].get(source,{}).get(unit,0)/result['datasets'][d][unit] for d in datasets]
            label={'strong':'final PPO','early':'early PPO','random':'random'}[source]
            ax.bar(datasets,values,bottom=bottom,label=label,color=color,width=.6);bottom+=values
        ax.set_title(title);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1.05)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    fig.savefig(reports/'dataset_sources.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.3),layout='constrained')
    labels=['PPO final','PPO early','Random'];sources=['final','early','random'];left=np.zeros(3)
    for outcome,color in [('success','#54a24b'),('collision','#e45756'),('timeout','#bab0ac')]:
        values=[result['references'][name][outcome+'_rate'] for name in sources]
        ax.barh(labels,values,left=left,label=outcome,color=color);left+=values
    ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"Collector references: {len(result['seeds']['evaluation'])} common test seeds")
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(reports/'collector_reference.png',dpi=180);plt.close(fig)
    demo_dir=reports/'demos';demo_dir.mkdir(exist_ok=True)
    for path in (root/'replays').glob('*'):
        if path.suffix in ['.gif','.mp4','.json'] and '.tmp' not in path.stem:shutil.copy2(path,demo_dir/path.name)
    (reports/'results_table.txt').write_text(results_table(result['aggregate'])+'\n')
    (reports/'reference_table.txt').write_text(reference_table(result['references'])+'\n')
    write_json(reports/'rendering.json', {'reporting_source_sha256':sha256(Path(__file__)),
               'input_results_sha256':sha256(root/'results.json'),'matplotlib_version':matplotlib.__version__})

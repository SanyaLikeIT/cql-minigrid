"""A fixed-budget alpha sweep and transition-balanced mixture comparison."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from .common import resource_path, sha256, write_json
from .evaluation import evaluate
from .experiments import provenance, run_one
from .reporting import METRICS, aggregate


def validate_study(config, *, config_path=None):
    study = config['study']
    alphas = study['alphas']
    if (not alphas or any(not isinstance(a, (float, int)) or isinstance(a, bool)
                          or not math.isfinite(a) or a < 0 for a in alphas)
            or len(set(alphas)) != len(alphas)):
        raise ValueError('Study alphas must be finite, nonnegative, nonempty, and unique')
    seeds = json.loads(resource_path(config['seeds_file'], config_path).read_text())
    training, evaluation = seeds['offline_training'], seeds['evaluation']
    if not training or len(training) != len(set(training)):
        raise ValueError('Training seeds must be nonempty and unique')
    if not evaluation or len(evaluation) != len(set(evaluation)):
        raise ValueError('Evaluation seeds must be nonempty and unique')
    reserved = set(seeds['validation']) | set(range(config['ppo']['seed'], config['ppo']['seed']+config['ppo']['n_envs']))
    if set(evaluation) & reserved:
        raise ValueError('Evaluation seeds overlap validation or PPO')
    if not study['datasets'] or not set(study.get('bc_datasets', [])) <= set(study['datasets']):
        raise ValueError('Study datasets or BC dataset selection is invalid')
    datasets = {}
    for name, filename in study['datasets'].items():
        if Path(name).name != name or name in ['.', '..']:
            raise ValueError('Dataset names must be single path components')
        path = resource_path(filename, config_path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        target = config['datasets']['transitions']
        if metadata['target'] != target or not target <= metadata['transitions'] < target + 256:
            raise ValueError('Dataset transition budget differs from the study config')
        if sha256(path) != metadata['sha256']:
            raise ValueError('Frozen dataset checksum mismatch')
        if set(evaluation) & {e['seed'] for e in metadata['episode_records']}:
            raise ValueError('Collection and evaluation seeds overlap')
        datasets[name] = {'path': str(path), **{k: metadata[k] for k in
                         ['sha256','transitions','episodes','weights','sources','observation','source_sha256']},
                         'sampling_unit':metadata.get('sampling_unit','episodes')}
    return seeds, datasets


def run_ablation(config, root, *, reports=None, config_path=None):
    started = time.perf_counter()
    seeds, datasets = validate_study(config,config_path=config_path)
    root = Path(root)
    source = provenance()
    identity = {'config': config, 'seeds': seeds, 'dataset_sha256':{d:m['sha256'] for d,m in datasets.items()},
                'source_sha256': source['source_sha256']}
    path = root/'study_identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Study configuration, sources, datasets, or seeds changed on resume')
    write_json(path, identity)
    jobs = [(d,'cql',float(a),s) for d in datasets for a in config['study']['alphas']
            for s in seeds['offline_training']]
    jobs += [(d,'bc',None,s) for d in config['study'].get('bc_datasets',[])
             for s in seeds['offline_training']]
    runs = []
    for dataset, method, alpha, seed in jobs:
        offline = {**config['offline'], 'alpha':alpha if alpha is not None else config['offline']['alpha']}
        name = f'cql-alpha-{alpha:g}' if method == 'cql' else 'bc'
        demo = (dataset in config['study'].get('bc_datasets', []) and seed == seeds['offline_training'][0]
                and (method == 'bc' or alpha in [0.,1.]))
        # run_one names demonstrations by method; distinguish alpha policies too.
        run = run_one(root, dataset, method, seed, offline, seeds['evaluation'], False,
                      dataset_path=datasets[dataset]['path'],run_name=name)
        if demo:
            from .experiments import export_checkpoint_replays
            export_checkpoint_replays(root,f'{dataset}-{name}',method,run['checkpoint'],seeds['evaluation'][0],
                                      {'dataset':dataset,'training_seed':seed,'alpha':alpha})
        run['alpha'] = alpha
        runs.append(run)
        write_json(root/'study_progress.json', {'expected_runs':len(jobs),'runs':runs,
                                              'session_seconds':time.perf_counter()-started})
        print(json.dumps({'completed_runs':len(runs),'expected_runs':len(jobs),'dataset':dataset,
                          'method':method,'alpha':alpha,'seed':seed,'success_rate':run['metrics']['success_rate']}),flush=True)
    references = {}
    ppo_root=resource_path(config['study']['ppo_root'],config_path)
    for name, method, checkpoint in [('final','ppo',ppo_root/'final.zip'),
                                      ('early','ppo',ppo_root/'early.zip'),
                                      ('random','random',None)]:
        references[name] = evaluate(method,checkpoint,seeds['evaluation'],root/'references'/f'{name}.json')['summary']
    result = {'completed_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,'seeds':seeds,
              'workers':1,'datasets':datasets,'runs':runs,
              'aggregate':aggregate(runs,group_by=('dataset','method','alpha')),
              'references':references,'session_seconds':time.perf_counter()-started,'provenance':source}
    write_json(root/'study_results.json',result)
    if reports:
        write_ablation_reports(result, root, Path(reports))
    return result


def format_stat(row, metric):
    stat = row[metric]
    if stat['mean'] is None:
        return '—'
    scale, precision = (100,1) if metric.endswith('_rate') else (1,3 if metric=='return_mean' else 1)
    return f"{scale*stat['mean']:.{precision}f} ± {scale*stat['std']:.{precision}f}"


def write_ablation_reports(result, root, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    reports.mkdir(parents=True,exist_ok=True)
    write_json(reports/'summary.json',result)
    fields = ['dataset','method','alpha','seed','updates',*METRICS,'training_seconds',
              'checkpoint_sha256','dataset_sha256']
    with (reports/'per_run.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for run in result['runs']:
            writer.writerow({key:run['metrics'][key] if key in METRICS else run[key] for key in fields})
    table = ['| Dataset | Method | Alpha | Success % | Collision % | Timeout % | Return | Successful steps |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in result['aggregate']:
        alpha = '—' if row['alpha'] is None else f"{row['alpha']:g}"
        table.append(f"| {row['dataset']} | {row['method'].upper()} | {alpha} | " +
                     ' | '.join(format_stat(row,m) for m in METRICS)+' |')
    (reports/'results_table.md').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    titles=['Success rate','Collision rate','Timeout rate','Return','Steps on successful episodes']
    alphas=result['config']['study']['alphas'];x=np.arange(len(alphas))
    for ax,metric,title in zip(axes.flat,METRICS,titles):
        for dataset in result['datasets']:
            rows=[next(r for r in result['aggregate'] if r['dataset']==dataset and r['method']=='cql'
                       and r['alpha']==a) for a in alphas]
            valid=[i for i,row in enumerate(rows) if row[metric]['mean'] is not None]
            ax.errorbar(x[valid],[rows[i][metric]['mean'] for i in valid],
                        yerr=[rows[i][metric]['std'] for i in valid],marker='o',capsize=3,label=dataset)
        ax.set_xticks(x,[f'{a:g}' for a in alphas]);ax.set_xlabel('CQL alpha (0 = DQN)');ax.set_title(title)
        ax.grid(alpha=.2)
        if metric.endswith('_rate'):ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax=axes.flat[-1];bottom=np.zeros(len(result['datasets']))
    for source,label in [('strong','final PPO'),('early','early PPO'),('random','random')]:
        values=[d['sources'].get(source,{}).get('transitions',0)/d['transitions'] for d in result['datasets'].values()]
        ax.bar(list(result['datasets']),values,bottom=bottom,label=label);bottom+=values
    ax.set_title('Actual transition composition');ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='x',labelsize=8);ax.legend(loc='lower left',fontsize=8)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=len(labels),frameon=False)
    updates=result['config']['offline']['updates'];count=len(result['seeds']['offline_training'])
    fig.suptitle(f'Fixed {updates:,}-update alpha sweep: mean ± sample SD over {count} training seeds')
    fig.savefig(reports/'alpha_metrics.png',dpi=180);plt.close(fig)
    demo_dir=reports/'demos';demo_dir.mkdir(exist_ok=True)
    for path in (root/'replays').glob('*'):
        if path.suffix in ['.gif','.mp4','.json'] and '.tmp' not in path.stem:
            shutil.copy2(path,demo_dir/path.name)
    write_json(reports/'rendering.json',{'reporting_source_sha256':sha256(__file__),
               'input_results_sha256':sha256(root/'study_results.json'),'matplotlib_version':matplotlib.__version__})

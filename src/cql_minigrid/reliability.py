"""Ten-seed extension with immutable historical inputs and resumable jobs.

Run: python -m cql_minigrid.reliability
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from scipy.stats import t

from .common import seed_everything, sha256, write_json
from .evaluation import EVALUATION_VERSION, evaluate
from .experiments import provenance
from .improvement import ALL_METRICS, NUMERICAL_SOURCES, choose_candidate
from .offline import train_offline
from .rollout import summarize

METRICS = ALL_METRICS + ['max_stationary_steps_max', 'max_stationary_steps_median']
SEEDS = list(range(10))


def statistics(values):
    values = np.array([x for x in values if x is not None], dtype=float)
    n = len(values)
    if not n:
        return dict(n=0, mean=None, std=None, median=None, minimum=None, maximum=None, ci95=None)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * sd / math.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, std=sd, median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                ci95=[mean-half, mean+half] if half is not None else None)


def aggregate(runs, keys):
    groups = {}
    for run in runs:
        groups.setdefault(tuple(run[k] for k in keys), []).append(run)
    result = []
    for key, rows in groups.items():
        seeds = [r['seed'] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate experimental unit')
        result.append({**dict(zip(keys, key)), 'n':len(rows), 'training_seeds':sorted(seeds),
                       **{m:statistics([r['metrics'][m] for r in rows]) for m in METRICS}})
    return result


def paired(left, right):
    a, b = {r['seed']:r for r in left}, {r['seed']:r for r in right}
    if a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError('Paired comparison requires identical unique training seeds')
    return {m:statistics([a[s]['metrics'][m]-b[s]['metrics'][m] for s in sorted(a)
                         if a[s]['metrics'][m] is not None and b[s]['metrics'][m] is not None])
            for m in METRICS}


def read(path):
    return json.loads(Path(path).read_text())


def check_seeds(test, seeds, metadata):
    reserved = set(seeds['offline_training']) | set(seeds['validation']) | set(seeds['evaluation'])
    reserved |= set(range(30000, 30200)) | set(range(12, 20))
    for meta in metadata.values():
        reserved |= {e['seed'] for e in meta['episode_records']}
        reserved |= {meta['policy_seed_start']}
    if len(test) != 200 or len(set(test)) != 200 or reserved & set(test):
        raise ValueError('Fresh navigation test seeds overlap a reserved range')


def report(study, runs, keys, reports):
    output = reports / study
    rows = aggregate(runs, keys)
    contrasts = []
    if study == 'baseline':
        for dataset in ['strong', 'mixed']:
            for other in ['dqn', 'bc']:
                contrasts.append({'dataset':dataset, 'contrast':'cql - '+other,
                    'statistics':paired([r for r in runs if r['dataset']==dataset and r['method']=='cql'],
                                        [r for r in runs if r['dataset']==dataset and r['method']==other])})
    if study == 'ablation':
        for dataset in dict.fromkeys(r['dataset'] for r in runs):
            reference=[r for r in runs if r['dataset']==dataset and r['method']=='cql' and r['alpha']==0]
            bc=[r for r in runs if r['dataset']==dataset and r['method']=='bc']
            for alpha in [0., .1, 1., 5.]:
                left=[r for r in runs if r['dataset']==dataset and r['method']=='cql' and r['alpha']==alpha]
                for label,right in [('DQN (alpha=0)',reference),('BC',bc)]:
                    if right:
                        contrasts.append({'dataset':dataset,'alpha':alpha,'contrast':'CQL - '+label,
                                          'statistics':paired(left,right)})
    if study.startswith('navigation'):
        names = list(dict.fromkeys(r['candidate'] for r in runs))
        for name in names:
            if name != 'baseline':
                contrasts.append({'contrast':name+' - baseline',
                    'statistics':paired([r for r in runs if r['candidate']==name],
                                        [r for r in runs if r['candidate']=='baseline'])})
    result = {'training_seeds':SEEDS, 'runs':runs, 'aggregate':rows, 'paired_differences':contrasts,
              'experimental_unit':'independently trained model; episodes are repeated observations',
              'interval':'unadjusted two-sided 95% Student t interval across models; no clipping',
              'conditional_lengths':'exclude models without successes only for conditional length statistics'}
    write_json(output/'summary.json', result)
    fields = [*keys, 'seed', *METRICS, 'checkpoint', 'checkpoint_sha256', 'evaluation_file']
    with (output/'per_run.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for run in runs:writer.writerow({k:run['metrics'][k] if k in METRICS else run[k] for k in fields})
    with (output/'statistics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=[*keys,'metric','n','mean','std','median','minimum','maximum','ci95_low','ci95_high']);writer.writeheader()
        for row in rows:
            for metric in METRICS:
                stat=row[metric];ci=stat['ci95'] or [None,None]
                writer.writerow({**{k:row[k] for k in keys},'metric':metric,
                    **{k:stat[k] for k in ['n','mean','std','median','minimum','maximum']},
                    'ci95_low':ci[0],'ci95_high':ci[1]})
    table=['| Configuration | Mean success % | Sample SD % | 95% t CI % | Median % | Min–max % |',
           '|---|---:|---:|---|---:|---|']
    for row in rows:
        s=row['success_rate'];lo,hi=s['ci95']
        table.append('| '+' / '.join(str(row[k]) for k in keys)+f" | {s['mean']*100:.2f} | {s['std']*100:.2f} | [{lo*100:.2f}, {hi*100:.2f}] | {s['median']*100:.2f} | {s['minimum']*100:.1f}–{s['maximum']*100:.1f} |")
    (output/'results_table.md').write_text('\n'.join(table)+'\n')
    return result


def run(root, reports, archive=None):
    started=time.perf_counter()
    root, reports=Path(root),Path(reports)
    seed_everything(0)
    config=read('configs/main.json')['offline']
    seeds=read('configs/seeds.json');seeds['offline_training']=SEEDS
    datasets={name:Path(path) for name,path in {
        'strong':'artifacts/final/datasets/strong.npz', 'mixed':'artifacts/final/datasets/mixed.npz',
        'mixed_balanced':'artifacts/ablation/datasets/mixed_balanced.npz',
        'guided':'artifacts/navigation-guided/datasets/guided.npz'}.items()}
    metadata={name:read(p.with_suffix('.json')) for name,p in datasets.items()}
    for name,p in datasets.items():
        if sha256(p)!=metadata[name]['sha256']:raise ValueError('Frozen dataset changed: '+name)
    test=list(range(40000,40200));check_seeds(test,seeds,metadata)
    source=provenance()
    identity={'training_seeds':SEEDS,'validation':seeds['validation'],'evaluation':seeds['evaluation'],
              'fresh_navigation_test':test,'dataset_hashes':{n:m['sha256'] for n,m in metadata.items()},
              'offline':config,'numerical_sources':{k:v for k,v in source['source_sha256'].items()
                                                    if Path(k).name in NUMERICAL_SOURCES}}
    if (root/'identity.json').exists() and read(root/'identity.json')!=identity:
        raise ValueError('Numerical protocol or datasets changed on resume')
    write_json(root/'identity.json',identity)
    historical=[]
    for name in ['reports/summary.json','reports/ablation/summary.json','reports/navigation/summary.json',
                 'reports/navigation/controls/mixed_penalty.json','reports/navigation/controls/guided_without_features.json']:
        r=read(name)
        historical.extend(r.get('runs',r.get('validation_runs',[])))
    expected={r['checkpoint']:r['checkpoint_sha256'] for r in historical}
    inventory={r['checkpoint']:r for r in read(root/'checkpoint_inventory.json')} if (root/'checkpoint_inventory.json').exists() else {}
    baseline=[]; ablation=[]; navigation=[]
    def checkpoint(path, dataset, method, seed, cfg, initial=None):
        path=Path(path)
        if str(path) in expected and path.exists() and sha256(path)!=expected[str(path)]:
            raise ValueError('Historical checkpoint changed: '+str(path))
        wanted={'method':method,'seed':seed,'config':cfg,'dataset_sha256':metadata[dataset]['sha256']}
        if initial:wanted['initialization']={'sha256':sha256(initial),'updates':5000}
        existed=path.exists();done=False
        if existed:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['identity']!=wanted:raise ValueError('Incompatible checkpoint: '+str(path))
            done=saved['updates']==cfg['updates'];del saved
        if not done:
            train_offline(datasets[dataset],path,cfg,method=method,seed=seed,resume=existed,initialize_from=initial)
        inventory[str(path)]={'checkpoint':str(path),'sha256':sha256(path),'reused':done,
                              'method':method,'seed':seed,'dataset':dataset,
                              'newly_trained':inventory.get(str(path),{}).get('newly_trained',not done)}
        write_json(root/'checkpoint_inventory.json',list(inventory.values()))
        return path
    def evaluation(path, policy_method, env_seeds, output, **labels):
        output=Path(output);old=read(output) if output.exists() else None
        if not (old and old.get('checkpoint_sha256')==sha256(path) and old.get('seeds')==env_seeds
                and old.get('evaluation_version')==EVALUATION_VERSION):
            old=evaluate(policy_method,path,env_seeds,output)
        metrics=summarize(old['episodes'])
        if metrics!=old['summary']:raise ValueError('Evaluation aggregate mismatch')
        stationary=[e['max_stationary_steps'] for e in old['episodes']]
        metrics.update(max_stationary_steps_max=max(stationary),max_stationary_steps_median=float(np.median(stationary)))
        return {**labels,'checkpoint':str(path),'checkpoint_sha256':sha256(path),
                'evaluation_file':str(output),'metrics':metrics}
    def progress(phase,rows,total):
        write_json(root/(phase+'_progress.json'),{'runs':rows,'expected_runs':total,'seconds':time.perf_counter()-started})
        print(json.dumps({'phase':phase,'complete':len(rows),'expected':total,'last':{k:v for k,v in rows[-1].items() if k in ['seed','dataset','method','candidate','alpha']}}),flush=True)
    paths={}
    for dataset in ['strong','mixed']:
        for method in ['dqn','cql','bc']:
            for seed in SEEDS:
                old=Path(f'artifacts/final/runs/{dataset}/{method}/seed-{seed}.pt')
                path=old if old.exists() else root/'baseline'/dataset/method/f'seed-{seed}.pt'
                path=checkpoint(path,dataset,method,seed,config);paths[dataset,method,seed]=path
                baseline.append(evaluation(path,method,seeds['evaluation'],root/'baseline-evaluation'/dataset/method/f'seed-{seed}.json',dataset=dataset,method=method,seed=seed))
                progress('baseline',baseline,60)
    report('baseline',baseline,('dataset','method'),reports)
    for dataset,actual in [('strong','strong'),('mixed_episode','mixed'),('mixed_balanced','mixed_balanced')]:
        for method,alpha in [('cql',a) for a in [0.,.1,1.,5.]]+([('bc',None)] if actual=='mixed_balanced' else []):
            for seed in SEEDS:
                name=f'cql-alpha-{alpha:g}' if method=='cql' else 'bc'
                old=Path(f'artifacts/ablation/runs/{dataset}/{name}/seed-{seed}.pt')
                actual_method=method
                if old.exists():path=checkpoint(old,actual,method,seed,{**config,'alpha':alpha if alpha is not None else config['alpha']})
                elif actual in ['strong','mixed'] and alpha in [0.,1.]:
                    actual_method='dqn' if alpha==0 else 'cql';path=paths[actual,actual_method,seed]
                else:path=checkpoint(root/'ablation'/dataset/name/f'seed-{seed}.pt',actual,method,seed,{**config,'alpha':alpha if alpha is not None else config['alpha']})
                ablation.append(evaluation(path,actual_method,seeds['evaluation'],root/'ablation-evaluation'/dataset/name/f'seed-{seed}.json',dataset=dataset,method=method,alpha=alpha,seed=seed))
                progress('ablation',ablation,130)
    report('ablation',ablation,('dataset','method','alpha'),reports)
    for seed in SEEDS:
        path=paths['mixed','cql',seed]
        navigation.append(evaluation(path,'cql',seeds['validation'],root/'navigation/baseline'/f'seed-{seed}.validation.json',candidate='baseline',seed=seed))
    for file in ['navigation','navigation_guided','navigation_geometry']:
        c=read('configs/'+file+'.json')
        for candidate in c['improvement']['candidates']:
            name=candidate['name'];dataset='mixed' if file=='navigation' else 'guided'
            cfg={**c['offline'],**candidate.get('offline',{})}
            for seed in SEEDS:
                oldname={'navigation':'navigation-penalty','navigation_guided':'navigation-guided','navigation_geometry':'navigation-geometry'}[file]
                old=Path('artifacts')/oldname/'runs'/name/f'seed-{seed}.pt'
                restored=root/'restored'/oldname/name/old.name
                if not old.exists() and restored.exists():
                    old=restored
                if not old.exists() and archive:
                    archived=Path(archive)/'team_project/initial'/old
                    if archived.exists():
                        restored=root/'restored'/oldname/name/old.name
                        if not restored.exists():
                            if sha256(archived)!=expected[str(old)]:raise ValueError('Archived checkpoint mismatch')
                            restored.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archived,restored)
                        old=restored
                path=old if old.exists() else root/'navigation/runs'/name/f'seed-{seed}.pt'
                path=checkpoint(path,dataset,'cql',seed,cfg,paths['mixed','cql',seed])
                navigation.append(evaluation(path,'cql',seeds['validation'],root/'navigation/validation'/name/f'seed-{seed}.json',candidate=name,seed=seed))
                progress('navigation',navigation,90)
    validation=report('navigation_validation',navigation,('candidate',),reports)
    rows=validation['aggregate'];base=next(r for r in rows if r['candidate']=='baseline')
    selected=choose_candidate([r for r in rows if r['candidate']!='baseline'],base,
                              **read('configs/navigation_geometry.json')['improvement']['quality_tolerances'])
    if selected is None:raise ValueError('No eligible navigation candidate; fresh test unopened')
    selection={'candidate':selected['candidate'],'validation_aggregate':rows,'test_seeds':test}
    lock=root/'navigation/test_lock.json'
    if lock.exists() and read(lock)!=selection:raise ValueError('Selection already frozen')
    write_json(lock,selection)
    final=[]
    for row in navigation:
        if row['candidate'] in ['baseline',selected['candidate']]:
            final.append(evaluation(row['checkpoint'],'cql',test,root/'navigation/test'/row['candidate']/f"seed-{row['seed']}.json",candidate=row['candidate'],seed=row['seed']))
            progress('test',final,20)
    result=report('navigation',final,('candidate',),reports)
    representative=min([r for r in navigation if r['candidate']==selected['candidate']],
                       key=lambda r:(-r['metrics']['success_rate'],r['metrics']['stall_rate'],r['metrics']['successful_steps_mean'] or math.inf,r['seed']))
    best=root/'navigation/best.pt';shutil.copy2(representative['checkpoint'],best)
    write_json(root/'navigation/best.json',{**representative,'selected_on':'validation'})
    write_json(root/'navigation/test-seeds.json',{'evaluation':test})
    write_json(root/'navigation/evaluation-config.json',{'seeds_file':'test-seeds.json'})
    summary={'status':'complete','baseline_models':60,'ablation_models':130,'navigation_candidates':80,
             'baseline_evaluation_episodes':sum(r['metrics']['episodes'] for r in baseline),
             'ablation_evaluation_episodes':sum(r['metrics']['episodes'] for r in ablation),
             'navigation_validation_episodes':sum(r['metrics']['episodes'] for r in navigation),
             'fresh_navigation_test_episodes':sum(r['metrics']['episodes'] for r in final),
             'selected':selected['candidate'],'unique_checkpoints':len(inventory),
             'newly_trained_models':sum(r['newly_trained'] for r in inventory.values()),
             'reused_models':sum(not r['newly_trained'] for r in inventory.values()),
             'session_seconds':time.perf_counter()-started,'provenance':source,'identity':identity}
    write_json(root/'complete.json',summary);write_json(reports/'complete.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='artifacts/reliability')
    parser.add_argument('--reports',default='reports/reliability')
    parser.add_argument('--archive',help='Optional previous cleanup archive for historical control weights')
    args=parser.parse_args();run(args.root,args.reports,args.archive)

if __name__=='__main__':main()

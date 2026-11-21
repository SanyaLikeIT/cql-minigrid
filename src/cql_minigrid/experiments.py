"""Run/resume fixed-seed experiments; checkpoints and episode reports stay local."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time

from .collector import collect, train_ppo
from .common import sha256, write_json
from .evaluation import evaluate, export_gif, load_policy
from .offline import train_offline
from .reporting import aggregate, write_reports


def run_one(root, dataset, method, seed, config, evaluation_seeds, demo):
    root = Path(root)
    checkpoint = root / 'runs' / dataset / method / f'seed-{seed}.pt'
    train_offline(root / 'datasets' / f'{dataset}.npz', checkpoint, config,
                  method=method, seed=seed, resume=checkpoint.exists())
    report_path = checkpoint.with_suffix('.eval.json')
    checkpoint_hash = sha256(checkpoint)
    previous = json.loads(report_path.read_text()) if report_path.exists() else None
    started = time.perf_counter()
    if previous and previous['checkpoint_sha256'] == checkpoint_hash and previous['seeds'] == evaluation_seeds:
        report = previous
    else:
        report = evaluate(method, checkpoint, evaluation_seeds, report_path)
    if demo:
        gif = root / 'replays' / f'{dataset}-{method}.gif'
        gif_report = gif.with_suffix('.json')
        old = json.loads(gif_report.read_text()) if gif_report.exists() else {}
        if not gif.exists() or old.get('checkpoint_sha256') != checkpoint_hash:
            policy, _ = load_policy(method, checkpoint)
            replay = export_gif(policy, gif, evaluation_seeds[0])
            write_json(gif_report, {**replay, 'checkpoint_sha256': checkpoint_hash,
                                    'method': method, 'dataset': dataset, 'training_seed': seed})
    training = json.loads(checkpoint.with_suffix('.train.json').read_text())
    return {'dataset': dataset, 'method': method, 'seed': seed, 'metrics': report['summary'],
            'updates': training['updates'], 'training_seconds': training['seconds'],
            'evaluation_and_replay_seconds': time.perf_counter() - started,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
            'evaluation_file': str(report_path), 'dataset_sha256': training['identity']['dataset_sha256']}


def provenance():
    root = Path(__file__).parent
    return {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu',
            'threads_per_training_worker': 1,
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch','gymnasium','minigrid','stable-baselines3','numpy','matplotlib']},
            'base_git_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'source_sha256': {str(path.relative_to(Path.cwd())): sha256(path)
                              for path in sorted(root.glob('*.py'))}}


def run_matrix(config, root, *, workers=1, reports=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    started = time.perf_counter()
    root = Path(root)
    seeds = json.loads(Path(config['seeds_file']).read_text())
    if set(seeds['validation']) & set(seeds['evaluation']):
        raise ValueError('Validation and evaluation seeds overlap')
    if set(range(config['ppo']['seed'], config['ppo']['seed'] + config['ppo']['n_envs'])) & set(seeds['evaluation']):
        raise ValueError('PPO and evaluation seeds overlap')
    datasets = {}
    for name in config['datasets']['mixtures']:
        path = root / 'datasets' / f'{name}.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != metadata['sha256']:
            raise ValueError('Frozen dataset checksum mismatch')
        if set(seeds['evaluation']) & {e['seed'] for e in metadata['episode_records']}:
            raise ValueError('Collection and evaluation seeds overlap')
        datasets[name] = {k: metadata[k] for k in ['sha256','transitions','episodes','weights','sources',
            'observation','source_sha256','episode_seed_start','policy_seed_start']}
    for filename, value in [('experiment_config.json',config), ('experiment_seeds.json',seeds)]:
        path = root / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Experiment config or seeds changed on resume')
        write_json(path, value)
    training_seeds = seeds['offline_training']
    if not training_seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError('Training seeds must be nonempty and unique')
    jobs = [(str(root), dataset, method, seed, config['offline'], seeds['evaluation'], seed == training_seeds[0])
            for dataset in datasets for method in ['dqn','cql','bc'] for seed in training_seeds]
    runs = []
    if workers == 1:
        for job in jobs:
            runs.append(run_one(*job))
            write_json(root/'matrix_progress.json', {'expected_runs':len(jobs), 'runs':runs,
                       'session_seconds':time.perf_counter()-started})
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_one,*job) for job in jobs]
            for future in as_completed(futures):
                run = future.result(); runs.append(run)
                write_json(root/'matrix_progress.json', {'expected_runs':len(jobs), 'runs':runs,
                           'session_seconds':time.perf_counter()-started})
                print(json.dumps({'completed_runs':len(runs),'expected_runs':len(jobs),
                                  'dataset':run['dataset'],'method':run['method'],'seed':run['seed'],
                                  'success_rate':run['metrics']['success_rate']}),flush=True)
    runs.sort(key=lambda r:(r['dataset'],r['method'],r['seed']))
    references = {}
    for source, method, checkpoint in [('final','ppo',root/'ppo/final.zip'),
                                       ('early','ppo',root/'ppo/early.zip'), ('random','random',None)]:
        result = evaluate(method,checkpoint,seeds['evaluation'],root/'references'/f'{source}.json')
        references[source] = result['summary']
    source_audit = json.loads((root/'source_audit.json').read_text())
    validation_sources = {name: {'training_timesteps':source['training_timesteps'],
                                **source['summary']} for name,source in source_audit['sources'].items()}
    result = {'completed_at_utc':datetime.now(timezone.utc).isoformat(), 'config':config, 'seeds':seeds,
              'workers':workers, 'datasets':datasets, 'references':references, 'runs':runs,
              'aggregate':aggregate(runs), 'matrix_session_seconds':time.perf_counter()-started,
              'validation_sources':validation_sources,
              'ppo':json.loads((root/'ppo/progress.json').read_text()), 'provenance':provenance()}
    write_json(root/'results.json',result)
    if reports:
        write_reports(result,root,Path(reports))
    return result


def pipeline(config, root, *, workers=1, reports=None):
    started = time.perf_counter()
    root = Path(root)
    train_ppo(config['ppo'],root/'ppo',resume=(root/'ppo/latest.zip').exists())
    collect(config,root)
    result = run_matrix(config,root,workers=workers,reports=reports)
    write_json(root/'pipeline_timing.json', {'seconds':time.perf_counter()-started,
               'runs':len(result['runs']), 'root':str(root)})
    return result

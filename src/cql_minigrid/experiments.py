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
from .common import resource_path, sha256, write_json
from .evaluation import EVALUATION_VERSION, REPLAY_VERSION, evaluate, export_replay, load_policy
from .offline import train_offline
from .reporting import aggregate, write_reports


def export_checkpoint_replays(root, name, method, checkpoint, seed, extra):
    checkpoint_hash = sha256(checkpoint)
    policy = None
    for kind in ['gif', 'mp4']:
        path = Path(root) / 'replays' / f'{name}.{kind}'
        sidecar = path.with_suffix(path.suffix + '.json')
        old = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        valid = (path.exists() and old.get('checkpoint_sha256') == checkpoint_hash
                 and old.get('seed') == seed and old.get('replay_version') == REPLAY_VERSION
                 and old.get('file_sha256') == sha256(path))
        if not valid:
            if policy is None:
                policy, _ = load_policy(method, checkpoint)
            replay = export_replay(policy, path, seed)
            write_json(sidecar, {**replay, 'checkpoint':str(checkpoint),
                                'checkpoint_sha256':checkpoint_hash, 'method':method, **extra})


def run_one(root, dataset, method, seed, config, evaluation_seeds, demo, *, dataset_path=None, run_name=None):
    root = Path(root)
    checkpoint = root / 'runs' / dataset / (run_name or method) / f'seed-{seed}.pt'
    train_offline(dataset_path or root / 'datasets' / f'{dataset}.npz', checkpoint, config,
                  method=method, seed=seed, resume=checkpoint.exists())
    report_path = checkpoint.with_suffix('.eval.json')
    checkpoint_hash = sha256(checkpoint)
    previous = json.loads(report_path.read_text()) if report_path.exists() else None
    started = time.perf_counter()
    if (previous and previous['checkpoint_sha256'] == checkpoint_hash and previous['seeds'] == evaluation_seeds
            and previous.get('evaluation_version') == EVALUATION_VERSION):
        report = previous
    else:
        report = evaluate(method, checkpoint, evaluation_seeds, report_path)
    if demo:
        export_checkpoint_replays(root, f'{dataset}-{method}', method, checkpoint, evaluation_seeds[0],
                                  {'dataset':dataset, 'training_seed':seed})
    training = json.loads(checkpoint.with_suffix('.train.json').read_text())
    return {'dataset': dataset, 'method': method, 'seed': seed, 'metrics': report['summary'],
            'updates': training['updates'], 'training_seconds': training['seconds'],
            'evaluation_and_replay_seconds': time.perf_counter() - started,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
            'evaluation_file': str(report_path), 'dataset_sha256': training['identity']['dataset_sha256']}


def provenance():
    root = Path(__file__).resolve().parent
    project = root.parent.parent if root.parent.name == 'src' else root.parent
    revision, state, reason = None, 'unavailable', 'no_repository'
    # An extracted project may itself be inside an unrelated Git checkout.
    if (project / '.git').exists():
        def git(*args):
            return subprocess.check_output(['git', '-C', str(project), *args], text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        try:
            if Path(git('rev-parse', '--show-toplevel')).resolve() == project:
                revision = git('rev-parse', 'HEAD')
                state = 'dirty' if git('status', '--porcelain', '--untracked-files=all') else 'clean'
                reason = None
        except FileNotFoundError:
            revision = None
            reason = 'git_missing'
        except (OSError, subprocess.SubprocessError):
            revision = None
            reason = 'git_unavailable'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu',
            'threads_per_training_worker': 1,
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch','gymnasium','minigrid','stable-baselines3','numpy','matplotlib',
                          'imageio-ffmpeg','pillow']},
            'base_git_revision': revision, 'git_status': state, 'git_unavailable_reason': reason,
            'source_sha256': {str(Path('src') / root.name / path.name): sha256(path)
                              for path in sorted(root.glob('*.py'))}}


def run_matrix(config, root, *, workers=1, reports=None, config_path=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    started = time.perf_counter()
    root = Path(root)
    seeds = json.loads(resource_path(config['seeds_file'], config_path).read_text())
    if set(seeds['validation']) & set(seeds['evaluation']):
        raise ValueError('Validation and evaluation seeds overlap')
    if set(range(config['ppo']['seed'], config['ppo']['seed'] + config['ppo']['n_envs'])) & set(seeds['evaluation']):
        raise ValueError('PPO and evaluation seeds overlap')
    datasets = {}
    for name in config['datasets']['mixtures']:
        path = root / 'datasets' / f'{name}.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        target = config['datasets']['transitions']
        if metadata['target'] != target or not target <= metadata['transitions'] < target + 256:
            raise ValueError('Dataset transition budget differs from the experiment config')
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
    def complete(run):
        runs.append(run)
        write_json(root/'matrix_progress.json', {'expected_runs':len(jobs), 'runs':runs,
                   'session_seconds':time.perf_counter()-started})
        print(json.dumps({'completed_runs':len(runs),'expected_runs':len(jobs),
                          'dataset':run['dataset'],'method':run['method'],'seed':run['seed'],
                          'success_rate':run['metrics']['success_rate']}),flush=True)
    if workers == 1:
        for job in jobs:
            complete(run_one(*job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(run_one,*job) for job in jobs]):
                complete(future.result())
    runs.sort(key=lambda r:(r['dataset'],r['method'],r['seed']))
    references = {}
    for source, method, checkpoint in [('final','ppo',root/'ppo/final.zip'),
                                       ('early','ppo',root/'ppo/early.zip'), ('random','random',None)]:
        result = evaluate(method,checkpoint,seeds['evaluation'],root/'references'/f'{source}.json')
        references[source] = result['summary']
    export_checkpoint_replays(root,'ppo-collector','ppo',root/'ppo/final.zip',seeds['evaluation'][0],
                              {'dataset':None,'training_seed':config['ppo']['seed']})
    source_audit = json.loads((root/'source_audit.json').read_text())
    validation_sources = {name: {'training_timesteps':source['training_timesteps'],
                                **source['summary']} for name,source in source_audit['sources'].items()}
    result = {'completed_at_utc':datetime.now(timezone.utc).isoformat(), 'config':config, 'seeds':seeds,
              'workers':workers, 'dataset_target':config['datasets']['transitions'],
              'requirements':{'source':'docs/Presentation.pdf','sha256':sha256(resource_path('docs/Presentation.pdf',config_path))},
              'datasets':datasets, 'references':references, 'runs':runs,
              'aggregate':aggregate(runs), 'matrix_session_seconds':time.perf_counter()-started,
              'validation_sources':validation_sources,
              'collection':json.loads((root/'collection_timing.json').read_text()) if (root/'collection_timing.json').exists() else None,
              'ppo':json.loads((root/'ppo/progress.json').read_text()), 'provenance':provenance()}
    write_json(root/'results.json',result)
    if reports:
        write_reports(result,root,Path(reports))
    return result


def pipeline(config, root, *, workers=1, reports=None, config_path=None):
    started = time.perf_counter()
    root = Path(root)
    resume = (root/'ppo/latest.zip').exists() or (root/'ppo/progress.json').exists()
    train_ppo(config['ppo'],root/'ppo',resume=resume)
    collect(config,root,config_path=config_path)
    result = run_matrix(config,root,workers=workers,reports=reports,config_path=config_path)
    write_json(root/'pipeline_timing.json', {'seconds':time.perf_counter()-started,
               'runs':len(result['runs']), 'root':str(root)})
    return result

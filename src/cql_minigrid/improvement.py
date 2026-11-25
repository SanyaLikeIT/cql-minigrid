"""Fine-tune on frozen data; select on validation before opening a fresh test set."""
import json
import math
from pathlib import Path
import shutil
import time

from .common import resource_path, sha256, write_json
from .evaluation import EVALUATION_VERSION, evaluate, export_replay, load_policy
from .experiments import provenance
from .offline import train_offline
from .reporting import METRICS, aggregate
from .rollout import NAVIGATION_METRICS, STALL_THRESHOLD

ALL_METRICS = METRICS + NAVIGATION_METRICS
NUMERICAL_SOURCES = {'common.py', 'data.py', 'env.py', 'evaluation.py', 'offline.py', 'rollout.py'}


def compatible_identity(saved, current):
    """Keep numerical code fixed while allowing report/orchestration fixes on resume."""
    if ({k: v for k, v in saved.items() if k != 'source_sha256'}
            != {k: v for k, v in current.items() if k != 'source_sha256'}):
        return False
    return ({key: digest for key, digest in saved['source_sha256'].items()
             if Path(key).name in NUMERICAL_SOURCES}
            == {key: digest for key, digest in current['source_sha256'].items()
                if Path(key).name in NUMERICAL_SOURCES})


def choose_candidate(candidates, baseline, *, success_tolerance=0., collision_tolerance=0.02,
                     minimum_success=0.):
    """Reject fast failures before minimizing stationary runs and successful length."""
    eligible = [r for r in candidates
                if r['success_rate']['mean'] >= max(minimum_success,
                                                    baseline['success_rate']['mean'] - success_tolerance)
                and r['collision_rate']['mean'] <= baseline['collision_rate']['mean'] + collision_tolerance
                and r['successful_steps_mean']['mean'] is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (r['stall_rate']['mean'],
                                       r['successful_steps_mean']['mean'],
                                       -r['success_rate']['mean'], r['candidate']))


def cached_evaluate(checkpoint, seeds, output):
    path = Path(output)
    digest = sha256(checkpoint)
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous.get('checkpoint_sha256') == digest and previous.get('seeds') == seeds
                and previous.get('evaluation_version') == EVALUATION_VERSION):
            return previous
    return evaluate('cql', checkpoint, seeds, path)


def run_improvement(config, root, *, reports=None, config_path=None, finalize=False):
    started = time.perf_counter()
    study = config['improvement']
    seeds = json.loads(resource_path(config['seeds_file'], config_path).read_text())
    training, validation = seeds['offline_training'], seeds['validation']
    test = list(range(study['test_seed_start'], study['test_seed_start'] + study['test_episodes']))
    groups = [training, validation, test]
    if any(not s or len(set(s)) != len(s) for s in groups):
        raise ValueError('Training, validation, and test seeds must be nonempty and unique')
    if (set(test) & (set(validation) | set(seeds['evaluation']))
            or set(validation) & set(seeds['evaluation'])):
        raise ValueError('Fresh test, validation, and historical evaluation seeds overlap')
    names = [c['name'] for c in study['candidates']]
    if (not names or len(names) != len(set(names))
            or any(Path(n).name != n or n in ['.', '..', 'baseline'] for n in names)):
        raise ValueError('Candidate names must be unique single path components')
    datasets = {}
    initializations = {}
    for c in study['candidates']:
        dataset = resource_path(c['dataset'], config_path)
        meta = json.loads(dataset.with_suffix('.json').read_text())
        if sha256(dataset) != meta['sha256']:
            raise ValueError('Frozen dataset checksum mismatch')
        if (set(validation) | set(test)) & {e['seed'] for e in meta['episode_records']}:
            raise ValueError('Collection seeds overlap validation or test')
        datasets[c['name']] = {'path': str(dataset), 'sha256': meta['sha256'],
                               'transitions': meta['transitions']}
        if c.get('initialize_from'):
            for seed in training:
                path = resource_path(c['initialize_from'].format(seed=seed), config_path)
                initializations[f"{c['name']}/{seed}"] = sha256(path)
    baseline_paths = {seed: resource_path(study['baseline'].format(seed=seed), config_path)
                      for seed in training}
    source = provenance()
    identity = {'config': config, 'seeds': seeds, 'test_seeds': test, 'datasets': datasets,
                'initializations': initializations,
                'baseline_sha256': {str(s): sha256(p) for s, p in baseline_paths.items()},
                'source_sha256': source['source_sha256']}
    root = Path(root)
    identity_path = root / 'identity.json'
    if identity_path.exists():
        recorded_identity = json.loads(identity_path.read_text())
        if not compatible_identity(recorded_identity, identity):
            raise ValueError('Improvement configuration, data, initialization, or numerical source changed on resume')
    else:
        recorded_identity = identity
        write_json(identity_path, identity)
    runs = []
    for seed, checkpoint in baseline_paths.items():
        report = cached_evaluate(checkpoint, validation, root / 'baseline' / f'seed-{seed}.validation.json')
        runs.append({'candidate': 'baseline', 'seed': seed, 'checkpoint': str(checkpoint),
                     'checkpoint_sha256': sha256(checkpoint), 'metrics': report['summary']})
    for candidate in study['candidates']:
        name = candidate['name']
        offline = {**config['offline'], **candidate.get('offline', {})}
        for seed in training:
            checkpoint = root / 'runs' / name / f'seed-{seed}.pt'
            initial = (resource_path(candidate['initialize_from'].format(seed=seed), config_path)
                       if candidate.get('initialize_from') else None)
            train_offline(datasets[name]['path'], checkpoint, offline, method='cql', seed=seed,
                          resume=checkpoint.exists(), initialize_from=initial)
            report = cached_evaluate(checkpoint, validation, checkpoint.with_suffix('.validation.json'))
            runs.append({'candidate': name, 'seed': seed, 'checkpoint': str(checkpoint),
                         'checkpoint_sha256': sha256(checkpoint), 'metrics': report['summary']})
            write_json(root / 'progress.json', {'runs': runs, 'session_seconds': time.perf_counter() - started})
            print(json.dumps({'candidate': name, 'seed': seed, **report['summary']}), flush=True)
    rows = aggregate(runs, group_by=('candidate',), metrics=ALL_METRICS)
    baseline = next(r for r in rows if r['candidate'] == 'baseline')
    selected = choose_candidate([r for r in rows if r['candidate'] != 'baseline'], baseline,
                                **study.get('quality_tolerances', {}))
    selected_runs = [r for r in runs if selected and r['candidate'] == selected['candidate']]
    representative = (min(selected_runs, key=lambda r: (-r['metrics']['success_rate'],
                          r['metrics']['stall_rate'], r['metrics']['successful_steps_mean'] or math.inf,
                          r['seed'])) if selected_runs else None)
    selection = {'candidate': selected['candidate'] if selected else None,
                 'representative': representative, 'validation_aggregate': rows,
                 'rule': 'Meet baseline success/collision gates, then minimize stall rate and successful steps; '
                         'representative maximizes validation success, then minimizes stalls and length.',
                 'stall_threshold': STALL_THRESHOLD}
    selection_path = root / 'selection.json'
    lock_path = root / 'test_lock.json'
    if ((root / 'final.json').exists() or lock_path.exists()):
        locked = json.loads((lock_path if lock_path.exists() else selection_path).read_text())
        if locked != selection:
            raise ValueError('Selection is locked after final test evaluation starts')
    write_json(selection_path, selection)
    final_runs = (json.loads((root / 'final.json').read_text())['runs']
                  if (root / 'final.json').exists() else [])
    if finalize and not final_runs:
        if selected is None:
            raise ValueError('No candidate passed validation quality gates; final test remains unopened')
        write_json(lock_path, selection)
        for run in [r for r in runs if r['candidate'] in ['baseline', selected['candidate']]]:
            report = cached_evaluate(run['checkpoint'], test,
                                     root / 'test' / run['candidate'] / f"seed-{run['seed']}.json")
            final_runs.append({**run, 'metrics': report['summary']})
        for name, checkpoint in [('improved', representative['checkpoint']),
                                 ('baseline', baseline_paths[representative['seed']])]:
            policy, _ = load_policy('cql', checkpoint)
            # A fixed historical example illustrates behavior; selection uses validation only.
            for kind in ['mp4', 'gif']:
                path = root / 'replays' / f'{name}.{kind}'
                meta = export_replay(policy, path, study.get('demo_seed', 20000))
                write_json(path.with_suffix(path.suffix + '.json'),
                           {**meta, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint)})
        write_json(root / 'final.json', {'selection': selection, 'test_seeds': test, 'runs': final_runs})
    if final_runs:
        best = root / 'best.pt'
        if not best.exists() or sha256(best) != representative['checkpoint_sha256']:
            temporary = root / 'best.tmp.pt'
            shutil.copy2(representative['checkpoint'], temporary)
            temporary.replace(best)
        write_json(root / 'best.json', {**representative, 'selected_on': 'validation',
                                        'alias': str(best)})
        write_json(root / 'test-seeds.json', {**seeds, 'evaluation': test})
        write_json(root / 'evaluation-config.json', {'seeds_file': 'test-seeds.json'})
    result = {'config': config, 'selection': selection, 'validation_runs': runs,
              'test_seeds': test if final_runs else [], 'test_runs': final_runs,
              'test_aggregate': aggregate(final_runs, group_by=('candidate',), metrics=ALL_METRICS),
              'session_seconds': time.perf_counter() - started, 'provenance': source,
              'training_source_sha256': recorded_identity['source_sha256']}
    write_json(root / 'results.json', result)
    if reports:
        write_json(Path(reports) / 'summary.json', result)
    return result

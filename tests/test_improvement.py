import json

import numpy as np
import pytest
import torch

from cql_minigrid.collector import collect_dataset
from cql_minigrid.common import sha256
from cql_minigrid.improvement import choose_candidate, compatible_identity, run_improvement
from cql_minigrid.offline import train_offline
from cql_minigrid.rollout import NavigationStats, random_policy, summarize


def tiny_dataset(tmp_path):
    path = tmp_path / 'random.npz'
    collect_dataset(path, {'random': random_policy}, {'random': 1.}, target=16,
                    episode_seed_start=1000000, policy_seed_start=120,
                    source_hashes={}, source_audit={})
    config = json.loads(open('configs/smoke.json').read())['offline']
    return path, {**config, 'updates': 4, 'checkpoint_interval': 2, 'log_interval': 1}


def test_training_penalty_changes_targets_without_rewriting_data(tmp_path, monkeypatch):
    from cql_minigrid import offline
    dataset, config = tiny_dataset(tmp_path)
    digest = sha256(dataset)
    with np.load(dataset) as data:
        rewards = data['reward']
        n = len(rewards)
    sampled = np.random.default_rng(0).integers(n, size=config['batch_size'])
    seen = []
    original = offline.td_targets
    def capture(reward, *args):
        seen.append(reward.clone())
        return original(reward, *args)
    monkeypatch.setattr(offline, 'td_targets', capture)
    train_offline(dataset, tmp_path / 'penalized.pt', {**config, 'step_penalty': .005},
                  method='cql', stop_after=1)
    torch.testing.assert_close(seen[0], torch.from_numpy(rewards[sampled]) - .005)
    assert sha256(dataset) == digest


def test_warm_start_and_penalized_resume_are_exact(tmp_path):
    dataset, config = tiny_dataset(tmp_path)
    base = tmp_path / 'base.pt'
    train_offline(dataset, base, config, method='cql')
    digest = sha256(base)
    # With zero learning rate, new runs must retain the initialized weights exactly.
    fine = {**config, 'step_penalty': .005, 'learning_rate': 0.}
    kwargs = dict(method='cql', initialize_from=base)
    train_offline(dataset, tmp_path / 'full.pt', fine, **kwargs)
    train_offline(dataset, tmp_path / 'resumed.pt', fine, stop_after=2, **kwargs)
    train_offline(dataset, tmp_path / 'resumed.pt', fine, resume=True, **kwargs)
    initial, full, resumed = [torch.load(tmp_path / p, weights_only=False)
                              for p in ['base.pt', 'full.pt', 'resumed.pt']]
    for name in initial['model']:
        assert torch.equal(initial['model'][name], full['model'][name])
        assert torch.equal(full['model'][name], resumed['model'][name])
    assert full['history'] == resumed['history']
    assert full['identity']['initialization']['sha256'] == digest == sha256(base)
    with pytest.raises(ValueError, match='differs'):
        train_offline(dataset, tmp_path / 'resumed.pt', {**fine, 'step_penalty': .01},
                      resume=True, **kwargs)


@pytest.mark.parametrize('penalty', [-.1, float('nan'), float('inf'), True])
def test_invalid_penalty_is_rejected_before_data_access(tmp_path, penalty):
    with pytest.raises(ValueError, match='step_penalty'):
        train_offline('missing.npz', tmp_path / 'bad.pt', {'step_penalty': penalty})


def test_stationary_counter_resets_on_translation_and_survives_turns():
    stats = NavigationStats()
    for _ in range(16):
        stats.update(0, (1, 1), (1, 1))
    stats.update(2, (1, 1), (2, 1))
    stats.update(1, (2, 1), (2, 1))
    assert stats.report() == {'turn_steps': 17, 'stationary_steps': 17, 'max_stationary_steps': 16}
    episodes = [{'outcome': 'success', 'return': 1., 'steps': 18, **stats.report()}]
    assert summarize(episodes)['stall_rate'] == 1.
    # Historical collection summaries retain their original metric contract.
    assert 'stall_rate' not in summarize([{'outcome': 'success', 'return': 1., 'steps': 18}])


def test_selection_rejects_fast_failures_and_collision_regressions():
    def row(name, success, collision, stall, steps):
        return {'candidate': name, **{k: {'mean': v} for k, v in
                [('success_rate', success), ('collision_rate', collision),
                 ('stall_rate', stall), ('successful_steps_mean', steps)]}}
    baseline = row('baseline', .6, .2, .8, 60)
    candidates = [row('fast-failure', .1, .2, 0., 10), row('unsafe', .8, .5, 0., 12),
                  row('efficient', .7, .2, .1, 20), row('slow', .8, .1, .5, 50)]
    assert choose_candidate(candidates, baseline)['candidate'] == 'efficient'
    assert choose_candidate(candidates[:2], baseline) is None
    assert choose_candidate(candidates, baseline, minimum_success=.9) is None


def test_study_does_not_open_final_test_before_selection(tmp_path):
    dataset, offline = tiny_dataset(tmp_path)
    base = tmp_path / 'base-0.pt'
    train_offline(dataset, base, offline, method='cql')
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps({'offline_training': [0], 'validation': [10000, 10001],
                                 'evaluation': [20000]}))
    config = {'seeds_file': str(seeds), 'offline': offline, 'improvement': {
        'baseline': str(tmp_path / 'base-{seed}.pt'), 'test_seed_start': 30000, 'test_episodes': 2,
        'candidates': [{'name': 'fine', 'dataset': str(dataset),
                        'initialize_from': str(tmp_path / 'base-{seed}.pt')} ]}}
    result = run_improvement(config, tmp_path / 'study')
    assert result['test_runs'] == [] and result['test_seeds'] == []
    assert not (tmp_path / 'study/test').exists()
    assert run_improvement(config, tmp_path / 'study')['selection'] == result['selection']
    config['improvement']['test_seed_start'] = 10000
    with pytest.raises(ValueError, match='overlap'):
        run_improvement(config, tmp_path / 'different')


def test_guided_teacher_moves_safely_and_reaches_the_goal():
    from cql_minigrid.env import make_env
    from cql_minigrid.guided import shortest_path_policy
    for seed in [10000, 10001, 10002, 10003]:
        env = make_env()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(256):
            action = shortest_path_policy(obs, rng)
            if action == 2:
                front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
                assert front is None or front.type == 'goal'
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                assert info['outcome'] == 'success'
                break
        else:
            pytest.fail('Teacher failed to finish within the environment horizon')
        env.close()


def test_local_geometry_matches_the_observed_front_cell():
    from minigrid.core.constants import OBJECT_TO_IDX
    from cql_minigrid.env import make_env
    from cql_minigrid.offline import local_navigation_features
    env = make_env()
    obs, _ = env.reset(seed=10000)
    for _ in range(4):
        features = local_navigation_features(torch.from_numpy(obs)[None])
        front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
        kind = OBJECT_TO_IDX[front.type if front else 'empty']
        assert features.shape == (1, 50)
        assert int(features[0, :11].argmax()) == kind
        np.testing.assert_allclose(features[0, -2:].numpy(), np.array(env.unwrapped.agent_pos) / 7.)
        obs, *_ = env.step(0)
    env.close()


def test_geometry_expansion_preserves_initial_policy_and_resume(tmp_path):
    from cql_minigrid.env import make_env
    from cql_minigrid.offline import NetworkPolicy
    dataset, config = tiny_dataset(tmp_path)
    base = tmp_path / 'base.pt'
    train_offline(dataset, base, config, method='cql')
    fine = {**config, 'navigation_features': True, 'learning_rate': 0.}
    kwargs = dict(method='cql', initialize_from=base)
    train_offline(dataset, tmp_path / 'geometry.pt', fine, stop_after=2, **kwargs)
    train_offline(dataset, tmp_path / 'geometry.pt', fine, resume=True, **kwargs)
    old, new = NetworkPolicy(base), NetworkPolicy(tmp_path / 'geometry.pt')
    env = make_env()
    obs, _ = env.reset(seed=10000)
    tensor = torch.from_numpy(obs)[None]
    torch.testing.assert_close(old.model(tensor), new.model(tensor), rtol=0., atol=0.)
    env.close()


def test_resume_keeps_numerical_code_fixed_and_retains_report_fixes():
    original = {'config': {'updates': 4}, 'source_sha256': {
        'src/cql_minigrid/offline.py': 'training', 'src/cql_minigrid/reporting.py': 'old-report'}}
    updated = {**original, 'source_sha256': {**original['source_sha256'],
                                          'src/cql_minigrid/reporting.py': 'fixed-report'}}
    assert compatible_identity(original, updated)
    updated['source_sha256']['src/cql_minigrid/offline.py'] = 'different-training'
    assert not compatible_identity(original, updated)


def test_final_test_is_locked_and_retained_on_resume(tmp_path, monkeypatch):
    from cql_minigrid import improvement
    dataset, offline = tiny_dataset(tmp_path)
    train_offline(dataset, tmp_path / 'base-0.pt', offline, method='cql')
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps({'offline_training': [0], 'validation': [10000], 'evaluation': [20000]}))
    config = {'seeds_file': str(seeds), 'offline': offline, 'improvement': {
        'baseline': str(tmp_path / 'base-{seed}.pt'), 'test_seed_start': 30000, 'test_episodes': 2,
        'candidates': [{'name': 'fine', 'dataset': str(dataset),
                        'initialize_from': str(tmp_path / 'base-{seed}.pt')}]}}
    # Selection quality is tested separately; this fixture exercises finalization and its lock.
    monkeypatch.setattr(improvement, 'choose_candidate', lambda candidates, baseline, **kw: candidates[0])
    result = run_improvement(config, tmp_path / 'study', finalize=True)
    assert len(result['test_runs']) == 2 and result['test_seeds'] == [30000, 30001]
    assert sha256(tmp_path / 'study/best.pt') == result['selection']['representative']['checkpoint_sha256']
    assert json.loads((tmp_path / 'study/best.json').read_text())['selected_on'] == 'validation'
    resumed = run_improvement(config, tmp_path / 'study')
    assert resumed['test_runs'] == result['test_runs']
    assert resumed['test_seeds'] == result['test_seeds']
    monkeypatch.setattr(improvement, 'choose_candidate', lambda *args, **kw: None)
    with pytest.raises(ValueError, match='locked'):
        run_improvement(config, tmp_path / 'study')


def test_interrupted_final_test_already_locks_validation_selection(tmp_path, monkeypatch):
    from cql_minigrid import improvement
    dataset, offline = tiny_dataset(tmp_path)
    train_offline(dataset, tmp_path / 'base-0.pt', offline, method='cql')
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps({'offline_training': [0], 'validation': [10000], 'evaluation': [20000]}))
    config = {'seeds_file': str(seeds), 'offline': offline, 'improvement': {
        'baseline': str(tmp_path / 'base-{seed}.pt'), 'test_seed_start': 30000, 'test_episodes': 2,
        'candidates': [{'name': 'fine', 'dataset': str(dataset),
                        'initialize_from': str(tmp_path / 'base-{seed}.pt')}]}}
    original = improvement.cached_evaluate
    def interrupt(checkpoint, selected_seeds, output):
        if selected_seeds[0] == 30000:
            raise RuntimeError('interrupted final test')
        return original(checkpoint, selected_seeds, output)
    monkeypatch.setattr(improvement, 'cached_evaluate', interrupt)
    monkeypatch.setattr(improvement, 'choose_candidate', lambda candidates, baseline, **kw: candidates[0])
    with pytest.raises(RuntimeError, match='interrupted'):
        run_improvement(config, tmp_path / 'study', finalize=True)
    assert (tmp_path / 'study/test_lock.json').exists()
    assert not (tmp_path / 'study/final.json').exists()
    monkeypatch.setattr(improvement, 'choose_candidate', lambda *args, **kw: None)
    with pytest.raises(ValueError, match='locked'):
        run_improvement(config, tmp_path / 'study')

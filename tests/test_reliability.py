import math

import pytest

from cql_minigrid.reliability import METRICS, aggregate, check_seeds, paired, statistics


def row(seed, value):
    return {'seed': seed, 'method': 'cql', 'metrics': dict.fromkeys(METRICS, value)}


def test_student_interval_uses_models_and_sample_sd():
    result = statistics(range(10))
    assert result['n'] == 10
    assert result['mean'] == result['median'] == 4.5
    assert result['std'] == pytest.approx(math.sqrt(55 / 6))
    half = 2.2621571627409915 * math.sqrt(55 / 6) / math.sqrt(10)
    assert result['ci95'] == pytest.approx([4.5 - half, 4.5 + half])
    assert statistics([1] * 10)['ci95'] == [1, 1]
    assert statistics([None])['n'] == 0
    assert statistics([1])['ci95'] is None


def test_paired_interval_uses_differences_and_retains_failures():
    left = [row(seed, seed / 10) for seed in range(10)]
    right = [row(seed, seed / 10 + .1) for seed in range(10)]
    result = paired(left, right)['success_rate']
    assert result['n'] == 10
    assert result['mean'] == pytest.approx(-.1)
    assert result['ci95'] == pytest.approx([-.1, -.1])
    assert aggregate(left, ('method',))[0]['success_rate']['minimum'] == 0
    with pytest.raises(ValueError, match='identical unique'):
        paired(left, right[:-1])


def test_duplicate_model_is_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate([row(0, .5), row(0, .7)], ('method',))


def test_fresh_seed_range_excludes_all_prior_ranges_and_collection():
    seeds = {'offline_training': list(range(10)), 'validation': list(range(10000, 10064)),
             'evaluation': list(range(20000, 20100))}
    metadata = {'guided': {'episode_records': [{'seed': 1300000}], 'policy_seed_start': 300120}}
    check_seeds(list(range(40000, 40200)), seeds, metadata)
    for start in [0, 10000, 20000, 30000, 1300000, 300120]:
        with pytest.raises(ValueError, match='overlap'):
            check_seeds(list(range(start, start + 200)), seeds, metadata)

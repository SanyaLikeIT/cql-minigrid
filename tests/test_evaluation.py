import json
from PIL import Image
from cql_minigrid.evaluation import evaluate, export_gif
from cql_minigrid.rollout import random_policy


def test_metrics_and_replay_use_the_same_seeded_episode(tmp_path):
    report = evaluate('random', None, [20000,20001], tmp_path / 'metrics.json')
    demo = export_gif(random_policy, tmp_path / 'demo.gif', 20000)
    episode = report['episodes'][0]
    assert demo['steps'] == episode['steps']
    assert demo['return'] == episode['return']
    assert demo['outcome'] == episode['outcome']
    assert sum(report['summary'][x + '_rate'] for x in ['success','collision','timeout']) == 1
    assert json.loads((tmp_path/'metrics.json').read_text())['seeds'] == [20000,20001]
    with Image.open(tmp_path / 'demo.gif') as gif:
        assert gif.format == 'GIF' and gif.n_frames > 1

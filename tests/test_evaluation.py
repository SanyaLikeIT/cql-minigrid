import json
import imageio_ffmpeg
import pytest
from PIL import Image
from cql_minigrid.evaluation import evaluate, export_gif, export_mp4
from cql_minigrid.rollout import random_policy


def test_metrics_gif_and_mp4_use_the_same_seeded_episode(tmp_path):
    report = evaluate('random', None, [20000,20001], tmp_path / 'metrics.json')
    gif_meta = export_gif(random_policy, tmp_path / 'demo.gif', 20000)
    mp4_meta = export_mp4(random_policy, tmp_path / 'demo.mp4', 20000)
    episode = report['episodes'][0]
    for replay in [gif_meta,mp4_meta]:
        for key in ['steps','return','outcome','seed','terminated','truncated']:
            assert replay[key] == episode[key]
        assert replay['elapsed_steps']==episode['steps']
        assert replay['frame_indices'][0]==0 and replay['frame_indices'][-1]==episode['steps']
    assert sum(report['summary'][x + '_rate'] for x in ['success','collision','timeout']) == 1
    assert json.loads((tmp_path/'metrics.json').read_text())['seeds'] == [20000,20001]
    assert (tmp_path/'demo.gif.json').exists() and (tmp_path/'demo.mp4.json').exists()
    with Image.open(tmp_path/'demo.gif') as gif:assert gif.format=='GIF' and gif.n_frames>1
    reader=imageio_ffmpeg.read_frames(str(tmp_path/'demo.mp4'))
    try:
        stream=next(reader)
        assert stream['codec']=='h264' and stream['pix_fmt'].startswith('yuv420p')
        assert stream['size']==(256,256)
    finally:reader.close()
    assert mp4_meta['encoded_frames']==mp4_meta['sampled_frames']>1


def test_successful_steps_and_all_outcome_rates(tmp_path,monkeypatch):
    import cql_minigrid.rollout as rollout
    from cql_minigrid.env import make_env
    monkeypatch.setattr(rollout,'make_env',lambda **kwargs:make_env(n_obstacles=0,**kwargs))
    def route(obs,rng):return 1 if round(float(obs[-1])*256)==5 else 2
    report=rollout.run_episodes(route,[20000,20001])
    row=report['summary']
    assert row['success_rate']==1 and row['collision_rate']==row['timeout_rate']==0
    assert row['successful_steps_mean']==11
    assert row['return_mean']==pytest.approx(1-.9*11/256)

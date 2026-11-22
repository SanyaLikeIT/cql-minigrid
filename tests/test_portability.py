import json
from pathlib import Path

from cql_minigrid.common import resource_path
from cql_minigrid.collector import train_ppo


def test_relative_config_inputs_are_resolved_without_mutating_config(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/main.json'
    config=json.loads(path.read_text());before=json.dumps(config,sort_keys=True)
    monkeypatch.chdir(tmp_path)
    assert resource_path(config['seeds_file'],path)==root/'configs/seeds.json'
    assert resource_path('docs/Presentation.pdf',path)==root/'docs/Presentation.pdf'
    assert json.dumps(config,sort_keys=True)==before


def test_cli_evaluation_from_another_directory(tmp_path, monkeypatch):
    import sys
    from cql_minigrid import cli
    config=Path(__file__).resolve().parents[1]/'configs/smoke.json'
    output=tmp_path/'metrics.json'
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/smoke_seeds.json').write_text(json.dumps({'evaluation':[999]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PATH','')
    monkeypatch.setattr(sys,'argv',['cql-minigrid','evaluate','--method','random',
                                  '--config',str(config),'--root',str(tmp_path/'run'),'--output',str(output)])
    cli.main()
    result=json.loads(output.read_text())
    assert result['seeds']==json.loads((config.parent/'smoke_seeds.json').read_text())['evaluation']


def test_completed_ppo_needs_no_latest_duplicate_or_environment(tmp_path, monkeypatch):
    from cql_minigrid import collector
    config={'total_timesteps':256,'seed':12}
    progress={'config':config,'timesteps':256,'seconds':1.}
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    # Small local fixtures test the completed-run guard, without loading or retraining PPO.
    for name in ['early.zip','final.zip']:(tmp_path/name).write_bytes(b'checkpoint-guard-fixture')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    def forbidden(*args,**kwargs):raise AssertionError('Completed PPO must not construct environments')
    monkeypatch.setattr(collector,'DummyVecEnv',forbidden)
    assert train_ppo(config,tmp_path,resume=True)==progress
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}

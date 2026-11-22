import json

import pytest
import torch

from cql_minigrid.ablation import run_ablation, validate_study
from cql_minigrid.collector import collect_dataset
from cql_minigrid.common import write_json
from cql_minigrid.offline import train_offline
from cql_minigrid.reporting import aggregate
from cql_minigrid.rollout import random_policy


@pytest.mark.parametrize('alphas', [[], [1,1], [-1], [float('nan')], [float('inf')], [True]])
def test_invalid_alpha_sweep_is_rejected_before_loading_data(alphas):
    with pytest.raises(ValueError,match='alphas'):
        validate_study({'study':{'alphas':alphas}})


def test_alpha_zero_trains_identically_to_dqn(tmp_path):
    dataset=tmp_path/'random.npz'
    collect_dataset(dataset,{'random':random_policy},{'random':1.},target=16,
                    episode_seed_start=1000000,policy_seed_start=120,source_hashes={},source_audit={})
    config=json.load(open('configs/smoke.json'))['offline']
    config={**config,'alpha':0.}
    for method in ['dqn','cql']:
        train_offline(dataset,tmp_path/f'{method}.pt',config,method=method,seed=0)
    dqn,cql=[torch.load(tmp_path/f'{m}.pt',weights_only=False) for m in ['dqn','cql']]
    assert dqn['history']==cql['history']
    for key,value in dqn['model'].items():
        assert torch.equal(value,cql['model'][key])


def test_ablation_separates_alpha_checkpoints_and_rejects_changed_resume(tmp_path, monkeypatch):
    from cql_minigrid import ablation
    dataset=tmp_path/'random.npz'
    collect_dataset(dataset,{'random':random_policy},{'random':1.},target=16,
                    episode_seed_start=1000000,policy_seed_start=120,source_hashes={},source_audit={})
    seeds=tmp_path/'seeds.json'
    write_json(seeds,{'offline_training':[0],'evaluation':[20000,20001],'validation':[10000]})
    config=json.load(open('configs/smoke.json'))
    config['seeds_file']=str(seeds);config['datasets']['transitions']=16
    config['study']={'alphas':[0.,1.],'datasets':{'test':str(dataset)},'bc_datasets':[],'ppo_root':str(tmp_path)}
    # Collector reference wiring is covered by the full pipeline smoke; no PPO fixture is needed here.
    monkeypatch.setattr(ablation,'evaluate',lambda *args,**kwargs:{'summary':{}})
    result=run_ablation(config,tmp_path/'runs')
    assert len(result['runs'])==2 and len(result['aggregate'])==2
    checkpoints=[r['checkpoint'] for r in result['runs']]
    assert len(set(checkpoints))==2
    assert run_ablation(config,tmp_path/'runs')['aggregate']==result['aggregate']
    config['study']['alphas']=[0.,.1,1.]
    with pytest.raises(ValueError,match='changed on resume'):
        run_ablation(config,tmp_path/'runs')

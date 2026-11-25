"""Audit complete ten-seed matrices against checkpoints and raw episode records."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cql_minigrid.common import sha256
from cql_minigrid.reliability import METRICS, SEEDS, aggregate, check_seeds
from cql_minigrid.rollout import summarize
import numpy as np


def verify(project):
    root=project/'reports/reliability';total=0
    configurations={
        'baseline':(6,('dataset','method'),list(range(20000,20100))),
        'ablation':(13,('dataset','method','alpha'),list(range(20000,20100))),
        'navigation_validation':(9,('candidate',),list(range(10000,10064))),
        'navigation':(2,('candidate',),list(range(40000,40200)))}
    for study,(groups,keys,seeds) in configurations.items():
        result=json.loads((root/study/'summary.json').read_text())
        assert len(result['runs'])==groups*10,(study,'run count')
        computed=aggregate(result['runs'],keys)
        assert computed==result['aggregate'],(study,'statistics')
        assert len(computed)==groups and all(r['training_seeds']==SEEDS for r in computed)
        for run in result['runs']:
            assert sha256(project/run['checkpoint'])==run['checkpoint_sha256']
            report=json.loads((project/run['evaluation_file']).read_text())
            assert report['seeds']==seeds
            assert [e['seed'] for e in report['episodes']]==seeds
            summary=summarize(report['episodes'])
            for key,value in summary.items():assert run['metrics'][key]==value,(study,key)
            stationary=[e['max_stationary_steps'] for e in report['episodes']]
            assert run['metrics']['max_stationary_steps_max']==max(stationary)
            assert run['metrics']['max_stationary_steps_median']==np.median(stationary)
            assert all(k in run['metrics'] for k in METRICS)
            total+=len(seeds)
    identity=json.loads((project/'artifacts/reliability/identity.json').read_text())
    files={'strong':'artifacts/final/datasets/strong.npz','mixed':'artifacts/final/datasets/mixed.npz',
           'mixed_balanced':'artifacts/ablation/datasets/mixed_balanced.npz','guided':'artifacts/navigation-guided/datasets/guided.npz'}
    meta={}
    for name,path in files.items():
        assert sha256(project/path)==identity['dataset_hashes'][name]
        meta[name]=json.loads((project/Path(path).with_suffix('.json')).read_text())
    seeds=json.loads((project/'configs/seeds.json').read_text())
    check_seeds(list(range(40000,40200)),seeds,meta)
    lock=json.loads((project/'artifacts/reliability/navigation/test_lock.json').read_text())
    final=json.loads((root/'navigation/summary.json').read_text())
    assert set(r['candidate'] for r in final['runs'])=={'baseline',lock['candidate']}
    print(json.dumps({'verified':True,'baseline_models':60,'ablation_conditions':130,
                      'navigation_validation_models':90,'fresh_test_models':20,
                      'logical_evaluation_episodes':total,'dataset_hashes_unchanged':True},indent=2))


if __name__=='__main__':verify(Path(__file__).resolve().parents[1])

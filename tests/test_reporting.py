import pytest
from cql_minigrid.reporting import aggregate


def test_mean_and_sample_sd_are_over_training_seeds():
    runs=[]
    for seed,value in enumerate([.2,.4,.6]):
        runs.append({'dataset':'strong','method':'cql','seed':seed,
                     'metrics':{'success_rate':value,'collision_rate':1-value,'timeout_rate':0.,
                                'return_mean':2*value-1,'successful_steps_mean':None}})
    row=aggregate(runs)[0]
    assert row['training_seeds']==[0,1,2]
    assert row['success_rate']['mean']==pytest.approx(.4)
    assert row['success_rate']['std']==pytest.approx(.2)
    assert row['return_mean']['std']==pytest.approx(.4)
    assert row['successful_steps_mean']['mean'] is None
    with pytest.raises(ValueError,match='Duplicate'):
        aggregate([runs[0],runs[0]])

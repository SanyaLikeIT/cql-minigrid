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
def test_successful_length_uses_only_seeds_with_successes():
    from cql_minigrid.reporting import aggregate,results_table,reference_table
    runs=[]
    for seed,length in enumerate([10,None,14]):
        runs.append({'dataset':'strong','method':'cql','seed':seed,
                     'metrics':{'success_rate':.5 if length else 0.,'collision_rate':.5,
                                'timeout_rate':0. if length else .5,'return_mean':0.,
                                'successful_steps_mean':length}})
    row=aggregate(runs)[0]
    assert row['successful_steps_mean']['mean']==12
    assert row['successful_steps_mean']['valid_seeds']==2
    table=results_table([row])
    assert 'Timeout %' in table and 'Successful steps' in table
    refs={name:{'episodes':100,'success_rate':0.,'collision_rate':1.,'timeout_rate':0.,
                'return_mean':-1.,'successful_steps_mean':None} for name in ['final','early','random']}
    rendered=reference_table(refs)
    assert 'PPO final' in rendered and 'Timeout %' in rendered and '—' in rendered

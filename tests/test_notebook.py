import json
from pathlib import Path

import torch


def test_notebook_numerical_example_and_saved_outputs(monkeypatch):
    from cql_minigrid import offline,env
    path=Path(__file__).resolve().parents[1]/'notebooks/CQL_value_update_demo.ipynb'
    notebook=json.loads(path.read_text())
    cells=[cell for cell in notebook['cells'] if cell['cell_type']=='code']
    assert [c['execution_count'] for c in cells]==list(range(1,len(cells)+1))
    assert all(c['outputs'] for c in cells)
    assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
    outputs=json.dumps([cell['outputs'] for cell in cells])
    assert '/home/' not in outputs and '/tmp/' not in outputs
    def forbidden(*args,**kwargs):raise AssertionError('The notebook must not train an agent or step an environment')
    monkeypatch.setattr(offline,'train_offline',forbidden)
    monkeypatch.setattr(env,'make_env',forbidden)
    monkeypatch.chdir(path.parent)
    namespace={}
    for index,cell in enumerate(cells):
        exec(compile(''.join(cell['source']),f'{path.name}:cell{index}','exec'),namespace)
    assert abs(namespace['target'].item()-1.72)<1e-6
    torch.testing.assert_close(namespace['q_dqn'],torch.tensor([[.2,.7,-.1]]),atol=1e-6,rtol=1e-6)
    assert namespace['q_cql'][0,0]<namespace['q_dqn'][0,0]
    assert namespace['q_cql'][0,1]>namespace['q_dqn'][0,1]

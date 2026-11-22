import shutil
import subprocess
from pathlib import Path

import pytest

from cql_minigrid import experiments


def copy_source(tmp_path, monkeypatch):
    source = Path(experiments.__file__).resolve().parent
    target = tmp_path / 'project/src/cql_minigrid'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(experiments, '__file__', str(target / 'experiments.py'))
    return target.parent.parent


def test_zip_provenance_ignores_parent_repository_and_cwd(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    project = copy_source(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    first = experiments.provenance()
    monkeypatch.chdir(project)
    assert experiments.provenance() == first
    assert first['base_git_revision'] is None
    assert first['git_unavailable_reason'] == 'no_repository'
    assert 'src/cql_minigrid/experiments.py' in first['source_sha256']
    assert all(not Path(key).is_absolute() for key in first['source_sha256'])


def test_repository_provenance_uses_source_checkout_from_other_cwd(tmp_path, monkeypatch):
    expected = experiments.provenance()
    monkeypatch.chdir(tmp_path)
    actual = experiments.provenance()
    assert actual == expected
    assert actual['base_git_revision']


@pytest.mark.parametrize('error', [FileNotFoundError(), subprocess.TimeoutExpired('git', 5),
                                  subprocess.CalledProcessError(128, 'git')])
def test_provenance_survives_unavailable_git(tmp_path, monkeypatch, error):
    project = copy_source(tmp_path, monkeypatch)
    (project / '.git').mkdir()
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(experiments.subprocess, 'check_output', fail)
    result = experiments.provenance()
    assert result['base_git_revision'] is None
    assert result['git_status'] == 'unavailable'
    assert result['source_sha256']

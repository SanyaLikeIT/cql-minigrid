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


@pytest.mark.skipif(shutil.which('git') is None, reason='Requires an installed Git executable')
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
    assert actual['source_sha256']


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


def test_valid_checkout_provenance_when_available():
    if shutil.which('git') is None:
        pytest.skip('Requires an installed Git executable')
    package = Path(experiments.__file__).resolve().parent
    project = package.parent.parent
    if not (project / '.git').exists():
        pytest.skip('The source distribution has no Git checkout')
    probe = subprocess.run(['git','-C',str(project),'rev-parse','HEAD'],text=True,capture_output=True)
    if probe.returncode:
        pytest.skip('No readable source-checkout HEAD is available')
    expected = probe.stdout.strip()
    assert experiments.provenance()['base_git_revision'] == expected


def test_real_missing_git_executable(tmp_path, monkeypatch):
    project = copy_source(tmp_path, monkeypatch)
    (project / '.git').mkdir()
    monkeypatch.setenv('PATH', '')
    result = experiments.provenance()
    assert result['base_git_revision'] is None
    assert result['git_unavailable_reason'] == 'git_missing'
    assert result['source_sha256']


def test_actual_zip_extraction_without_git(tmp_path, monkeypatch):
    import zipfile
    source = Path(experiments.__file__).resolve().parent
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive,'w') as file:
        for path in source.glob('*.py'):
            file.write(path,'project/src/cql_minigrid/'+path.name)
    with zipfile.ZipFile(archive) as file:
        file.extractall(tmp_path)
    (tmp_path / '.git').mkdir()
    monkeypatch.setattr(experiments,'__file__',str(tmp_path/'project/src/cql_minigrid/experiments.py'))
    monkeypatch.setenv('PATH','')
    result=experiments.provenance()
    assert result['base_git_revision'] is None
    assert result['git_unavailable_reason']=='no_repository'
    assert result['source_sha256']

import importlib.util
from pathlib import Path
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('release_assets',Path(__file__).resolve().parents[1]/'scripts/build_release_assets.py')
assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)


def test_bundle_reproducibility_and_exact_contents_without_git(tmp_path,monkeypatch):
    monkeypatch.setenv('PATH','')
    source=tmp_path/'artifacts/example.json';source.parent.mkdir();source.write_text('{"value": 1}\n')
    entry=assets.checked_file(tmp_path,'artifacts/example.json')
    first,second=tmp_path/'one.zip',tmp_path/'two.zip'
    assets.write_bundle(tmp_path,first,[entry]);assets.write_bundle(tmp_path,second,[entry])
    assert first.read_bytes()==second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist()==['artifacts/example.json']
        assert archive.read(entry['path'])==source.read_bytes()


def test_bundle_rejects_corruption_before_replacing_output(tmp_path):
    source=tmp_path/'model.pt';source.write_bytes(b'original test fixture')
    entry=assets.checked_file(tmp_path,'model.pt')
    output=tmp_path/'models.zip';output.write_bytes(b'previous output')
    source.write_bytes(b'changed test fixture')
    with pytest.raises(ValueError,match='checksum mismatch'):
        assets.write_bundle(tmp_path,output,[entry])
    assert output.read_bytes()==b'previous output'


def test_inventory_rejects_paths_outside_project(tmp_path):
    with pytest.raises(ValueError,match='inside the project'):
        assets.checked_file(tmp_path,'../private.txt')

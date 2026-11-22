import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_submission',ROOT/'scripts/build_submission.py')
submission=importlib.util.module_from_spec(spec);spec.loader.exec_module(submission)


def test_archive_is_reproducible_and_hashes_match_without_git(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH','')
    (tmp_path/'example.py').write_text('print(42)\n')
    files={'example.py':'source'}
    manifest={'mode':'draft','submission_complete':False,'files':[{'path':'example.py','sha256':submission.digest(tmp_path/'example.py')}]}
    first,second=tmp_path/'first.zip',tmp_path/'second.zip'
    for output in [first,second]:submission.write_archive(tmp_path,output,files,manifest)
    assert first.read_bytes()==second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        actual=json.loads(archive.read('cql-minigrid/SUBMISSION_MANIFEST.json'))
        assert actual==manifest and not actual['submission_complete']
        assert archive.read('cql-minigrid/example.py')==(tmp_path/'example.py').read_bytes()


def test_final_requires_both_author_media_before_writing(tmp_path):
    output=tmp_path/'dist/final.zip'
    with pytest.raises(ValueError,match='FINAL_PRESENTATION.pdf.*DEMO.mp4'):
        submission.build_submission(tmp_path,output,'final')
    assert not output.exists()


def test_empty_media_cannot_pass_as_final_submission(tmp_path):
    for name in [submission.PRESENTATION,submission.DEMO]:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'')
    with pytest.raises(ValueError,match='PDF'):
        submission.build_submission(tmp_path,tmp_path/'final.zip','final')
    assert not (tmp_path/'final.zip').exists()


def test_full_manifest_selects_final_evidence_and_excludes_redundancy():
    if not all((ROOT/name).exists() for name in submission.DATASETS):
        pytest.skip('Full frozen artifacts are local or included in the submission ZIP')
    files,datasets=submission.select_files(ROOT)
    assert len(datasets)==3
    assert sum(category=='final_offline_checkpoint' for category in files.values())==57
    assert not any('.parts' in p or p.endswith('.gif') or 'latest.zip' in p for p in files)
    assert not any(p.startswith('artifacts/main/') or p.startswith('artifacts/pilots/') for p in files)

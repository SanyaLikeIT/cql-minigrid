"""Build a deterministic Moodle draft or a final ZIP with author-supplied media.

Uses only the Python standard library. No Git executable or metadata is needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import zipfile

PRESENTATION = 'submission/presentation/FINAL_PRESENTATION.pdf'
DEMO = 'submission/demo/DEMO.mp4'
DATASETS = ('artifacts/final/datasets/strong.npz', 'artifacts/final/datasets/mixed.npz',
            'artifacts/ablation/datasets/mixed_balanced.npz')
REPORTS = ('reports/summary.json', 'reports/ablation/summary.json')
FIXED_TIME = (2020, 1, 1, 0, 0, 0)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_media(root):
    missing = [name for name in [PRESENTATION, DEMO] if not (root / name).is_file()]
    if missing:
        raise ValueError('Final submission requires author-supplied media: ' + ', '.join(missing))
    pdf = root / PRESENTATION
    with pdf.open('rb') as file:
        header = file.read(8)
        file.seek(max(0, pdf.stat().st_size - 4096))
        trailer = file.read()
    if not header.startswith(b'%PDF-') or b'%%EOF' not in trailer:
        raise ValueError('Final presentation is empty or lacks a PDF header/trailer')
    proposal = root / 'docs/Presentation.pdf'
    if proposal.exists() and digest(pdf) == digest(proposal):
        raise ValueError('The earlier project proposal cannot serve as the final presentation')
    video = root / DEMO
    boxes = set()
    size = video.stat().st_size
    with video.open('rb') as file:
        offset = 0
        while offset < size:
            file.seek(offset)
            header = file.read(8)
            if len(header) != 8:
                raise ValueError('Final demo is not a complete MP4 file')
            length, kind = struct.unpack('>I4s', header)
            minimum = 8
            if length == 1:
                extended = file.read(8)
                if len(extended) != 8:
                    raise ValueError('Invalid MP4 extended box')
                length = struct.unpack('>Q', extended)[0]
                minimum = 16
            elif length == 0:
                length = size - offset
            if length < minimum or offset + length > size:
                raise ValueError('Invalid MP4 box size')
            boxes.add(kind)
            offset += length
    if not {b'ftyp', b'moov', b'mdat'} <= boxes:
        raise ValueError('Final demo must be a nonempty MP4 with ftyp, moov, and mdat boxes')


def select_files(root):
    files = {}
    def add(name, category):
        name = Path(name).as_posix()
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('Unsafe submission path: ' + name)
        if any(re.fullmatch(r'agents(?:\.override)?\.md', part, re.I) for part in Path(name).parts):
            raise ValueError('Agent instruction files must not be packaged')
        if not path.is_file():
            raise ValueError('Required submission file is missing: ' + name)
        files[name] = category

    for name in ['pyproject.toml', '.python-version', 'README.md', 'SUBMISSION_README.md', 'submission/README.md']:
        add(name, 'documentation' if name.endswith('.md') else 'installation')
    for folder, suffixes, category in [('src', {'.py'}, 'source'), ('configs', {'.json'}, 'configuration'),
                                       ('tests', {'.py'}, 'test'), ('scripts', {'.py'}, 'script'),
                                       ('docs', {'.md', '.pdf'}, 'research_documentation')]:
        for path in sorted((root / folder).rglob('*')):
            if path.suffix in suffixes and '__pycache__' not in path.parts:
                add(path.relative_to(root), category)
    add('notebooks/CQL_value_update_demo.ipynb', 'executed_notebook')
    notebook = json.loads((root / 'notebooks/CQL_value_update_demo.ipynb').read_text())
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    if (not code or any(cell.get('execution_count') != i for i, cell in enumerate(code, 1))
            or any(output.get('output_type') == 'error' for cell in code for output in cell.get('outputs', []))):
        raise ValueError('The submission notebook must contain successful saved Run All outputs')
    for path in sorted((root / 'reports').rglob('*')):
        if not path.is_file():
            continue
        if 'demos' in path.parts and (not (path.name.endswith('.mp4') or path.name.endswith('.mp4.json'))
                                     or 'pilot-50k' in path.parts):
            continue
        if path.suffix in {'.json', '.csv', '.png', '.txt', '.md', '.mp4'}:
            add(path.relative_to(root), 'saved_report_or_supporting_gameplay')

    datasets = []
    for name in DATASETS:
        add(name, 'frozen_dataset')
        metadata_path = str(Path(name).with_suffix('.json'))
        add(metadata_path, 'dataset_metadata')
        metadata = json.loads((root / metadata_path).read_text())
        if digest(root / name) != metadata['sha256']:
            raise ValueError('Frozen dataset checksum mismatch: ' + name)
        datasets.append({'path': name, 'metadata': metadata_path, 'transitions': metadata['transitions']})
    checkpoint_count = 0
    for name in REPORTS:
        report = json.loads((root / name).read_text())
        for run in report['runs']:
            checkpoint = run['checkpoint']
            add(checkpoint, 'final_offline_checkpoint')
            add(run['evaluation_file'], 'saved_evaluation')
            add(str(Path(checkpoint).with_suffix('.train.json')), 'training_history')
            if digest(root / checkpoint) != run['checkpoint_sha256']:
                raise ValueError('Checkpoint checksum mismatch: ' + checkpoint)
            checkpoint_count += 1
    if checkpoint_count != 57:
        raise ValueError(f'Expected the 57 reported final offline checkpoints, found {checkpoint_count}')
    for name in ['early.zip', 'final.zip', 'progress.json']:
        add('artifacts/final/ppo/' + name, 'ppo_checkpoint' if name.endswith('.zip') else 'ppo_training_history')
    for name in ['artifacts/final/results.json', 'artifacts/final/source_audit.json',
                 'artifacts/final/experiment_config.json', 'artifacts/final/experiment_seeds.json',
                 'artifacts/final/collection_timing.json', 'artifacts/final/collector_reuse.json',
                 'artifacts/ablation/study_results.json', 'artifacts/ablation/collection_timing.json',
                 'artifacts/ablation/source_audit.json', 'artifacts/ablation/session_history.json']:
        add(name, 'experiment_record')
    return files, datasets


def write_archive(root, output, files, manifest):
    """Stable ordering, timestamps, permissions, and JSON make identical inputs reproducible."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='submission-', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted([*files, 'SUBMISSION_MANIFEST.json']):
                info = zipfile.ZipInfo('cql-minigrid/' + name, FIXED_TIME)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if Path(name).suffix in {'.npz', '.zip', '.mp4', '.png', '.pdf'} else zipfile.ZIP_DEFLATED
                data = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode() if name == 'SUBMISSION_MANIFEST.json' else (root / name).read_bytes()
                archive.writestr(info, data, compresslevel=6)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC validation failed')
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_submission(root, output, mode='draft'):
    root, output = Path(root).resolve(), Path(output).resolve()
    if mode not in ['draft', 'final']:
        raise ValueError('Mode must be draft or final')
    if mode == 'final':
        validate_media(root)
    files, datasets = select_files(root)
    if mode == 'final':
        files.update({PRESENTATION: 'author_final_presentation', DEMO: 'author_final_demo'})
    entries = [{'path': name, 'category': files[name], 'bytes': (root / name).stat().st_size,
                'sha256': digest(root / name)} for name in sorted(files)]
    manifest = {'format_version': 1, 'mode': mode, 'submission_complete': mode == 'final',
                'final_media_included': mode == 'final', 'required_final_media': [PRESENTATION, DEMO],
                'proposal_is_not_final_presentation': True, 'datasets': datasets,
                'final_offline_checkpoints': 57, 'ppo_checkpoints': ['early.zip', 'final.zip'],
                'reproduction': 'All 57 reported offline agents can be evaluated; all offline runs can be retrained from the three frozen datasets and included configurations. PPO can be trained from source. Intermediate collection shards and obsolete pilots are omitted.',
                'excluded': ['Git metadata', 'virtual environments', 'caches', 'temporary files',
                             'raw pilots', 'intermediate dataset shards', 'PPO latest duplicate',
                             'duplicate PPO copies', 'GIF copies of supporting MP4 recordings', 'previous archives'],
                'files': entries}
    write_archive(root, output, files, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['draft', 'final'], default='draft')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'dist' / f'cql-minigrid-moodle-{args.mode}.zip'
    try:
        manifest = build_submission(args.root, output, args.mode)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Submission not created: {error}\n')
    print(json.dumps({'archive': str(output.resolve()), 'mode': args.mode,
                      'submission_complete': manifest['submission_complete'],
                      'files': len(manifest['files']) + 1, 'bytes': output.stat().st_size,
                      'sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()

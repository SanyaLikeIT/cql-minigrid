"""Build reproducible, Git-independent model and dataset assets for a release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

DATASETS = ('artifacts/final/datasets/strong.npz', 'artifacts/final/datasets/mixed.npz',
            'artifacts/ablation/datasets/mixed_balanced.npz')
REPORTS = ('reports/summary.json', 'reports/ablation/summary.json')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_file(root, name, expected=None):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
        raise ValueError('Artifact path must stay inside the project: ' + name)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Required artifact is missing or a symlink: ' + name)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Artifact checksum mismatch: ' + name)
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': digest}


def inventory(root):
    groups = {'models': {}, 'datasets': {}}
    experiments = {}
    for report_path in REPORTS:
        report = json.loads((root / report_path).read_text())
        experiments[report_path] = {'report_sha256': sha256(root / report_path),
                                    'training_source_revision': report['provenance']['base_git_revision'],
                                    'runs': len(report['runs'])}
        for run in report['runs']:
            name = run['checkpoint']
            groups['models'][name] = checked_file(root, name, run['checkpoint_sha256'])
            for extra in [run['evaluation_file'], str(Path(name).with_suffix('.train.json'))]:
                groups['models'][extra] = checked_file(root, extra)
    if sum(name.endswith('.pt') for name in groups['models']) != 57:
        raise ValueError('Expected all 57 reported final offline checkpoints')
    source_hashes = json.loads((root / 'artifacts/final/datasets/strong.json').read_text())['source_sha256']
    for name, source in [('final', 'strong'), ('early', 'early')]:
        path = f'artifacts/final/ppo/{name}.zip'
        groups['models'][path] = checked_file(root, path, source_hashes[source])
    for name in ['artifacts/final/ppo/progress.json', 'artifacts/final/source_audit.json',
                 'artifacts/final/experiment_config.json', 'artifacts/final/experiment_seeds.json',
                 'artifacts/final/results.json', 'artifacts/final/collection_timing.json',
                 'artifacts/ablation/study_results.json', 'artifacts/ablation/source_audit.json',
                 'artifacts/ablation/collection_timing.json']:
        groups['models'][name] = checked_file(root, name)
    for name in DATASETS:
        metadata_path = str(Path(name).with_suffix('.json'))
        metadata = json.loads((root / metadata_path).read_text())
        groups['datasets'][name] = {**checked_file(root, name, metadata['sha256']),
                                    'transitions': metadata['transitions']}
        groups['datasets'][metadata_path] = checked_file(root, metadata_path)
    return groups, experiments


def write_bundle(root, output, entries):
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='release-', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for entry in sorted(entries, key=lambda item: item['path']):
                # Recheck at write time so an altered input cannot get a stale manifest hash.
                name = entry['path']
                checked_file(root, name, entry['sha256'])
                info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if Path(name).suffix in ['.npz', '.zip'] else zipfile.ZIP_DEFLATED
                archive.writestr(info, (root / name).read_bytes(), compresslevel=6)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC validation failed')
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    groups, experiments = inventory(root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'format_version': 1, 'experiments': experiments, 'bundles': {},
                'extraction': 'Extract ZIP files at the project root; paths start with artifacts/.',
                'excluded': ['source code', 'obsolete pilots', 'intermediate shards', 'PPO latest duplicate', 'recordings']}
    for group, mapping in groups.items():
        filename = f'cql-minigrid-{group}.zip'
        entries = [mapping[name] for name in sorted(mapping)]
        path = output / filename
        write_bundle(root, path, entries)
        manifest['bundles'][filename] = {'bytes': path.stat().st_size, 'sha256': sha256(path), 'files': entries}
    manifest_path = output / 'artifacts-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    names = [*manifest['bundles'], manifest_path.name]
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(output / name)}  {name}\n' for name in sorted(names)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'dist/release-assets'
    try:
        manifest = build(args.root, output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Assets not built: {error}\n')
    print(json.dumps({name: {k: value[k] for k in ['bytes', 'sha256']}
                      for name, value in manifest['bundles'].items()}, indent=2))


if __name__ == '__main__':
    main()

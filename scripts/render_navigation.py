"""Render the frozen navigation study; does not train or select policies."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from cql_minigrid.common import sha256
from cql_minigrid.rollout import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='artifacts/navigation-geometry')
    parser.add_argument('--output', default='reports/navigation')
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    result = json.loads((root / 'results.json').read_text())
    if not result['test_runs']:
        parser.error('Finalize the study before rendering held-out results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    selected = result['selection']['candidate']
    rows = [next(r for r in result['test_aggregate'] if r['candidate'] == name)
            for name in ['baseline', selected]]
    labels = ['Original CQL', 'Improved CQL']
    metrics = [('success_rate', 'Success %', 100), ('collision_rate', 'Collision %', 100),
               ('timeout_rate', 'Timeout %', 100), ('stall_rate', 'Stationary run >=16 %', 100),
               ('successful_steps_mean', 'Successful steps', 1), ('return_mean', 'Return', 1)]
    table = ['| Policy | ' + ' | '.join(title for _, title, _ in metrics) + ' |',
             '|---|' + '---:|' * len(metrics)]
    for label, row in zip(labels, rows, strict=True):
        values = [f"{row[key]['mean'] * scale:.2f} ± {row[key]['std'] * scale:.2f}"
                  for key, _, scale in metrics]
        table.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    (output / 'results_table.md').write_text('\n'.join(table) + '\n')
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), layout='constrained')
    for ax, key, title in zip(axes, ['success_rate', 'collision_rate', 'stall_rate', 'successful_steps_mean'],
                               ['Success', 'Collision', 'Stationary run >=16', 'Steps on success'], strict=True):
        ax.bar(labels, [row[key]['mean'] for row in rows],
               yerr=[row[key]['std'] for row in rows], color=['#777777', '#349b67'], capsize=4)
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if key.endswith('_rate'):
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"{len(result['test_seeds'])} fresh test seeds per policy; mean ± sample SD over 3 training seeds")
    fig.savefig(output / 'metrics.png', dpi=180)
    plt.close(fig)
    demos = output / 'demos'
    demos.mkdir(exist_ok=True)
    for path in (root / 'replays').iterdir():
        if path.suffix in ['.mp4', '.gif', '.json']:
            shutil.copy2(path, demos / path.name)
    # Recompute metrics from complete episodes, independently of the plot aggregates.
    audited_episodes = 0
    for run in result['test_runs']:
        if sha256(run['checkpoint']) != run['checkpoint_sha256']:
            raise ValueError('Test checkpoint checksum mismatch')
        report = json.loads((root / 'test' / run['candidate'] / f"seed-{run['seed']}.json").read_text())
        if (report['seeds'] != result['test_seeds']
                or [e['seed'] for e in report['episodes']] != result['test_seeds']
                or summarize(report['episodes']) != run['metrics']):
            raise ValueError('Test episode records disagree with reported metrics')
        audited_episodes += len(report['episodes'])
    if sha256(root / 'best.pt') != result['selection']['representative']['checkpoint_sha256']:
        raise ValueError('Selected checkpoint alias checksum mismatch')
    for path in demos.glob('*.json'):
        meta = json.loads(path.read_text())
        if sha256(demos / path.name.removesuffix('.json')) != meta['file_sha256']:
            raise ValueError('Demonstration checksum mismatch')
    audit = {'test_episode_records_verified': audited_episodes,
             'test_environment_seeds': result['test_seeds'],
             'best_checkpoint_sha256': sha256(root / 'best.pt'),
             'source_results_sha256': sha256(root / 'results.json'),
             'renderer_sha256': sha256(__file__),
             'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                       if p.is_file() and p.name != 'integrity.json'}}
    (output / 'integrity.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()

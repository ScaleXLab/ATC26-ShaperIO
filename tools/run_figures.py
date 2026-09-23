#!/usr/bin/env python3
"""Run the selected figures on PM9A3, one measurement per point by default."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_runner import ROOT, load_config, make_plan

SERIES = {'fig1a': ('bam', 'gds'), 'fig1b': ('bam',),
          'fig1c': ('bam',), 'fig3': ('bam', 'gds', 'shaperio')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panels', nargs='+', choices=SERIES)
    parser.add_argument('--device', type=Path, default=ROOT / 'configs/device.json')
    parser.add_argument('--build', type=Path, default=ROOT / 'build')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pm9a3')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--threads', type=int, nargs='+')
    parser.add_argument('--backend', choices=('bam', 'gds', 'shaperio'))
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--allow-write', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    if len(set(args.panels)) != len(args.panels):
        parser.error('Specify each panel once')
    if not args.plan and not args.allow_write:
        parser.error('Experiments format the configured SSD; pass --allow-write')
    device = json.loads(args.device.read_text())
    plans = []
    for panel in args.panels:
        profile = 'fig1ab' if panel in ('fig1a', 'fig1b') else panel
        config = json.loads((ROOT / f'configs/{profile}.json').read_text())
        config.update(device, repetitions=args.repetitions)
        backends = [args.backend] if args.backend else SERIES[panel]
        for backend in backends:
            if backend not in SERIES[panel]:
                parser.error(f'{panel} supports {", ".join(SERIES[panel])}')
            jobs = make_plan(panel, backend, config, args.build.resolve(), selected_threads=args.threads)
            plans.append((panel, backend, config, jobs))
    if args.plan:
        print(json.dumps([{'panel': p, 'backend': b, 'config': c, 'jobs': j}
                          for p, b, c, j in plans], indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for panel, backend, config, jobs in plans:
        path = args.output.resolve() / f'{panel}-{backend}.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        load_config(path)
        output = args.output.resolve() / f'{panel}-{backend}'
        command = [sys.executable, str(ROOT / 'tools/experiment_runner.py'), 'run', panel,
                   '--backend', backend, '--config', str(path), '--build', str(args.build.resolve()),
                   '--output', str(output), '--allow-write']
        if args.threads:
            command += ['--threads', *map(str, args.threads)]
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, str(ROOT / 'tools/experiment_runner.py'), 'aggregate', str(output)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/plot_figures.py'), '--input', str(args.output.resolve()),
                    '--output', str(args.output.resolve() / 'figures')], check=True)


if __name__ == '__main__':
    main()

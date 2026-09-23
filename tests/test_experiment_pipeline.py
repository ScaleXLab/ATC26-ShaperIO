import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import experiment_runner
from measurement_records import load_runs
from run_figures import SERIES
import prepare_pm9a3


def profile(panel):
    name = 'fig1ab' if panel in ('fig1a', 'fig1b') else panel
    config = json.loads((ROOT / f'configs/{name}.json').read_text())
    config.update(json.loads((ROOT / 'configs/device.json').read_text()))
    return config


class ExperimentPipeline(unittest.TestCase):
    def test_commands_match_reference_measurements(self):
        count = 0
        for path in (ROOT / 'data/reference').glob('*/run.json'):
            run = json.loads(path.read_text())
            panel, backend = run['panel'], run['backend']
            new = experiment_runner.make_plan(panel, backend, profile(panel), Path('/build'))
            by_condition = {(job['x'], job['condition']): job for job in new}
            for job in run['jobs']:
                plan = by_condition[job['x'], job['condition']]
                self.assertEqual(len(plan['steps']), len(job['steps']))
                for actual, expected in zip(plan['steps'], job['steps']):
                    self.assertEqual(actual['argv'][1:], expected['argv'][1:])
                    self.assertEqual(actual['expected_bytes'], expected['expected_bytes'])
                    count += 1
        self.assertEqual(count, 90)

    def test_all_profiles_validate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            for panel in SERIES:
                config = profile(panel)
                path.write_text(json.dumps(config))
                self.assertEqual(experiment_runner.load_config(path), config)

    def test_gpu_reader_and_window(self):
        config = profile('fig3')
        jobs = experiment_runner.make_plan('fig3', 'shaperio', config, Path('/build'))
        for job in jobs:
            write, read = job['steps']
            self.assertEqual(write['argv'][write['argv'].index('--window') + 1], '256')
            self.assertEqual(read['argv'][read['argv'].index('--backend') + 1], 'bam')
            self.assertEqual(read['argv'][read['argv'].index('--threads') + 1], '32')
            self.assertEqual(write['expected_bytes'], max(1 << 30, job['x'] * (2 << 20)))

    def test_queue_depth_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            config = profile('fig3')
            config['queue_depth'] = 32
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'exceed'):
                experiment_runner.load_config(path)

    def test_device_serial_mismatch_stops_identification(self):
        with tempfile.TemporaryDirectory() as temp:
            pci = Path(temp)
            (pci / 'nvme/nvme1').mkdir(parents=True)
            with patch.object(prepare_pm9a3, 'PCI', pci), patch.object(prepare_pm9a3, 'run') as run:
                run.return_value = json.dumps({'sn': 'another-ssd', 'mn': prepare_pm9a3.MODEL})
                with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                    prepare_pm9a3.identify()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[:2], ('nvme', 'id-ctrl'))

    def test_partition_mounts_unmounted_before_namespace(self):
        tree = {'blockdevices': [{'path': '/dev/nvme1n1', 'mountpoints': [None],
                                'children': [{'path': '/dev/nvme1n1p1', 'mountpoints': ['/mnt/partition']}]}]}
        with patch.object(prepare_pm9a3, 'run', return_value=json.dumps(tree)) as run, \
             patch.object(prepare_pm9a3.subprocess, 'run') as findmnt, \
             contextlib.redirect_stdout(io.StringIO()):
            findmnt.return_value = SimpleNamespace(returncode=1, stdout='', stderr='')
            prepare_pm9a3.unmount(Path('/dev/nvme1n1'))
            self.assertIn((('umount', '/mnt/partition'),), run.call_args_list)

    def test_read_sweep_prepares_once_per_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'build/bin/gpu-io-benchmark'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/usr/bin/env python3\nimport json, sys\n'
                              'a = sys.argv\nprint(json.dumps({"op": a[a.index("--op")+1],'
                              '"bytes": int(a[a.index("--bytes")+1]),'
                              '"seconds": 1.0, "bandwidth_gib_s": 1.0}))\n')
            binary.chmod(0o755)
            device = root / 'device'
            device.touch()
            config = profile('fig1c')
            config.update(controller=str(device), reset_each_condition=False, repetitions=2)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            args = SimpleNamespace(panel='fig1c', backend='bam', config=path,
                                   build=root / 'build', output=root / 'runs/fig1c-bam',
                                   no_prewrite=False, allow_write=True)
            jobs = experiment_runner.make_plan('fig1c', 'bam', config, args.build, selected_threads=[1, 16])
            with contextlib.redirect_stdout(io.StringIO()):
                experiment_runner.execute(args, config, jobs)
                experiment_runner.aggregate(args.output)
            run = json.loads((args.output / 'run.json').read_text())
            self.assertEqual(sum(r['phase'] == 'prepare' for r in run['records']), 2)
            self.assertEqual(sum(r['phase'] == 'read' for r in run['records']), 4)
            self.assertEqual(len(load_runs(root / 'runs')), 2)

    def test_raw_log_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'fig3-gds'
            shutil.copytree(ROOT / 'data/reference/fig3-gds', target)
            run = json.loads((target / 'run.json').read_text())
            log = target / run['records'][0]['log']
            log.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Raw log mismatch'):
                load_runs(Path(temp))

    def test_duplicate_points_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('first', 'second'):
                shutil.copytree(ROOT / 'data/reference/fig3-gds', Path(temp) / name)
            with self.assertRaisesRegex(ValueError, 'Duplicate summary point'):
                load_runs(Path(temp))

    def test_changed_method_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('a', 'b'):
                shutil.copytree(ROOT / f'data/reference/fig3-bam-{name}', Path(temp) / name)
            path = Path(temp) / 'b/run.json'
            run = json.loads(path.read_text())
            run['config']['read_io_bytes'] = 4096
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'Inconsistent configuration'):
                load_runs(Path(temp))


if __name__ == '__main__':
    unittest.main()

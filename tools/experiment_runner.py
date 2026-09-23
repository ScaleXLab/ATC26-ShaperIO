#!/usr/bin/env python3
"""Plan, run, and aggregate the PM9A3 experiments."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = ('fig1a', 'fig1b', 'fig1c', 'fig3')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path):
    c = json.loads(path.read_text())
    for key in ('namespace_id', 'queues', 'queue_depth', 'total_bytes',
                'buffer_bytes', 'shaperio_window', 'ring_entries', 'repetitions', 'timeout_seconds'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['total_bytes'] % (65536 * 4096):
        raise ValueError('total_bytes must be a multiple of 256 MiB')
    pages = c['total_bytes'] // 65536
    if pages & (pages - 1):
        raise ValueError('total_bytes/65536 must be a power of two')
    if not 1 <= c['shaperio_window'] <= 16:
        raise ValueError('shaperio_window must be 1..16')
    write_window = c.get('shaperio_write_window', c['shaperio_window'])
    if type(write_window) is not int or not 1 <= write_window <= 256:
        raise ValueError('shaperio_write_window must be 1..256')
    if c.get('write_submit_mode', 'batch') == 'batch' and c['queue_depth'] <= c.get('knee_limit', 32):
        raise ValueError('Batched write queue depth must exceed the knee limit')
    for key in ('read_io_bytes', 'write_io_bytes'):
        if c.get(key, 65536) not in (4096, 65536):
            raise ValueError(f'{key} must be 4096 or 65536')
    for key in ('read_tile_pages', 'write_tile_pages'):
        tile = c.get(key, 0)
        if type(tile) is not int or tile < 0 or (tile and tile & (tile - 1)):
            raise ValueError(f'{key} must be zero (chunk) or a power of two')
    if type(c.get('reuse_prepared_reads', False)) is not bool:
        raise ValueError('reuse_prepared_reads must be boolean')
    if c.get('write_submit_mode', 'batch') not in ('batch', 'serial'):
        raise ValueError('write_submit_mode must be batch or serial')
    if c.get('worker_order', 'linear') not in ('linear', 'warp-transpose'):
        raise ValueError('worker_order must be linear or warp-transpose')
    if c.get('postwrite_per_thread_bytes'):
        value = c['postwrite_per_thread_bytes']
        if type(value) is not int or value % c.get('write_io_bytes', 65536) or value & (value - 1):
            raise ValueError('Per-thread bytes must be a positive aligned power of two')
    return c


def make_plan(panel, backend, c, build, no_prewrite=False, selected_threads=None):
    if panel not in PANELS or backend not in ('bam', 'gds', 'shaperio'):
        raise ValueError('Unknown panel or backend')
    if panel in ('fig1b', 'fig1c') and backend != 'bam':
        raise ValueError(f'{panel} uses the BaM reader')
    if no_prewrite and panel != 'fig1c':
        raise ValueError('--no-prewrite is only valid for fig1c')
    jobs = []
    base = c['gds_offset_bytes'] if backend == 'gds' else c['offset_bytes']

    def transfer(op, threads, pattern='sequential', actual_backend=None, latency=False, name=None,
                 bytes_count=None, batch_bytes=None):
        actual_backend = actual_backend or backend
        bytes_count = bytes_count or c['total_bytes']
        args = ['--op', op, '--pattern', pattern, '--threads', threads,
                '--bytes', bytes_count, '--io', c.get(op + '_io_bytes', 65536), '--offset', base,
                '--gpu', c['gpu'], '--seed', c['seed'], '--latency', int(latency)]
        if c.get(op + '_tile_pages', 0):
            args += ['--tile-pages', c[op + '_tile_pages']]
        if c.get('worker_order', 'linear') != 'linear':
            args += ['--worker-order', c['worker_order']]
        if actual_backend == 'gds':
            binary = 'gds-io-benchmark'
            args += ['--file', c['gds_file']]
        else:
            binary = 'gpu-io-benchmark'
            args += ['--controller', c['controller'], '--backend', actual_backend,
                     '--nsid', c['namespace_id'], '--queues', c['queues'], '--qd', c['queue_depth'],
                     '--buffer-bytes', c['buffer_bytes'], '--ring', c['ring_entries'],
                     '--window', (c.get('shaperio_write_window', c['shaperio_window']) if op == 'write'
                                  else c['shaperio_window']) if actual_backend == 'shaperio' else 1]
            if batch_bytes:
                args += ['--batch-bytes', batch_bytes]
        if op == 'write':
            args += ['--allow-write', 1]
        return {'phase': name or op, 'writes': op == 'write', 'expected_bytes': bytes_count,
                'argv': [str(build / 'bin' / binary)] + [str(x) if x is not None else 'UNSET' for x in args]}

    def postwrite(threads):
        per_worker = c.get('postwrite_per_thread_bytes')
        if not per_worker or backend == 'gds':
            return transfer('write', threads)
        batch = threads * per_worker
        batches = max(1, (c['total_bytes'] + batch - 1) // batch)
        if batches > c.get('postwrite_max_batches', 8):
            return transfer('write', threads)
        return transfer('write', threads, bytes_count=batch * batches, batch_bytes=batch)

    def append(x, tag, phases):
        jobs.append({'panel': panel, 'backend': backend, 'x': x, 'condition': tag, 'steps': phases})

    def prepare():
        return [] if no_prewrite else [transfer('write', 1, name='prepare')]

    if panel in ('fig1a', 'fig3'):
        sweep = [1, 8, 64, 512, 2048, 4096] if panel == 'fig1a' else [1, 8, 64, 128, 256, 512, 1024, 4096]
        for threads in sweep:
            if backend == 'gds' and threads > 64:
                continue
            # Same direct GPU reader isolates the preceding GPU write ordering.
            readers = 32 if panel == 'fig3' else (64 if backend == 'gds' else 128)
            read_backend = 'gds' if backend == 'gds' else 'bam'
            append(threads, f'readers={readers}', [postwrite(threads),
                   transfer('read', readers, actual_backend=read_backend, name='postwrite-read')])
    elif panel == 'fig1b':
        for writers in (1, 512):
            for readers in (1, 4, 16, 64, 128, 256):
                append(readers, f'writers={writers}', [postwrite(writers),
                       transfer('read', readers, actual_backend='bam', name='postwrite-read')])
    elif panel == 'fig1c':
        for threads in (1, 16, 256, 1024, 4096):
            if backend == 'gds' and threads > 64:
                continue
            append(threads, 'read-latency', prepare() + [transfer('read', threads, latency=True)])
    if selected_threads:
        jobs = [job for job in jobs if job['x'] in selected_threads]
    if not jobs:
        raise ValueError('No selected data points')
    return jobs


def execute(args, config, jobs):
    reuse_reads = (config.get('reuse_prepared_reads', False) and not args.no_prewrite
                   and args.panel == 'fig1c')
    if config.get('reset_each_condition'):
        from prepare_pm9a3 import configure
        configure(config)
    if any(step['writes'] for job in jobs for step in job['steps']) and not args.allow_write:
        raise ValueError('This plan writes storage; --allow-write is required')
    if args.backend == 'gds':
        path = Path(config['gds_file'] or '')
        if not config['gds_file'] or (not path.is_file() and not config.get('reset_each_condition')):
            raise ValueError('Select an existing GDS experiment file')
        if not config.get('reset_each_condition') and path.stat().st_size < config['gds_offset_bytes'] + config['total_bytes']:
            raise ValueError('GDS file is smaller than the requested range')
    else:
        if not config['controller'] or (not Path(config['controller']).exists() and not config.get('reset_each_condition')):
            raise ValueError('A dedicated libnvm controller is not configured/available')
        need = max(
            step['expected_bytes'] for job in jobs for step in job['steps'])
        if type(config['offset_bytes']) is not int or config['offset_bytes'] < 0 or config['offset_bytes'] % 65536:
            raise ValueError('Configure an aligned raw-device offset')
        if type(config['scratch_bytes']) is not int or config['scratch_bytes'] < need:
            raise ValueError('Configured raw scratch range is too small')
    binaries = {step['argv'][0] for job in jobs for step in job['steps']}
    if any(not Path(binary).is_file() for binary in binaries):
        raise ValueError('Build all binaries referenced by this plan')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = {'status': 'running', 'config': config,
            'panel': args.panel, 'backend': args.backend, 'jobs': jobs,
            'no_prewrite': args.no_prewrite, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'binary_sha256': {p: sha(Path(p)) for p in binaries}, 'records': []}
    meta['reset_tool_sha256'] = sha(ROOT / 'tools/prepare_pm9a3.py')
    meta['source_manifest_sha256'] = sha(ROOT / 'provenance/source.json')
    library = args.build.resolve() / 'lib/libnvm.so'
    if library.is_file():
        meta['libnvm_sha256'] = sha(library)
    meta['resets'] = []
    meta['read_preparation_protocol'] = 'once-per-repeat' if reuse_reads else 'per-condition'
    env = os.environ.copy()
    env['CUDA_MODULE_LOADING'] = 'EAGER'
    env['CUFILE_ENV_PATH_JSON'] = str(ROOT / 'configs/cufile.direct.json')
    meta['cufile_config_sha256'] = sha(Path(env['CUFILE_ENV_PATH_JSON']))
    meta['cuda_module_loading'] = env['CUDA_MODULE_LOADING']
    def save():
        (out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n')
    save()
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print('Stop requested; completing the current condition before exiting.', flush=True)
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        for repeat in range(config['repetitions']):
            prepared = False
            order = list(range(len(jobs)))
            random.Random(config['seed'] + repeat).shuffle(order)
            for index in order:
                if stopping:
                    raise InterruptedError('Stopped at a condition boundary')
                job = jobs[index]
                prepare_condition = not reuse_reads or not prepared
                if config.get('reset_each_condition') and prepare_condition:
                    if not args.allow_write:
                        raise ValueError('Device reset requires --allow-write')
                    name = f'{repeat + 1:02d}-{index:03d}-device-reset.log'
                    command = [sys.executable, str(ROOT / 'tools/prepare_pm9a3.py'),
                               'format-gds' if args.backend == 'gds' else 'format-raw',
                               '--config', str(args.config.resolve()),
                               '--allow-destroy', '--file-bytes', str(config['total_bytes'])]
                    print(name, flush=True)
                    with (out / name).open('w') as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=config['timeout_seconds'])
                    meta['resets'].append({'repeat': repeat + 1, 'job': index, 'argv': command, 'log': name})
                    save()
                for step in job['steps']:
                    if step['phase'] == 'prepare' and not prepare_condition:
                        continue
                    name = f'{repeat + 1:02d}-{index:03d}-{step["phase"]}.log'
                    print(name, flush=True)
                    with (out / name).open('w') as log:
                        status = subprocess.run(step['argv'], stdout=log, stderr=subprocess.STDOUT,
                                                cwd=out, env=env, timeout=config['timeout_seconds'])
                    if status.returncode:
                        raise RuntimeError(f'{name}: exit {status.returncode}; see raw log')
                    records = []
                    for line in (out / name).read_text(errors='replace').splitlines():
                        if line.startswith('{'):
                            row = json.loads(line)
                            if row.get('backend') == 'shaperio' and row.get('op') == 'write':
                                if row.get('write_submit_mode', 'batch') != config.get('write_submit_mode', 'batch'):
                                    raise ValueError('Compiled scheduler write submission mode does not match config')
                                if row.get('knee_limit') != config.get('knee_limit', 32):
                                    raise ValueError('Compiled scheduler knee limit does not match config')
                            for key in ('bandwidth_gib_s', 'logical_page_bandwidth_gib_s', 'seconds', 'p50_us', 'p99_us'):
                                if key in row and (not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] <= 0):
                                    raise ValueError(f'{name}: invalid {key}')
                            if 'seconds' not in row or not any(k in row for k in ('bandwidth_gib_s', 'logical_page_bandwidth_gib_s')):
                                raise ValueError(f'{name}: incomplete measurement')
                            if row.get('bytes') != step['expected_bytes']:
                                raise ValueError(f'{name}: byte count mismatch')
                            records.append({'repeat': repeat + 1, 'job': index, 'phase': step['phase'],
                                            'log': name, 'measurement': row})
                    if not records:
                        raise RuntimeError(f'{name}: missing JSON measurement')
                    meta['records'].extend(records)
                    save()
                prepared = True
        meta['status'] = 'completed'
    except BaseException as error:
        meta['status'] = 'failed'
        meta['error'] = str(error)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


def aggregate(directory):
    meta = json.loads((directory / 'run.json').read_text())
    if meta['status'] != 'completed':
        raise ValueError('Run is incomplete or failed')
    groups = defaultdict(list)
    for record in meta['records']:
        if record['phase'] == 'prepare':
            continue
        job = meta['jobs'][record['job']]
        row = record['measurement']
        for key in ('bandwidth_gib_s', 'logical_page_bandwidth_gib_s', 'p50_us', 'p99_us',
                    'ssd_read_bandwidth_gib_s', 'ssd_write_bandwidth_gib_s'):
            if key in row:
                group = (job['panel'], job['backend'], job['x'], job['condition'], record['phase'], row['op'], key)
                groups[group].append((record['repeat'], row[key]))
    for key, values in groups.items():
        if len(values) != meta['config']['repetitions'] or len({r for r, v in values}) != len(values):
            raise ValueError(f'Missing or duplicated repetition: {key}')
    with (directory / 'summary.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['panel', 'backend', 'x', 'condition', 'phase', 'op', 'metric', 'n', 'median', 'min', 'max'])
        for key, items in sorted(groups.items()):
            values = [value for repeat, value in items]
            writer.writerow([*key, len(values), statistics.median(values), min(values), max(values)])
    print(directory / 'summary.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'run'):
        p = sub.add_parser(command)
        p.add_argument('panel', choices=PANELS)
        p.add_argument('--backend', choices=('bam', 'gds', 'shaperio'), required=True)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--build', type=Path, default=ROOT / 'build')
        p.add_argument('--threads', type=int, nargs='+')
        p.add_argument('--no-prewrite', action='store_true')
        if command == 'run':
            p.add_argument('--allow-write', action='store_true')
            p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('aggregate')
    p.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'aggregate':
        aggregate(args.directory)
        return
    config = load_config(args.config)
    jobs = make_plan(args.panel, args.backend, config, args.build.resolve(), args.no_prewrite, args.threads)
    if args.command == 'plan':
        print(json.dumps({'config': config, 'jobs': jobs}, indent=2))
    else:
        execute(args, config, jobs)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))

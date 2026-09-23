#!/usr/bin/env python3
"""Verify source snapshots, reference logs, summaries, and figure coverage."""
import hashlib
import json
from pathlib import Path

from measurement_records import load_runs

ROOT = Path(__file__).resolve().parents[1]


def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def main():
    count = 0
    for manifest in ('source', 'benchmark-snapshot'):
        for item in json.loads((ROOT / f'provenance/{manifest}.json').read_text())['files']:
            check(ROOT / item['path'], item['sha256'])
            count += 1
    measurements = json.loads((ROOT / 'provenance/measurements.json').read_text())
    for run in measurements['runs']:
        for name, digest in run['files'].items():
            check(ROOT / run['path'] / name, digest)
    rows = load_runs(ROOT / 'data/reference')
    expected = {
        ('fig1a', 'bam', 'readers=128'): {1, 8, 64, 512, 2048, 4096},
        ('fig1a', 'gds', 'readers=64'): {1, 8, 64},
        ('fig1b', 'bam', 'writers=1'): {1, 4, 16, 64, 128, 256},
        ('fig1b', 'bam', 'writers=512'): {1, 4, 16, 64, 128, 256},
        ('fig1c', 'bam', 'read-latency'): {1, 16, 256, 1024, 4096},
        ('fig3', 'bam', 'readers=32'): {1, 8, 64, 128, 256, 512, 1024, 4096},
        ('fig3', 'shaperio', 'readers=32'): {1, 8, 64, 128, 256, 512, 1024, 4096},
        ('fig3', 'gds', 'readers=32'): {1, 8, 64},
    }
    actual = {}
    for row in rows:
        if row['phase'] in ('read', 'postwrite-read') and row['metric'] == 'bandwidth_gib_s':
            actual.setdefault(tuple(row[k] for k in ('panel', 'backend', 'condition')), set()).add(int(row['x']))
    if actual != expected:
        raise ValueError(f'Figure coverage mismatch: {actual}')
    print(f'OK: {count} source files, {len(measurements["runs"])} runs, '
          f'{sum(map(len, actual.values()))} figure points, raw logs and summaries verified.')


if __name__ == '__main__':
    main()

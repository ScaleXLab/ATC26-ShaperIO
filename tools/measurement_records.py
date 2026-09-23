"""Read completed runs and validate their summaries against raw measurements."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEYS = ('panel', 'backend', 'x', 'condition', 'phase', 'op', 'metric')
METRICS = ('bandwidth_gib_s', 'p50_us', 'p99_us')


def load_runs(directory):
    rows, seen, methods = [], set(), {}
    for summary in sorted(Path(directory).glob('*/summary.csv')):
        run = json.loads((summary.parent / 'run.json').read_text())
        if run['status'] != 'completed':
            raise ValueError(f'Incomplete run: {summary.parent}')
        config = {k: v for k, v in run['config'].items() if k != 'state_protocol'}
        identity = (run['panel'], run['backend'])
        method = (config, run['binary_sha256'])
        if identity in methods and method != methods[identity]:
            raise ValueError(f'Inconsistent configuration or binary for {identity}')
        methods[identity] = method
        groups = defaultdict(list)
        record_ids = set()
        for record in run['records']:
            rid = (record['repeat'], record['job'], record['phase'])
            if rid in record_ids:
                raise ValueError(f'Duplicate measurement: {rid}')
            record_ids.add(rid)
            raw = [json.loads(line) for line in (summary.parent / record['log']).read_text().splitlines()
                   if line.startswith('{')]
            if raw != [record['measurement']]:
                raise ValueError(f'Raw log mismatch: {record["log"]}')
            if record['phase'] == 'prepare':
                continue
            job = run['jobs'][record['job']]
            measurement = record['measurement']
            step = next(s for s in job['steps'] if s['phase'] == record['phase'])
            if measurement['bytes'] != step['expected_bytes']:
                raise ValueError('Measured byte count differs from the command plan')
            for metric in METRICS:
                if metric in measurement:
                    value = measurement[metric]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f'Invalid {metric}: {value}')
                    key = (job['panel'], job['backend'], str(job['x']), job['condition'],
                           record['phase'], measurement['op'], metric)
                    groups[key].append((record['repeat'], value))
        for repeat in range(1, run['config']['repetitions'] + 1):
            for index, job in enumerate(run['jobs']):
                for step in job['steps']:
                    if step['phase'] != 'prepare' and (repeat, index, step['phase']) not in record_ids:
                        raise ValueError(f'Missing measurement: {repeat}, {index}, {step["phase"]}')
        with summary.open(newline='') as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(groups):
            raise ValueError(f'Summary row count mismatch: {summary}')
        for row in actual:
            key = tuple(row[k] for k in KEYS)
            if key in seen:
                raise ValueError(f'Duplicate summary point: {key}')
            seen.add(key)
            pairs = groups[key]
            if {repeat for repeat, value in pairs} != set(range(1, run['config']['repetitions'] + 1)):
                raise ValueError(f'Missing repetition: {key}')
            values = [value for repeat, value in pairs]
            expected = dict(n=len(values), median=statistics.median(values), min=min(values), max=max(values))
            if any(not math.isclose(float(row[k]), v, rel_tol=1e-12) for k, v in expected.items()):
                raise ValueError(f'Summary values differ from raw records: {key}')
            rows.append({**row, 'source': str(summary.parent)})
    if not rows:
        raise ValueError(f'No measured summaries in {directory}')
    return rows

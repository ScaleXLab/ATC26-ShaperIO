#!/usr/bin/env python3
"""Generate the measured figures using the paper's typography and series styles."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from measurement_records import load_runs

ROOT = Path(__file__).resolve().parents[1]
PANELS = ('fig1a', 'fig1b', 'fig1c', 'fig3')
STYLES = {
    'gds': ('GDS', '#9467bd', 'D', ':'),
    'bam': ('BaM', '#1f77b4', 'o', '-'),
    'shaperio': ('ShaperIO', '#d62728', 's', '-'),
}
CAPTIONS = {'fig1a': 'Figure 1(a)', 'fig1b': 'Figure 1(b)',
            'fig1c': 'Figure 1(c)', 'fig3': 'Figure 3: PM9A3'}


def thread_label(value):
    return f'{value // 1024}K' if value >= 1024 else str(value)


def draw_latency(ax, rows):
    points = defaultdict(dict)
    for row in rows:
        if row['phase'] == 'read':
            points[int(row['x'])][row['metric']] = float(row['median'])
    points = sorted(points.items())
    for metric, label, marker, color in [('p99_us', 'P99', 's', '#c0392b'),
                                        ('p50_us', 'P50', '^', '#e67e22')]:
        ax.plot([p['bandwidth_gib_s'] for _, p in points],
                [p[metric] / 1000 for _, p in points],
                marker=marker, color=color, label=label, markersize=5.5,
                linewidth=1.8, zorder=3)
    # Offsets keep labels separate where the high-concurrency curve turns back.
    offsets = {1: (-8, 6, 'right'), 16: (-7, 10, 'right'),
               256: (0, 18, 'center'), 1024: (-9, 1, 'right'),
               4096: (7, 0, 'left')}
    for threads, point in points:
        dx, dy, align = offsets.get(threads, (6, 6, 'left'))
        ax.annotate(f'T={thread_label(threads)}',
                    (point['bandwidth_gib_s'], point['p99_us'] / 1000),
                    xytext=(dx, dy), textcoords='offset points', fontsize=10,
                    fontweight='bold', color='#333333', ha=align, va='center')
    upper = max(p['p99_us'] / 1000 for _, p in points)
    lower = min(p['p50_us'] / 1000 for _, p in points)
    ax.set(xlabel='Read bandwidth (GiB/s)', ylabel='Latency (ms)', yscale='log',
           xlim=(0, max(7, max(p['bandwidth_gib_s'] for _, p in points) * 1.12)),
           ylim=(min(.01, lower / 4), max(200, upper * 2)))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis='both', alpha=.25, linewidth=.5)
    ax.legend(loc='upper right', ncol=2, handlelength=1.4, columnspacing=.8,
              prop={'size': 9.5, 'weight': 'bold'})


def draw_bandwidth(ax, panel, rows):
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'postwrite-read' and row['metric'] == 'bandwidth_gib_s':
            groups[row['backend'], row['condition']].append(row)
    ticks = sorted({int(row['x']) for series in groups.values() for row in series})
    positions = {threads: index for index, threads in enumerate(ticks)}
    for (backend, condition), series in sorted(groups.items(),
            key=lambda item: (tuple(STYLES).index(item[0][0]), item[0][1])):
        series.sort(key=lambda row: int(row['x']))
        label, color, marker, style = STYLES[backend]
        if panel == 'fig1b':
            writers = int(condition.split('=')[1])
            label = f'After {writers}-thread write'
            color, marker = ('#2ca02c', 'o') if writers == 1 else ('#d62728', 's')
        y = [float(row['median']) for row in series]
        bounds = [[v - float(row['min']) for v, row in zip(y, series)],
                  [float(row['max']) - v for v, row in zip(y, series)]]
        ax.errorbar([positions[int(row['x'])] for row in series], y, yerr=bounds,
                    label=label, color=color, marker=marker, linestyle=style,
                    linewidth=1.8, markersize=5.5, markeredgewidth=.6,
                    elinewidth=.8, capsize=2, zorder=3)
    ax.set_xticks(range(len(ticks)), [thread_label(t) for t in ticks],
                  rotation=45 if panel == 'fig3' else 0,
                  ha='right' if panel == 'fig3' else 'center')
    upper = max(float(row['max']) for series in groups.values() for row in series)
    ax.set(xlabel='Read threads' if panel == 'fig1b' else 'Write threads',
           ylabel='Read BW (GiB/s)' if panel == 'fig1b' else 'Post-write Read\nBW (GiB/s)',
           xlim=(-.3, len(ticks) - .7), ylim=(0, max(7.5 if panel == 'fig3' else 8.5, upper * 1.22)))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis='y', alpha=.25, linewidth=.5)
    ax.legend(loc='upper left' if panel == 'fig1b' else 'upper right',
              prop={'size': 9.5, 'weight': 'normal' if panel == 'fig3' else 'bold'})
    if panel == 'fig3':
        ax.set_title('PM9A3', fontsize=12, pad=7)
        # Keep the legend below the ShaperIO curve in the single-SSD panel.
        ax.get_legend().set_bbox_to_anchor((1, .79))


def draw(ax, panel, rows):
    if panel == 'fig1c':
        draw_latency(ax, rows)
    else:
        draw_bandwidth(ax, panel, rows)
    weight = 'normal' if panel == 'fig3' else 'bold'
    ax.xaxis.label.set_weight(weight)
    ax.yaxis.label.set_weight(weight)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(weight)
    ax.tick_params(axis='both', which='major', pad=3, length=3, width=.7)
    ax.set_axisbelow(True)


def save(fig, output, name):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def draw_row(panels, rows, output, name, captions):
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        draw(ax, panel, [r for r in rows if r['panel'] == panel])
    fig.tight_layout(w_pad=1.6, rect=(0, .065, 1, 1))
    for ax, caption in zip(axes.flat, captions):
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, .025, caption, ha='center', va='bottom', fontsize=12)
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/reference')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    rows = load_runs(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    panels = [panel for panel in PANELS if any(r['panel'] == panel for r in rows)]
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 12, 'axes.labelsize': 13, 'axes.linewidth': .7,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.framealpha': .95, 'legend.fancybox': False,
                         'legend.edgecolor': '#bbbbbb', 'legend.borderpad': .3,
                         'legend.labelspacing': .25, 'legend.handlelength': 1.9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
    for panel in panels:
        fig, ax = plt.subplots(figsize=(4, 2.9))
        draw(ax, panel, [r for r in rows if r['panel'] == panel])
        fig.tight_layout(pad=.5)
        save(fig, args.output, panel)
    figure1 = [p for p in panels if p.startswith('fig1')]
    if len(figure1) == 3:
        draw_row(figure1, rows, args.output, 'fig1', ['(a)', '(b)', '(c)'])
    draw_row(panels, rows, args.output, 'overview', [CAPTIONS[p] for p in panels])
    with (args.output / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != 'source'])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != 'source'} for r in rows)
    print(args.output.resolve())


if __name__ == '__main__':
    main()

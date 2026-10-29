# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""Plot the final paper measurements from the canonical Figure 2 CSV."""

from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'results/Figure2_QAPScalability/figure_2_preprocessing_times.csv'
SIZES = [20, 40, 60, 80, 100]
METHODS = [('Gurobi', '#4DAF00', 'D'), ('PSLP', '#FF3B00', 's'),
           ('GPU-Preprocessor', '#0072BD', 'o')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'figure_2_preprocessing_time'
    with args.data.open(newline='') as handle:
        data = list(csv.DictReader(handle))
    if not data:
        raise ValueError('No scalability results to plot')
    for row in data:
        row['N'] = int(row['N'])
    sizes = sorted({row['N'] for row in data})
    if len({(row['N'], row['method']) for row in data}) != len(data):
        raise ValueError('Duplicate method/size measurements')
    if {row['method'] for row in data} - {m[0] for m in METHODS}:
        raise ValueError('Unknown preprocessing method')
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    serif = next(name for name in ('Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif') if name in available)
    plt.rcParams.update({'font.family': serif, 'font.size': 11, 'mathtext.fontset': 'stix',
                         'axes.labelsize': 14, 'axes.linewidth': 0.8, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.98)
    for method, color, marker in METHODS:
        records = sorted((row for row in data if row['method'] == method), key=lambda row: row['N'])
        if not records:
            continue
        if [row['N'] for row in records] != sizes:
            raise ValueError(f'{method} is missing a selected QAP size')
        values = [float(row['preprocessing_time_s']) for row in records]
        if any(not value > 0 for value in values):
            raise ValueError(f'{method} has an invalid preprocessing time')
        line, = ax.plot(sizes, values, label=method, color=color, linewidth=1.4,
                        marker=marker, markersize=4.7, markerfacecolor='white',
                        markeredgecolor=color, markeredgewidth=0.9)
        assert list(line.get_ydata()) == values
    ax.set_yscale('log', base=10)
    ax.set_ylim(5e-3, 1e3)
    ax.set_xlim(12, 102)
    ax.set_xticks(sizes)
    ax.set_xlabel(r'QAP dimension $N$', labelpad=11)
    ax.set_ylabel('Preprocessing time (s)', labelpad=13)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_axisbelow(True)
    ax.grid(which='major', color='#C8C8C8', linewidth=0.55)
    ax.grid(which='minor', axis='y', color='#DDDDDD', linewidth=0.55, linestyle=(0, (1, 3)))
    ax.tick_params(which='major', direction='out', length=4, width=0.75, pad=6)
    ax.tick_params(which='minor', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_color('#C8C8C8')
        ax.spines[side].set_linewidth(0.55)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color('#333333')
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.015, 0.985), frameon=True,
                       fancybox=False, framealpha=1.0, borderpad=0.75, labelspacing=0.55,
                       handlelength=2.4, fontsize=11.5)
    legend.get_frame().set_edgecolor('#888888')
    legend.get_frame().set_linewidth(0.7)
    fig.savefig(stem.with_suffix('.pdf'), dpi=300, bbox_inches='tight',
                pad_inches=0.04, facecolor='white')
    plt.close(fig)
    print('FONT', serif)
    path = stem.with_suffix('.pdf')
    print(path, path.stat().st_size)


if __name__ == '__main__':
    main()

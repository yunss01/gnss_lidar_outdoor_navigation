#!/usr/bin/env python3
"""Generate publication-ready (300 dpi) figures from the frozen ICCE runs.

The script intentionally reads the registered evaluation CSV instead of
rediscovering sessions, so the plots reproduce the exact 60-run comparison
reported in the paper.  It also writes a compact CSV/Markdown paper table.

Run with the system Python to avoid the user's NumPy 2 / matplotlib ABI mix:

  env PYTHONNOUSERSITE=1 /usr/bin/python3 tools/generate_paper_artifacts.py \
      --environment-image /tmp/codex-clipboard-kmV8aV.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


DPI = 300
OFFICIAL_VARIANTS = (
    'b0_fixed_ground',
    'b1_plane_ground',
    'proposed_recovery_v2',
)
METHODS = (
    ('b0_fixed_ground', 'B0: Fixed Z'),
    ('b1_plane_ground', 'B1: Global plane'),
    ('proposed_recovery_v2', 'Proposed: Local plane'),
)
SCENARIOS = ('F9', 'F10')
COLORS = {
    'b0_fixed_ground': '#C1633A',
    'b1_plane_ground': '#4A72A8',
    'proposed_recovery_v2': '#5C8A5C',
}
MARKERS = {'F9': 'o', 'F10': 's'}


def _float(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _bool(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _mean_std(rows, key):
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _load_runs(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    rows = [
        row for row in rows
        if row['variant'] in OFFICIAL_VARIANTS and row['scenario'] in SCENARIOS
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['variant'], row['scenario'])].append(row)
    expected = {(variant, scenario) for variant in OFFICIAL_VARIANTS
                for scenario in SCENARIOS}
    missing = expected - set(groups)
    invalid = {
        key: len(value) for key, value in groups.items() if len(value) != 10
    }
    if missing or invalid or len(rows) != 60:
        raise ValueError(
            'Expected the frozen 3 x 2 x 10 experiment. '
            'missing={} invalid_counts={} total={}'.format(
                sorted(missing), invalid, len(rows))
        )
    return rows, groups


def _setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': DPI,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _save(fig, directory, stem):
    fig.savefig(directory / (stem + '.png'), dpi=DPI, bbox_inches='tight',
                facecolor='white')
    fig.savefig(directory / (stem + '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def _successful(groups, variant, scenario):
    return [row for row in groups[(variant, scenario)] if _bool(row['success'])]


def _annotate_rate(ax, bars, rows):
    for bar, entries in zip(bars, rows):
        success = sum(_bool(entry['success']) for entry in entries)
        total = len(entries)
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2.8,
                '{}/{}'.format(success, total), ha='center', va='bottom',
                fontsize=8, fontweight='bold')


def plot_success_rate(groups, directory):
    fig, ax = plt.subplots(figsize=(7.1, 3.7), constrained_layout=True)
    positions = np.arange(len(SCENARIOS))
    width = 0.23
    for index, (variant, label) in enumerate(METHODS):
        data = [groups[(variant, scenario)] for scenario in SCENARIOS]
        rates = [100.0 * sum(_bool(row['success']) for row in rows) / len(rows)
                 for rows in data]
        bars = ax.bar(positions + (index - 1) * width, rates, width,
                      label=label, color=COLORS[variant], edgecolor='#303030',
                      linewidth=0.55, alpha=0.72)
        _annotate_rate(ax, bars, data)
    ax.set_xticks(positions, SCENARIOS)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Mission success rate (%)')
    ax.set_title('Mission completion over 10 independent trials per condition')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.20),
              frameon=False)
    _save(fig, directory, 'fig_success_rate')


def _boxplot_panel(ax, groups, scenario, metric, ylabel, title):
    values = []
    labels = []
    colors = []
    for variant, label in METHODS:
        completed = _successful(groups, variant, scenario)
        current = [_float(row[metric]) for row in completed]
        current = [value for value in current if math.isfinite(value)]
        values.append(current)
        labels.append('{}\n(n={})'.format(label.split(':')[0], len(current)))
        colors.append(COLORS[variant])
    boxes = ax.boxplot(values, patch_artist=True, widths=0.58, showfliers=False,
                       medianprops={'color': '#111111', 'linewidth': 1.3},
                       whiskerprops={'linewidth': 0.9},
                       capprops={'linewidth': 0.9})
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor('#303030')
    jitter = random.Random(20260831 + (0 if scenario == 'F9' else 1))
    for x, current in enumerate(values, start=1):
        offsets = [jitter.uniform(-0.10, 0.10) for _ in current]
        ax.scatter([x + offset for offset in offsets], current, s=14,
                   color='#171717', alpha=0.7, linewidths=0, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)


def plot_stop_distribution(groups, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.21,
                        wspace=0.08)
    for axis, scenario in zip(axes, SCENARIOS):
        _boxplot_panel(axis, groups, scenario, 'total_stop_s',
                       'Total stopped time (s)' if scenario == 'F9' else '',
                       scenario)
    _save(fig, directory, 'fig_total_stop_time')


def _bar_metric(ax, groups, metric, ylabel, title, formatter=None):
    positions = np.arange(len(SCENARIOS))
    width = 0.23
    for index, (variant, label) in enumerate(METHODS):
        completed = [_successful(groups, variant, scenario)
                     for scenario in SCENARIOS]
        means_stds = [_mean_std(rows, metric) for rows in completed]
        means = [item[0] for item in means_stds]
        stds = [item[1] for item in means_stds]
        ax.bar(positions + (index - 1) * width, means, width,
               yerr=stds, capsize=3, label=label, color=COLORS[variant],
               edgecolor='#303030', linewidth=0.55, alpha=0.72,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8})
    ax.set_xticks(positions, SCENARIOS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    if formatter:
        ax.yaxis.set_major_formatter(formatter)


def plot_efficiency(groups, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55),
                             constrained_layout=True)
    _bar_metric(axes[0], groups, 'duration_s', 'Completion time (s)',
                'Completion time')
    _bar_metric(axes[1], groups, 'path_length_ratio',
                'Driven / mission distance', 'Path-length ratio')
    axes[1].axhline(1.0, color='#444444', linestyle='--', linewidth=0.9,
                    label='Mission polyline')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    _save(fig, directory, 'fig_completion_efficiency')


def _read_frames(session_path):
    frames = Path(session_path) / 'frames.csv'
    rows = []
    with frames.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            x = _float(row.get('vehicle_x'))
            y = _float(row.get('vehicle_y'))
            if math.isfinite(x) and math.isfinite(y):
                rows.append(row)
    if len(rows) < 2:
        raise ValueError('not enough trajectory samples: {}'.format(frames))
    return rows


def _representative_row(groups, scenario):
    completed = _successful(groups, 'proposed_recovery_v2', scenario)
    completed = sorted(completed, key=lambda row: _float(row['path_length_ratio']))
    return completed[len(completed) // 2]


def _goal_transition_points(rows):
    """Approximate each reached goal with the final vehicle pose in its stage."""
    stages = defaultdict(list)
    for row in rows:
        index = int(_float(row.get('route_index'), 0))
        if index > 0:
            stages[index].append(row)
    points = []
    for index in sorted(stages):
        row = stages[index][-1]
        points.append((index, _float(row['vehicle_x']), _float(row['vehicle_y'])))
    return points


def _draw_direction_arrows(ax, rows, scenario):
    """Put one small arrowhead on each waypoint-to-waypoint trajectory stage."""
    stages = defaultdict(list)
    for row in rows:
        index = int(_float(row.get('route_index'), 0))
        if index > 0:
            stages[index].append(row)
    first_stage = min(stages) if stages else None
    # Use stage-specific fractions only where the representative trace turns
    # sharply.  This keeps every arrow visibly centred on an unambiguous piece
    # of the sampled trajectory (rather than on a near-stationary point).
    fractions = {
        'F9': {4: 0.62},
        'F10': {3: 0.42},
    }
    for stage_index, stage_rows in stages.items():
        # The first arrow would sit next to the Start star, so leave it out.
        if stage_index == first_stage:
            continue
        if len(stage_rows) < 8:
            continue
        fraction = fractions.get(scenario, {}).get(stage_index, 0.54)
        centre = int(fraction * len(stage_rows))
        half_window = min(4, max(1, (len(stage_rows) - 1) // 8))
        centre = max(half_window, min(len(stage_rows) - half_window - 1,
                                      centre))
        # Sample symmetrically around the selected on-trajectory point.  The
        # wider local segment avoids vanishing arrows on slowly moving stages.
        first = stage_rows[centre - half_window]
        last = stage_rows[centre + half_window]
        start = (_float(first['vehicle_x']), _float(first['vehicle_y']))
        end = (_float(last['vehicle_x']), _float(last['vehicle_y']))
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 0.05:
            continue
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=15,
            linewidth=1.6, edgecolor='#0072B2', facecolor='#0072B2',
            connectionstyle='arc3,rad=0', zorder=4,
        ))


def _plot_trajectory_panel(ax, row, scenario):
    frames = _read_frames(row['session_path'])
    x = np.asarray([_float(item['vehicle_x']) for item in frames])
    y = np.asarray([_float(item['vehicle_y']) for item in frames])
    ax.plot(x, y, color='#0072B2', linewidth=1.55, label='Vehicle trajectory')
    _draw_direction_arrows(ax, frames, scenario)
    ax.scatter(x[0], y[0], marker='*', s=120, color='#111111', zorder=4,
               label='Start')
    ax.scatter(x[-1], y[-1], marker='X', s=60, color='#D55E00', zorder=4,
               label='End')
    goal_points = _goal_transition_points(frames)
    labelled_goals = []
    if (scenario == 'F10' and len(goal_points) >= 5 and
            math.hypot(goal_points[0][1] - goal_points[-1][1],
                       goal_points[0][2] - goal_points[-1][2]) < 2.0):
        # The closed route returns to WP1.  One combined marker prevents the
        # two labels from hiding each other.
        first, last = goal_points[0], goal_points[-1]
        labelled_goals.append(('WP1 = WP5', (first[1] + last[1]) / 2.0,
                               (first[2] + last[2]) / 2.0))
        labelled_goals.extend(
            ('WP{}'.format(index), gx, gy)
            for index, gx, gy in goal_points[1:-1]
        )
    else:
        labelled_goals = [
            ('WP{}'.format(index), gx, gy)
            for index, gx, gy in goal_points
        ]
    for label, gx, gy in labelled_goals:
        ax.scatter(gx, gy, marker='o', s=38, facecolor='white',
                   edgecolor='#222222', linewidth=1.0, zorder=5)
        if label in {'WP1', 'WP1 = WP5'}:
            offset, alignment = (0, 6), 'center'
        elif label == 'WP4':
            offset, alignment = (9, -6), 'left'
        else:
            offset, alignment = (4, 4), 'left'
        ax.annotate(label, (gx, gy), xytext=offset,
                    textcoords='offset points', fontsize=7, color='#222222',
                    ha=alignment)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('CARLA world x (m)')
    ax.set_ylabel('CARLA world y (m)')
    ax.set_title('{} trajectory'.format(scenario))
    ax.grid(alpha=0.25, linewidth=0.6)
    return row['session']


def plot_trajectories(groups, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.18,
                        wspace=0.24)
    chosen = []
    for axis, scenario in zip(axes, SCENARIOS):
        chosen.append(_plot_trajectory_panel(
            axis, _representative_row(groups, scenario), scenario))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, 0.055), frameon=False)
    _save(fig, directory, 'fig_representative_trajectories')
    return chosen


def _box(ax, xy, width, height, label, color, fontsize=8.3):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.012,rounding_size=0.02',
        linewidth=1.0, facecolor=color, edgecolor='#2E2E2E',
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, label,
            ha='center', va='center', fontsize=fontsize, color='#111111')
    return patch


def _arrow(ax, start, end, *, style='-|>', color='#303030', lw=1.2,
           connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, connectionstyle=connectionstyle,
    ))


def plot_architecture(directory):
    fig, ax = plt.subplots(figsize=(7.15, 3.7), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Online LiDAR-guided navigation architecture', pad=7)

    _box(ax, (0.02, 0.63), 0.16, 0.18, 'GNSS waypoints\nF9 / F10 mission', '#D9EAF7')
    _box(ax, (0.02, 0.25), 0.16, 0.18, '3-D LiDAR\n+ odometry', '#D9EAF7')
    _box(ax, (0.25, 0.25), 0.18, 0.18,
         'Local-plane\nground removal', '#CDEBDD')
    _box(ax, (0.49, 0.58), 0.21, 0.20,
         'Rolling Nav2 costmaps\n+ Smac Hybrid-A*', '#FDE5C1')
    _box(ax, (0.49, 0.24), 0.21, 0.20,
         'Local avoidance\n+ path-validity gate', '#FDE5C1')
    _box(ax, (0.77, 0.41), 0.19, 0.20,
         'LiDAR safety gate\n(final command authority)', '#F6D0D0')
    _box(ax, (0.77, 0.08), 0.19, 0.16,
         'CARLA Ackermann\nvehicle', '#E8E8E8')

    _arrow(ax, (0.18, 0.72), (0.49, 0.68))
    _arrow(ax, (0.18, 0.34), (0.25, 0.34))
    _arrow(ax, (0.43, 0.34), (0.49, 0.34))
    _arrow(ax, (0.43, 0.38), (0.49, 0.62), connectionstyle='arc3,rad=0.18')
    _arrow(ax, (0.595, 0.58), (0.595, 0.44))
    _arrow(ax, (0.70, 0.34), (0.77, 0.47))
    # The safety gate applies the same ground-aware obstacle
    # classification as the Nav2 costmaps.  Route this branch below the
    # control blocks so it cannot be mistaken for a command-only input.
    branch_color = '#2F6B55'
    ax.plot(
        [0.34, 0.34, 0.735, 0.735],
        [0.25, 0.14, 0.14, 0.51],
        color=branch_color, linewidth=1.25, solid_capstyle='round',
        zorder=2,
    )
    _arrow(ax, (0.735, 0.51), (0.77, 0.51),
           color=branch_color, lw=1.25)
    ax.text(0.54, 0.115, 'shared ground-aware obstacles', fontsize=7,
            color=branch_color, ha='center', va='top')
    _arrow(ax, (0.865, 0.41), (0.865, 0.24))
    ax.text(0.705, 0.76, 'continuous map / plan update', fontsize=7,
            color='#555555', ha='center')
    _save(fig, directory, 'fig_system_architecture')


def plot_environment(image_path, directory):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    fig_width = width / DPI
    fig_height = height / DPI
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    ax.imshow(image)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _save(fig, directory, 'fig_experimental_environment_town10hd')


def _fmt_mean_std(rows, key, digits=1):
    mean, std = _mean_std(rows, key)
    if not math.isfinite(mean):
        return '—'
    return ('{:.%df} ± {:.%df}' % (digits, digits)).format(mean, std)


def write_result_table(groups, directory):
    fields = [
        'Method', 'Scenario', 'Success', 'Collision', 'Time (s)',
        'Path ratio', 'Total stop (s)', 'Maximum stop (s)',
    ]
    rows = []
    for variant, method in METHODS:
        for scenario in SCENARIOS:
            all_runs = groups[(variant, scenario)]
            completed = _successful(groups, variant, scenario)
            success = sum(_bool(row['success']) for row in all_runs)
            collision = sum(_bool(row['collision']) for row in all_runs)
            rows.append({
                'Method': method,
                'Scenario': scenario,
                'Success': '{}/{} ({:.0f}%)'.format(
                    success, len(all_runs), 100 * success / len(all_runs)),
                'Collision': '{}/{} ({:.0f}%)'.format(
                    collision, len(all_runs), 100 * collision / len(all_runs)),
                'Time (s)': _fmt_mean_std(completed, 'duration_s'),
                'Path ratio': _fmt_mean_std(completed, 'path_length_ratio', 2),
                'Total stop (s)': _fmt_mean_std(completed, 'total_stop_s'),
                'Maximum stop (s)': _fmt_mean_std(completed, 'max_stop_s'),
            })
    with (directory / 'table_quantitative_results.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = ['| ' + ' | '.join(fields) + ' |',
             '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in rows:
        table.append('| ' + ' | '.join(row[field] for field in fields) + ' |')
    table.extend([
        '',
        'All values after Success/Collision are mean ± sample SD over '
        'completed trials only. Success and collision use all 10 attempts.',
    ])
    (directory / 'table_quantitative_results.md').write_text(
        '\n'.join(table) + '\n', encoding='utf-8')


def write_readme(directory, trajectory_sessions, source_runs, image_path):
    content = """# ICCE-ASIA 2026 paper artifacts

All PNG figures were generated at **300 dpi**.  Every figure is also emitted
as a vector PDF, which is the preferred format for the final manuscript.

## Inputs

- Frozen experiment registry: `{runs}`
- Official experiment matrix: B0 / B1 / Proposed × F9 / F10 × 10 = 60 runs
- Environment image: `{image}`

## Contents

- `fig_system_architecture.*`: proposed online navigation pipeline.
- `fig_experimental_environment_town10hd.*`: CARLA Town10HD setting image.
- `fig_success_rate.*`: success count/rate across all trials.
- `fig_total_stop_time.*`: stop-time distributions, completed trials only.
- `fig_completion_efficiency.*`: completion time and path-ratio comparison.
- `fig_representative_trajectories.*`: median path-ratio proposed trajectories.
- `table_quantitative_results.*`: paper-table source.

The representative trajectories use these sessions, selected deterministically
as the median path-length-ratio completed Proposed trial in each scenario:

- F9: `{f9}`
- F10: `{f10}`
""".format(
        runs=source_runs.resolve(), image=image_path or 'not generated',
        f9=trajectory_sessions[0], f10=trajectory_sessions[1],
    )
    (directory / 'README.md').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--runs', type=Path,
        default=Path('/home/sukja/terrain_nav_data/evaluation/icce_asia_2026/runs.csv'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('/home/sukja/terrain_nav_data/evaluation/icce_asia_2026/paper_artifacts'),
    )
    parser.add_argument('--environment-image', type=Path)
    args = parser.parse_args()

    _setup_style()
    rows, groups = _load_runs(args.runs.expanduser())
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    plot_architecture(output)
    plot_success_rate(groups, output)
    plot_stop_distribution(groups, output)
    plot_efficiency(groups, output)
    trajectory_sessions = plot_trajectories(groups, output)
    write_result_table(groups, output)
    image = None
    if args.environment_image:
        image = args.environment_image.expanduser()
        if not image.is_file():
            raise FileNotFoundError(image)
        plot_environment(image, output)
    write_readme(output, trajectory_sessions, args.runs.expanduser(), image)
    print('Generated {} files in {}'.format(
        len(list(output.iterdir())), output))


if __name__ == '__main__':
    main()

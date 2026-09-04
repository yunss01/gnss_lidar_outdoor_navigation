#!/usr/bin/env python3
"""Render a same-scan qualitative B0/B1/local-plane obstacle-cloud ablation.

The official ICCE evaluation sessions intentionally store only CSV/JSON
telemetry.  This tool therefore replays one *archived raw LiDAR scan* through
the final three ground-processing variants.  It is a qualitative methodology
figure, not an additional navigation-performance trial.

Run from the workspace:
  env PYTHONNOUSERSITE=1 /usr/bin/python3 \
    tools/generate_ground_ablation_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'src' / 'terrain_navigation_pkg'))
from terrain_navigation_pkg.ground_obstacle_core import (  # noqa: E402
    extract_ground_relative_obstacles,
)


DPI = 300
DEFAULT_SAMPLE = Path(
    '/home/sukja/terrain_nav_data/learning/raw/'
    'session_20260828_165630_901785/samples/sample_000106.npz'
)
DEFAULT_OUTPUT = Path(
    '/home/sukja/terrain_nav_data/evaluation/icce_asia_2026/paper_artifacts'
)

METHODS = (
    ('B0: Fixed Z', '#C1633A', {
        'enabled': False,
        'local_ground_enabled': False,
        'local_ground_plane_enabled': False,
    }),
    ('B1: Global plane', '#4A72A8', {
        'enabled': True,
        'local_ground_enabled': False,
        'local_ground_plane_enabled': False,
    }),
    ('Proposed: Local plane', '#5C8A5C', {
        'enabled': True,
        'local_ground_enabled': True,
        'local_ground_plane_enabled': True,
    }),
)

FILTER_PARAMETERS = {
    'fixed_minimum_z_m': -1.4,
    'fixed_maximum_z_m': 1.0,
    'ground_fit_minimum_range_m': 2.0,
    'ground_fit_maximum_range_m': 20.0,
    'ground_fit_minimum_z_m': -2.5,
    'ground_fit_maximum_z_m': -0.8,
    'ground_seed_quantile': 0.35,
    'ground_inlier_tolerance_m': 0.06,
    'minimum_ground_points': 80,
    'low_obstacle_minimum_height_m': 0.07,
    'low_obstacle_maximum_height_m': 0.45,
    'local_ground_resolution_m': 0.75,
    'local_ground_radius_m': 1.5,
    'local_ground_quantile': 0.25,
    'local_ground_plane_residual_tolerance_m': 0.055,
}


def nav2_obstacle_mask(points, options):
    """Match the point selection used by lidar_obstacle_filter_node."""
    result = extract_ground_relative_obstacles(
        points, **FILTER_PARAMETERS, **options
    )
    x, y, _ = points.T
    range_sq = x * x + y * y
    mask = (
        result.obstacle_mask
        & (range_sq >= 1.5 ** 2)
        & (range_sq <= 45.0 ** 2)
    )
    ego_body = (
        (x >= -2.65) & (x <= 2.55) & (np.abs(y) <= 1.15)
    )
    mask &= ~ego_body
    indices = np.flatnonzero(mask)
    if indices.size:
        keys = np.floor(points[indices] / 0.20).astype(np.int32)
        _, unique = np.unique(keys, axis=0, return_index=True)
        chosen = indices[np.sort(unique)]
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[chosen] = True
    return mask


def _draw_panel(axis, points, mask, title, color):
    # Vehicle coordinates are preserved: x points forward, y points left.
    display = (
        (points[:, 0] >= -4.0) & (points[:, 0] <= 15.0)
        & (points[:, 1] >= -12.0) & (points[:, 1] <= 12.0)
    )
    axis.scatter(
        points[display, 0], points[display, 1], s=0.45,
        color='#B7B7B7', alpha=0.34, linewidths=0, rasterized=True,
        label='All LiDAR returns',
    )
    selected = display & mask
    axis.scatter(
        points[selected, 0], points[selected, 1], s=1.6,
        color=color, alpha=0.82, linewidths=0, rasterized=True,
        label='Nav2 obstacle-cloud return',
    )
    axis.add_patch(Rectangle(
        (-2.65, -1.15), 5.20, 2.30, fill=False,
        edgecolor='#202020', linewidth=0.8, zorder=4,
    ))
    axis.scatter(0.0, 0.0, marker='^', s=24, color='#202020', zorder=5)
    axis.annotate('', xy=(4.0, 0.0), xytext=(0.0, 0.0),
                  arrowprops={'arrowstyle': '->', 'color': '#202020',
                              'lw': 0.8})
    axis.text(4.15, 0.25, 'forward', fontsize=6.6, color='#303030')
    axis.set_title('{}\n{} obstacle points'.format(title, int(selected.sum())),
                   fontsize=8.2, pad=4)
    axis.set_xlim(-4.0, 15.0)
    axis.set_ylim(-12.0, 12.0)
    axis.set_aspect('equal', adjustable='box')
    axis.tick_params(labelsize=6.8)
    axis.grid(alpha=0.18, linewidth=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--output-directory', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.sample.is_file():
        raise FileNotFoundError(args.sample)
    arrays = np.load(args.sample)
    points = np.asarray(arrays['lidar_points_xyz'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not points.size:
        raise ValueError('sample contains no raw LiDAR points: {}'.format(args.sample))

    fig, axes = plt.subplots(1, 3, figsize=(6.35, 3.10), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.77, bottom=0.27,
                        wspace=0.08)
    for axis, (name, color, options) in zip(axes, METHODS):
        _draw_panel(axis, points, nav2_obstacle_mask(points, options),
                    name, color)
    fig.supxlabel('Vehicle-forward x (m)', fontsize=8.2, y=0.165)
    fig.supylabel('Vehicle-left y (m)', fontsize=8.2, x=0.012)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 0.035), frameon=False, fontsize=7.1)
    fig.suptitle('Ground-Model Comparison for LiDAR Obstacle-Cloud Extraction',
                 fontsize=10, y=0.975)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / 'fig_ground_model_qualitative_ablation'
    fig.savefig(stem.with_suffix('.png'), dpi=DPI, bbox_inches='tight',
                pad_inches=0.02)
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    source = stem.with_name(stem.name + '_source.txt')
    source.write_text(
        'Qualitative same-scan replay only; not part of the 60-run '\
        'performance evaluation.\nsource_sample={}\n'.format(args.sample),
        encoding='utf-8',
    )
    print('Wrote {}.png and {}.pdf'.format(stem, stem))


if __name__ == '__main__':
    main()

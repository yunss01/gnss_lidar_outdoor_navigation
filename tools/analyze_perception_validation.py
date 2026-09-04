#!/usr/bin/env python3
"""Quantify false occupied XY cells in manually verified free space.

Each stationary capture is replayed through the same B0, B1, and proposed
ground-obstacle filters used by Nav2.  A free-space polygon supplied by
``annotate_perception_free_space.py`` is the scene-level reference: any Nav2
occupied XY cell inside it is a false occupied cell.  One scan is used from
each stationary capture; repeated frames in the 15-s capture are deliberately
not treated as independent experimental observations.

Run from the workspace:

  python3 tools/analyze_perception_validation.py

Outputs are written to the paper-artifacts directory.  They are perception
validation artifacts only and do not modify the frozen 60-run navigation
evaluation CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Avoid mixing user-site NumPy 2.x with the system Matplotlib compiled against
# NumPy 1.x.  This mirrors the annotation utility's compatibility bootstrap.
if os.environ.get('PYTHONNOUSERSITE') != '1':
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path as PolygonPath
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'src' / 'terrain_navigation_pkg'))
from terrain_navigation_pkg.ground_obstacle_core import (  # noqa: E402
    extract_ground_relative_obstacles,
)


DPI = 300
VOXEL_RESOLUTION_M = 0.20
DEFAULT_RAW_ROOT = Path('/home/sukja/terrain_nav_data/perception_validation/raw')
DEFAULT_OUTPUT = Path(
    '/home/sukja/terrain_nav_data/evaluation/icce_asia_2026/paper_artifacts'
)

METHODS = (
    ('B0: Fixed Z', 'b0_fixed_z', '#C1633A', {
        'enabled': False,
        'local_ground_enabled': False,
        'local_ground_plane_enabled': False,
    }),
    ('B1: Global plane', 'b1_global_plane', '#4A72A8', {
        'enabled': True,
        'local_ground_enabled': False,
        'local_ground_plane_enabled': False,
    }),
    ('Proposed: Local plane', 'proposed_local_plane', '#5C8A5C', {
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


def nav2_obstacle_mask(points: np.ndarray, options: dict) -> np.ndarray:
    """Match the range, ego exclusion, and XYZ voxelization of Nav2 input."""
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
    ego_body = (x >= -2.65) & (x <= 2.55) & (np.abs(y) <= 1.15)
    mask &= ~ego_body

    indices = np.flatnonzero(mask)
    if indices.size:
        keys = np.floor(points[indices] / VOXEL_RESOLUTION_M).astype(np.int32)
        _, unique = np.unique(keys, axis=0, return_index=True)
        chosen = indices[np.sort(unique)]
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[chosen] = True
    return mask


def _xy_cell_keys(points: np.ndarray, mask: np.ndarray) -> set[tuple[int, int]]:
    return set(map(
        tuple,
        np.floor(points[mask, :2] / VOXEL_RESOLUTION_M).astype(np.int32),
    ))


def _polygon_grid_cell_count(vertices: np.ndarray) -> int:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    x_values = np.arange(
        np.floor(lower[0] / VOXEL_RESOLUTION_M) * VOXEL_RESOLUTION_M,
        np.ceil(upper[0] / VOXEL_RESOLUTION_M) * VOXEL_RESOLUTION_M,
        VOXEL_RESOLUTION_M,
    ) + 0.5 * VOXEL_RESOLUTION_M
    y_values = np.arange(
        np.floor(lower[1] / VOXEL_RESOLUTION_M) * VOXEL_RESOLUTION_M,
        np.ceil(upper[1] / VOXEL_RESOLUTION_M) * VOXEL_RESOLUTION_M,
        VOXEL_RESOLUTION_M,
    ) + 0.5 * VOXEL_RESOLUTION_M
    grid = np.asarray([(x, y) for x in x_values for y in y_values])
    return int(PolygonPath(vertices).contains_points(grid, radius=1e-9).sum())


def _polygon_area(vertices: np.ndarray) -> float:
    return float(0.5 * abs(
        np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
        - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
    ))


def _display_label(session: str, capture_label: str) -> str:
    # Human-readable names for the current ICCE perception-validation set.
    fixed = {
        'session_20260831_213148_859096': 'Flat road',
        'session_20260831_213246_343411': 'WP2–WP3 crown',
        'session_20260831_213341_852695': 'Curb-adjacent',
        'session_20260831_215818_819308': 'WP2–WP3 side region',
        'session_20260831_220955_745662': 'WP2–WP3 free space',
    }
    return fixed.get(session, capture_label.replace('_', ' ') or session)


def analyze_session(raw_root: Path, session: str, annotation: dict) -> list[dict]:
    sample_path = raw_root / session / 'samples' / 'sample_000001.npz'
    if not sample_path.is_file():
        raise FileNotFoundError('missing source scan: {}'.format(sample_path))
    points = np.asarray(
        np.load(sample_path)['lidar_points_xyz'], dtype=np.float64
    )
    vertices = np.asarray(annotation['polygon_vehicle_xy'], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError('invalid free-space polygon for {}'.format(session))

    polygon = PolygonPath(vertices)
    inside_free_space = polygon.contains_points(points[:, :2], radius=1e-9)
    free_cells = _polygon_grid_cell_count(vertices)
    if free_cells <= 0:
        raise ValueError('free-space polygon contains no 20-cm cells: {}'.format(session))

    metadata_path = raw_root / session / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    label = str(metadata.get('perception_capture_label', ''))
    common = {
        'session': session,
        'scene': _display_label(session, label),
        'capture_label': label,
        'source_sample': str(sample_path),
        'free_space_area_m2': _polygon_area(vertices),
        'free_space_grid_cells': free_cells,
        'raw_lidar_returns_inside_free_space': int(inside_free_space.sum()),
    }

    rows = []
    free_point_cells = _xy_cell_keys(points, inside_free_space)
    for method_name, method_key, _, options in METHODS:
        occupied_cells = len(
            _xy_cell_keys(points, nav2_obstacle_mask(points, options))
            & free_point_cells
        )
        row = dict(common)
        row.update({
            'method': method_name,
            'method_key': method_key,
            'false_occupied_xy_cells': occupied_cells,
            'false_occupied_cell_rate_percent': (
                100.0 * occupied_cells / free_cells
            ),
            'false_occupied_cells_per_100m2': (
                100.0 * occupied_cells / _polygon_area(vertices)
            ),
        })
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], output: Path) -> None:
    fields = [
        'session', 'scene', 'capture_label', 'method', 'method_key',
        'free_space_area_m2', 'free_space_grid_cells',
        'raw_lidar_returns_inside_free_space', 'false_occupied_xy_cells',
        'false_occupied_cell_rate_percent', 'false_occupied_cells_per_100m2',
        'source_sample',
    ]
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict], output: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row['session'], []).append(row)
    lines = [
        '| Verified free-space scene | Area (m²) | B0 cells (%) | '
        'B1 cells (%) | Proposed cells (%) |',
        '|---|---:|---:|---:|---:|',
    ]
    pooled = {method_key: 0 for _, method_key, _, _ in METHODS}
    total_cells = 0
    for session, group in grouped.items():
        group = sorted(group, key=lambda item: item['method_key'])
        by_method = {item['method_key']: item for item in group}
        area = group[0]['free_space_area_m2']
        total_cells += group[0]['free_space_grid_cells']
        for key in pooled:
            pooled[key] += by_method[key]['false_occupied_xy_cells']
        value = lambda key: '{} ({:.2f}%)'.format(
            by_method[key]['false_occupied_xy_cells'],
            by_method[key]['false_occupied_cell_rate_percent'],
        )
        lines.append('| {} | {:.2f} | {} | {} | {} |'.format(
            group[0]['scene'], area,
            value('b0_fixed_z'), value('b1_global_plane'),
            value('proposed_local_plane'),
        ))
    lines.append('| **Pooled** | — | {} ({:.2f}%) | {} ({:.2f}%) | {} ({:.2f}%) |'.format(
        pooled['b0_fixed_z'], 100.0 * pooled['b0_fixed_z'] / total_cells,
        pooled['b1_global_plane'], 100.0 * pooled['b1_global_plane'] / total_cells,
        pooled['proposed_local_plane'],
        100.0 * pooled['proposed_local_plane'] / total_cells,
    ))
    lines.extend([
        '',
        'Definition: a false occupied cell is a Nav2 20-cm XY occupied cell '
        'whose source LiDAR return lies inside a manually verified, clear, '
        'traversable polygon. One scan is used per stationary capture.',
        '',
        'Interpretation: this is a false-positive-only perception metric. '
        'A low B0 value does not establish superior obstacle perception, '
        'because false negatives are evaluated separately by the end-to-end '
        'route-completion and collision results.',
    ])
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _plot(rows: list[dict], output_stem: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row['session'], []).append(row)
    sessions = list(grouped)
    labels = [grouped[session][0]['scene'].replace(' ', '\n', 1)
              for session in sessions]
    x_values = np.arange(len(sessions), dtype=np.float64)
    bar_width = 0.23

    figure, axis = plt.subplots(figsize=(7.05, 3.55))
    for index, (method_name, method_key, color, _) in enumerate(METHODS):
        values = []
        for session in sessions:
            row = next(item for item in grouped[session]
                       if item['method_key'] == method_key)
            values.append(row['false_occupied_cells_per_100m2'])
        offset = (index - 1) * bar_width
        bars = axis.bar(
            x_values + offset, values, width=bar_width, label=method_name,
            color=color, alpha=0.72, edgecolor='none', zorder=3,
        )
        for bar, value in zip(bars, values):
            if value >= 3.0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0, value + 0.65,
                    '{:.1f}'.format(value), ha='center', va='bottom',
                    fontsize=7.5, color='#303030',
                )

    axis.set_xticks(x_values, labels, fontsize=8)
    axis.set_ylabel('False occupied cells per 100 m²', fontsize=9)
    axis.set_title('False Occupancy in Manually Verified Free Space',
                   fontsize=11, pad=9)
    axis.set_axisbelow(True)
    axis.grid(axis='y', alpha=0.24, linewidth=0.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.105, right=0.99, top=0.86, bottom=0.27)
    figure.savefig(output_stem.with_suffix('.png'), dpi=DPI,
                   bbox_inches='tight', pad_inches=0.025)
    figure.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight',
                   pad_inches=0.025)
    plt.close(figure)


def _plot_case_study(
    rows: list[dict], raw_root: Path, annotations: dict, output_stem: Path,
) -> None:
    """Show the strongest verified-free-space case, not a sparse bar chart.

    B0 is intentionally omitted here: its low false-positive count is caused
    in part by conservative fixed-Z rejection, and the main end-to-end table
    already reports its completion/collision trade-off.  This visual answers
    the focused perception question: how does the adaptive local model differ
    from the directly comparable global-plane model in the same free space?
    """
    by_session: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_session.setdefault(row['session'], {})[row['method_key']] = row
    selected_session = max(
        by_session,
        key=lambda session: (
            by_session[session]['b1_global_plane']['false_occupied_xy_cells']
            - by_session[session]['proposed_local_plane']['false_occupied_xy_cells']
        ),
    )
    selected = by_session[selected_session]
    sample_path = Path(selected['b1_global_plane']['source_sample'])
    points = np.asarray(np.load(sample_path)['lidar_points_xyz'], dtype=np.float64)
    vertices = np.asarray(
        annotations[selected_session]['polygon_vehicle_xy'], dtype=np.float64
    )
    polygon = PolygonPath(vertices)
    in_free_space = polygon.contains_points(points[:, :2], radius=1e-9)
    display = (
        (points[:, 0] >= -4.0) & (points[:, 0] <= 15.0)
        & (points[:, 1] >= -12.0) & (points[:, 1] <= 12.0)
    )

    comparison = (
        ('B1: Global plane', 'b1_global_plane', '#4A72A8', METHODS[1][3]),
        ('Proposed: Local plane', 'proposed_local_plane', '#5C8A5C', METHODS[2][3]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.35, 3.40), sharey=True)
    figure.subplots_adjust(left=0.105, right=0.995, top=0.77, bottom=0.28,
                            wspace=0.08)
    legend_handles = None
    for axis, (name, method_key, color, options) in zip(axes, comparison):
        obstacle = nav2_obstacle_mask(points, options)
        false_return = display & obstacle & in_free_space
        axis.scatter(
            points[display, 0], points[display, 1], s=0.48,
            color='#B7B7B7', alpha=0.30, linewidths=0, rasterized=True,
            label='All LiDAR returns',
        )
        axis.add_patch(Polygon(
            vertices, closed=True, facecolor='#F2C14E', edgecolor='#9B6A00',
            linewidth=0.9, alpha=0.20, zorder=2,
            label='Verified free space',
        ))
        axis.scatter(
            points[display & obstacle, 0], points[display & obstacle, 1],
            s=1.55, color=color, alpha=0.64, linewidths=0, rasterized=True,
            label='Nav2 obstacle-cloud return', zorder=3,
        )
        # Saturated points make the false-occupancy evidence readable even
        # when the transparent polygon lies under the obstacle cloud.
        axis.scatter(
            points[false_return, 0], points[false_return, 1],
            s=3.4, color=color, alpha=0.96, linewidths=0, rasterized=True,
            zorder=4,
        )
        axis.add_patch(Rectangle(
            (-2.65, -1.15), 5.20, 2.30, fill=False, edgecolor='#202020',
            linewidth=0.8, zorder=5,
        ))
        axis.scatter(0.0, 0.0, marker='^', s=24, color='#202020', zorder=6)
        axis.annotate('', xy=(4.0, 0.0), xytext=(0.0, 0.0),
                      arrowprops={'arrowstyle': '->', 'color': '#202020',
                                  'lw': 0.8})
        axis.set_title('{}\n{} false occupied cells'.format(
            name, selected[method_key]['false_occupied_xy_cells']),
            fontsize=8.8, pad=4,
        )
        axis.set_xlim(-4.0, 15.0)
        axis.set_ylim(-12.0, 12.0)
        axis.set_aspect('equal', adjustable='box')
        axis.tick_params(labelsize=6.8)
        axis.grid(alpha=0.18, linewidth=0.5)
        legend_handles = axis.get_legend_handles_labels()
    figure.supxlabel('Vehicle-forward x (m)', fontsize=8.2, y=0.17)
    figure.supylabel('Vehicle-left y (m)', fontsize=8.2, x=0.014)
    figure.suptitle('Verified Free-Space Occupancy at WP2–WP3',
                     fontsize=10.2, y=0.975)
    figure.legend(*legend_handles, ncol=3, loc='lower center',
                  bbox_to_anchor=(0.5, 0.035), frameon=False, fontsize=7.0)
    figure.savefig(output_stem.with_suffix('.png'), dpi=DPI,
                   bbox_inches='tight', pad_inches=0.025)
    figure.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight',
                   pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output-directory', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    annotation_path = (
        args.annotations.expanduser().resolve() if args.annotations
        else raw_root / 'free_space_annotations.json'
    )
    data = json.loads(annotation_path.read_text(encoding='utf-8'))
    annotations = data.get('annotations')
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError('no free-space annotations in {}'.format(annotation_path))

    rows = []
    for session, annotation in sorted(annotations.items()):
        rows.extend(analyze_session(raw_root, session, annotation))

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_directory / 'table_perception_false_occupancy.csv')
    _write_markdown(rows, output_directory / 'table_perception_false_occupancy.md')
    # The sparse all-scenes bar chart is retained as a generated diagnostic,
    # while this case-study visualization is the intended paper figure.
    _plot(rows, output_directory / 'fig_false_occupied_cell_rate')
    _plot_case_study(
        rows, raw_root, annotations,
        output_directory / 'fig_false_occupancy_case_study',
    )
    source = output_directory / 'fig_false_occupied_cell_rate_source.txt'
    source.write_text(
        'Perception validation only; one static raw LiDAR scan per manually '
        'annotated capture. This does not modify the 60-run navigation '\
        'evaluation.\nannotations={}\n'.format(annotation_path),
        encoding='utf-8',
    )
    print('Wrote perception-validation table and figure to {}'.format(
        output_directory
    ))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Annotate verified obstacle-free areas in static LiDAR captures.

The perception-validation captures intentionally contain raw LiDAR, rather
than a semantic ground-truth image.  A polygon drawn with this utility marks
only the part of each scene that was known to be clear and traversable while
the capture was made.  ``analyze_perception_validation.py`` uses those
polygons to calculate false occupied XY-cell counts for B0, B1, and the
proposed local-plane model.

Run from the workspace after sourcing ROS:

  source install/setup.bash
  python3 tools/annotate_perception_free_space.py

For each window, use left click to add a polygon vertex, right click or
Backspace to undo the latest vertex, Enter to save, and Escape to skip it.
The coordinate system is vehicle-forward x / vehicle-left y in metres.
Select only the part that is certainly clear and intended to be traversable;
leaving ambiguous space outside the polygon is desirable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ubuntu's system Matplotlib in this workspace is compiled against NumPy 1.x,
# whereas an optional user-site NumPy 2.x may be active in an interactive
# shell.  Re-exec before importing NumPy/Matplotlib so the utility works with
# a plain ``python3 tools/annotate_perception_free_space.py`` command too.
if os.environ.get('PYTHONNOUSERSITE') != '1':
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'src' / 'terrain_navigation_pkg'))
from terrain_navigation_pkg.ground_obstacle_core import (  # noqa: E402
    extract_ground_relative_obstacles,
)


DEFAULT_RAW_ROOT = Path('/home/sukja/terrain_nav_data/perception_validation/raw')
ROI_X = (-4.0, 15.0)
ROI_Y = (-12.0, 12.0)

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

GLOBAL_PLANE = {
    'enabled': True,
    'local_ground_enabled': False,
    'local_ground_plane_enabled': False,
}
LOCAL_PLANE = {
    'enabled': True,
    'local_ground_enabled': True,
    'local_ground_plane_enabled': True,
}


def nav2_obstacle_mask(points: np.ndarray, options: dict) -> np.ndarray:
    """Reproduce the point selection performed by the Nav2 obstacle filter."""
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

    # The running node retains one point per 20-cm XYZ voxel before Nav2.
    indices = np.flatnonzero(mask)
    if indices.size:
        keys = np.floor(points[indices] / 0.20).astype(np.int32)
        _, unique = np.unique(keys, axis=0, return_index=True)
        chosen = indices[np.sort(unique)]
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[chosen] = True
    return mask


def _load_annotations(path: Path) -> dict:
    if not path.is_file():
        return {
            'coordinate_frame': 'vehicle: x forward, y left (metres)',
            'purpose': (
                'Manually verified obstacle-free, traversable polygons for '
                'perception validation.'
            ),
            'annotations': {},
        }
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data.get('annotations'), dict):
        raise ValueError('annotations file has no annotations object: {}'.format(path))
    return data


def _draw_annotation_window(session: Path, previous: list[list[float]]) -> list[list[float]] | None:
    sample_path = session / 'samples' / 'sample_000001.npz'
    arrays = np.load(sample_path)
    points = np.asarray(arrays['lidar_points_xyz'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('invalid raw LiDAR array: {}'.format(sample_path))

    display = (
        (points[:, 0] >= ROI_X[0]) & (points[:, 0] <= ROI_X[1])
        & (points[:, 1] >= ROI_Y[0]) & (points[:, 1] <= ROI_Y[1])
    )
    global_mask = display & nav2_obstacle_mask(points, GLOBAL_PLANE)
    local_mask = display & nav2_obstacle_mask(points, LOCAL_PLANE)

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    figure.subplots_adjust(bottom=0.15, top=0.90)
    axis.scatter(
        points[display, 0], points[display, 1], s=0.55, color='#A8A8A8',
        alpha=0.25, linewidths=0, rasterized=True, label='All LiDAR returns',
    )
    axis.scatter(
        points[global_mask, 0], points[global_mask, 1], s=2.0,
        color='#4A72A8', alpha=0.60, linewidths=0,
        label='B1 obstacle-cloud return',
    )
    axis.scatter(
        points[local_mask, 0], points[local_mask, 1], s=2.0,
        color='#5C8A5C', alpha=0.68, linewidths=0,
        label='Proposed obstacle-cloud return',
    )
    axis.scatter(0.0, 0.0, marker='^', s=58, color='#202020', zorder=6)
    axis.annotate(
        '', xy=(4.0, 0.0), xytext=(0.0, 0.0),
        arrowprops={'arrowstyle': '->', 'color': '#202020', 'lw': 1.0},
    )
    axis.text(4.15, 0.30, 'forward', fontsize=9, color='#303030')
    axis.set_xlim(*ROI_X)
    axis.set_ylim(*ROI_Y)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('Vehicle-forward x (m)')
    axis.set_ylabel('Vehicle-left y (m)')
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.94)
    axis.set_title('Verified clear traversable area: {}'.format(session.name),
                   fontsize=11)
    figure.text(
        0.5, 0.025,
        'Left click: add vertex    Right click / Backspace: undo    '
        'Enter: save polygon    Escape: skip',
        ha='center', va='bottom', fontsize=9,
    )

    vertices = [tuple(point) for point in previous]
    patch = None
    line = None
    saved = {'value': False}

    def redraw():
        nonlocal patch, line
        if patch is not None:
            patch.remove()
            patch = None
        if line is not None:
            line.remove()
            line = None
        if len(vertices) >= 2:
            closed = vertices + [vertices[0]]
            line, = axis.plot(
                [point[0] for point in closed], [point[1] for point in closed],
                color='#202020', linewidth=1.3, zorder=7,
            )
        if len(vertices) >= 3:
            patch = Polygon(
                vertices, closed=True, facecolor='#F2C14E', edgecolor='#202020',
                linewidth=1.2, alpha=0.25, zorder=6,
            )
            axis.add_patch(patch)
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not axis:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            vertices.append((float(event.xdata), float(event.ydata)))
            redraw()
        elif event.button == 3 and vertices:
            vertices.pop()
            redraw()

    def on_key(event):
        if event.key in ('backspace', 'delete') and vertices:
            vertices.pop()
            redraw()
        elif event.key in ('enter', 'return'):
            if len(vertices) < 3:
                print('Need at least three vertices before saving {}.'.format(session.name))
                return
            saved['value'] = True
            plt.close(figure)
        elif event.key == 'escape':
            plt.close(figure)

    figure.canvas.mpl_connect('button_press_event', on_click)
    figure.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    if not saved['value']:
        return None
    return [[round(x, 4), round(y, 4)] for x, y in vertices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        '--output', type=Path,
        help='Default: <raw-root>/free_space_annotations.json',
    )
    parser.add_argument(
        '--session', action='append', default=[],
        help='Session directory name; repeat to annotate only selected sessions.',
    )
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              raw_root / 'free_space_annotations.json')
    sessions = sorted(path for path in raw_root.glob('session_*') if path.is_dir())
    if args.session:
        requested = set(args.session)
        sessions = [path for path in sessions if path.name in requested]
        missing = requested - {path.name for path in sessions}
        if missing:
            raise FileNotFoundError('unknown session(s): {}'.format(', '.join(sorted(missing))))
    if not sessions:
        raise FileNotFoundError('no perception capture sessions under {}'.format(raw_root))

    annotations = _load_annotations(output)
    for session in sessions:
        prior = annotations['annotations'].get(session.name, {})
        vertices = _draw_annotation_window(
            session, prior.get('polygon_vehicle_xy', [])
        )
        if vertices is None:
            print('Skipped {}; annotation unchanged.'.format(session.name))
            continue
        metadata_path = session / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        annotations['annotations'][session.name] = {
            'capture_label': metadata.get('perception_capture_label', ''),
            'source_sample': str(session / 'samples' / 'sample_000001.npz'),
            'polygon_vehicle_xy': vertices,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(annotations, indent=2) + '\n', encoding='utf-8')
        print('Saved {} vertices for {} to {}'.format(
            len(vertices), session.name, output
        ))


if __name__ == '__main__':
    main()

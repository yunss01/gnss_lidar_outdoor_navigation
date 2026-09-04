#!/usr/bin/env python3
"""Register recorder sessions and generate ICCE-style evaluation tables."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from .navigation_evaluation_core import (
    RUN_FIELDS,
    evaluate_session,
    summarize_runs,
)


def _write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _declared_evaluation_variant(session_path):
    metadata_path = Path(session_path) / 'metadata.json'
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding='utf-8')
        )
    except (OSError, ValueError):
        return ''
    return str(metadata.get('evaluation_variant', '')).strip()


def _format(value):
    try:
        return '%.3f' % float(value)
    except (TypeError, ValueError):
        return str(value)


def _write_markdown(path, summaries):
    headers = [
        'Variant', 'Scenario', 'Success', 'Collision', 'Time (s)',
        'Path (m)', 'Stop (s)', 'Max stop (s)', 'Min clearance (m)',
    ]
    lines = ['# Navigation evaluation summary', '', '| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in summaries:
        lines.append('| ' + ' | '.join([
            row['variant'], row['scenario'],
            '%d/%d (%.1f%%)' % (
                row['successes'], row['runs'], 100.0 * row['success_rate']
            ),
            '%d/%d (%.1f%%)' % (
                row['collision_runs'], row['runs'], 100.0 * row['collision_rate']
            ),
            '%s ± %s' % (_format(row['duration_s_mean']), _format(row['duration_s_std'])),
            '%s ± %s' % (_format(row['path_length_m_mean']), _format(row['path_length_m_std'])),
            '%s ± %s' % (_format(row['total_stop_s_mean']), _format(row['total_stop_s_std'])),
            '%s ± %s' % (_format(row['max_stop_s_mean']), _format(row['max_stop_s_std'])),
            '%s ± %s' % (_format(row['minimum_clearance_m_mean']), _format(row['minimum_clearance_m_std'])),
        ]) + ' |')
    lines.extend([
        '',
        'Mean ± sample standard deviation uses successful runs for time/path metrics.',
        'All attempted runs are included in success and collision rates.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Register F9/F10 runs and update evaluation CSV/Markdown.'
    )
    parser.add_argument('--input-root', default='~/terrain_nav_data/learning/raw')
    parser.add_argument('--output-dir', default='~/terrain_nav_data/evaluation/icce_asia_2026')
    parser.add_argument('--variant', help='proposed, b0_fixed_ground, or b1_plane_ground')
    parser.add_argument('--scenario', choices=['F9', 'F10', 'f9', 'f10'])
    parser.add_argument('--latest', type=int, default=0,
                        help='register the newest N sessions not already registered')
    parser.add_argument('--session', action='append', default=[],
                        help='explicit session directory; may be repeated')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser()
    registry = output / 'runs.csv'
    runs = _read_csv(registry)
    known = {row['session'] for row in runs}

    if not args.report_only:
        if not args.variant or not args.scenario:
            parser.error('--variant and --scenario are required when registering runs')
        candidates = [Path(item).expanduser() for item in args.session]
        if args.latest:
            root = Path(args.input_root).expanduser()
            discovered = sorted(
                (path for path in root.glob('session_*')
                 if (path / 'metadata.json').exists() and path.name not in known),
                key=lambda path: path.name,
            )
            if len(discovered) < args.latest:
                parser.error(
                    'requested newest %d unregistered sessions, but only %d '
                    'exist' % (args.latest, len(discovered))
                )
            candidates.extend(discovered[-args.latest:])
        unique = []
        seen = set()
        for path in candidates:
            if path.name not in seen and path.name not in known:
                unique.append(path)
                seen.add(path.name)
        if not unique:
            parser.error('no new sessions selected; use --latest N or --session PATH')
        declared = [
            (path.name, _declared_evaluation_variant(path))
            for path in unique
        ]
        mismatches = [
            (session, variant) for session, variant in declared
            if variant and variant != args.variant
        ]
        if mismatches:
            details = ', '.join(
                '{} declares {}'.format(session, declared)
                for session, declared in mismatches
            )
            parser.error(
                'requested variant {} does not match recorder metadata: {}'.format(
                    args.variant, details
                )
            )
        added = [
            evaluate_session(path, args.variant, args.scenario)
            for path in unique
        ]
        runs.extend(added)
        _write_csv(registry, runs, RUN_FIELDS)
        batch = output / ('batch_%s_%s_%s.csv' % (
            args.variant, args.scenario.upper(),
            datetime.now().strftime('%Y%m%d_%H%M%S'),
        ))
        _write_csv(batch, added, RUN_FIELDS)
        print('Registered %d run(s): %s' % (len(added), batch))

    summaries = summarize_runs(runs)
    _write_csv(output / 'summary.csv', summaries)
    _write_markdown(output / 'summary.md', summaries)
    print('Registry: %s' % registry)
    print('Summary:  %s' % (output / 'summary.md'))
    for row in summaries:
        print('%-18s %-3s success=%d/%d collision=%d/%d' % (
            row['variant'], row['scenario'], row['successes'], row['runs'],
            row['collision_runs'], row['runs'],
        ))

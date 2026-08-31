# Navigation evaluation

The recorder starts and closes one session automatically for every F9/F10
attempt. In the default paper-evaluation configuration it writes lightweight
`metadata.json` and `frames.csv` files but no LiDAR NPZ tensors.

## Experimental matrix

Use the same CARLA map, spawn pose, waypoint set, 4 km/h speed and simulator
timestep for every cell:

| Variant | F9 | F10 |
|---|---:|---:|
| `proposed` | 10 | 10 |
| `b0_fixed_ground` | 10 | 10 |
| `b1_plane_ground` | 10 | 10 |

This is 60 attempted runs. Keep failed and collision runs; never repeat or
discard an attempt merely because it failed. Interleave variant order when
possible to reduce time/order bias.

After completing one ten-run cell, register the ten newest unregistered
sessions:

```bash
source ~/terrain_nav_ws/install/setup.bash
ros2 run terrain_navigation_pkg evaluate_navigation_runs \
  --variant proposed \
  --scenario F9 \
  --latest 10
```

Change `--variant` and `--scenario` for the other cells. The registry is
append-only by session ID, so an already registered run is not counted twice.
Generated outputs are:

- `~/terrain_nav_data/evaluation/icce_asia_2026/runs.csv`: one row per attempt
- `summary.csv`: grouped statistics for analysis/plotting
- `summary.md`: a human-readable paper-table preview
- `batch_*.csv`: immutable record of each registration command

Regenerate the summaries without registering runs:

```bash
ros2 run terrain_navigation_pkg evaluate_navigation_runs --report-only
```

Primary metrics are mission success rate and collision rate. Secondary metrics
include completion time, driven distance, total/maximum stop duration,
obstacle-stop duration, invalid-path duration, and minimum recorded path
clearance. Time and distance statistics are calculated on successful runs;
success and collision rates use all attempts.

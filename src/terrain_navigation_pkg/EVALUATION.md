# Navigation evaluation

The recorder starts and closes one session automatically for every F9/F10
attempt. In the default paper-evaluation configuration it writes lightweight
`metadata.json` and `frames.csv` files but no LiDAR NPZ tensors.

## Experimental matrix

Use the same CARLA map, spawn pose, waypoint set, 4 km/h speed and simulator
timestep for every cell:

| Variant | F9 | F10 |
|---|---:|---:|
| `proposed` (`proposed_recovery_v2` in the frozen registry) | 10 | 10 |
| `b0_fixed_ground` | 10 | 10 |
| `b1_plane_ground` | 10 | 10 |

This is 60 attempted runs. Keep failed and collision runs; never repeat or
discard an attempt merely because it failed. Interleave variant order when
possible to reduce time/order bias.

The proposed cell collected on 2026-08-31 is already registered as
`proposed_recovery_v2`.  Select a ground-processing ablation at launch time;
the same override is applied to both the costmap obstacle filter and the
independent raw-LiDAR emergency-stop filter.  The path-clearance validator
consumes the resulting filtered obstacle cloud, so it changes transitively:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true guide_mode:=far \
  evaluation_variant:=b0_fixed_ground
```

Valid values are:

- `proposed`: local plane per road neighbourhood (the frozen proposed system)
- `b0_fixed_ground`: legacy fixed sensor-frame Z crop only
- `b1_plane_ground`: one scan-wide fitted ground plane, without local fitting

The launch prints `ICCE evaluation variant: ...`, and the recorder writes the
same value to each session's `metadata.json`.  Restart the complete Nav2 launch
when changing variants; do not change the parameter during a route.  The
evaluation command rejects a batch if its requested `--variant` disagrees with
the recorder metadata.

Collect and register one complete cell before changing variant or scenario:

```bash
# B0 / F9: launch this profile for every attempt and press F9.
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true guide_mode:=far \
  evaluation_variant:=b0_fixed_ground

# After exactly 10 B0/F9 attempts:
ros2 run terrain_navigation_pkg evaluate_navigation_runs \
  --variant b0_fixed_ground --scenario F9 --latest 10
```

Then collect B0/F10 and register it with `--scenario F10`.  Repeat the same
two cells with `evaluation_variant:=b1_plane_ground` and matching
`--variant b1_plane_ground`.  The proposed F9/F10 cells are already frozen and
must not be repeated or replaced.

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

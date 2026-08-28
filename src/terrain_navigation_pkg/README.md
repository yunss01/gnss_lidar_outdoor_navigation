# Terrain navigation

This package contains the current outdoor-navigation stack. It uses GNSS
waypoints, odometry/IMU state, and 3D LiDAR obstacle geometry to drive a
forward-only Ackermann vehicle. Camera/IMU roughness prediction is not part of
this repository's navigation architecture.

The planner policy is defined in
[`PLANNING_DESIGN.md`](PLANNING_DESIGN.md) documents the safety boundary,
rolling local map, Nav2 planning, and GNSS waypoint policy.

## Navigation learning recorder

`terrain_navigation_nav2.launch.py` starts a subscriber-only recorder by
default. Pressing F9 or F10 starts a session automatically; completion,
failure, cancellation, or shutdown closes it. The recorder never publishes a
control command, and compressed file I/O runs on a background thread.

```text
~/terrain_nav_data/learning/raw/session_YYYYMMDD_HHMMSS_microseconds/
  metadata.json
  frames.csv
  samples/sample_000001.npz
```

Each sample contains the clipped raw 3D points, four-channel vehicle-centric
LiDAR BEV, local/global costmap crops, Nav2 plan, long-range guide and subgoal,
goal direction, odometry, teacher commands, and safety state. This preserves
both successful and failed teacher behavior for later filtering and local
policy training. Disable recording for a run with:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  record_learning_data:=false
```

## GNSS goal manager

`gnss_goal_manager_node` subscribes to:

- `/vectornav/gnss` (`sensor_msgs/NavSatFix`)
- `/vehicle/odometry` (`nav_msgs/Odometry`)
- `/navigation/goal_gnss` (`sensor_msgs/NavSatFix`)

It fixes the first valid GNSS sample as a local WGS84 ENU origin and publishes:

- `/navigation/current_local`
- `/navigation/goal_local`
- `/navigation/goal_pose`
- `/navigation/goal_vector`
- `/navigation/distance_to_goal`
- `/navigation/bearing_to_goal_deg`
- `/navigation/goal_reached`
- `/navigation/status`
- `/navigation/direct_path`

`direct_path` is only the unobstructed straight-line goal reference. It must
not be treated as a collision-free global plan. A map-based planner will
replace it in the next stage.

Set a goal after the node starts:

```bash
ros2 topic pub --once /navigation/goal_gnss sensor_msgs/msg/NavSatFix \
  "{latitude: 37.0, longitude: 127.0, altitude: 50.0}"
```

Or start the goal manager and low-speed GPS controller together:

```bash
ros2 launch launch_pkg terrain_navigation.launch.py \
  goal_enabled:=true \
  goal_latitude:=37.0 \
  goal_longitude:=127.0 \
  goal_altitude:=50.0
```

The legacy launch keeps the optional 3D terrain mapper disabled. It is not
required for the first A-to-B test.

## Phase 1: low-speed A-to-B test

First load a normal CARLA town instead of Mine. This CARLA installation ships
the rendered `Town10HD_Opt` map; its streets, buildings, intersections, and
open areas are suitable for the later waypoint loop:

```bash
python3 ~/carla/0.10.0/PythonAPI/util/config.py --map Town10HD_Opt
```

Start the vehicle client in explicit ROS control mode. This mode cannot be
combined with CARLA autopilot, and a missing `/cmd_vel` applies full brake
after 0.5 seconds:

```bash
source /opt/ros/humble/setup.bash
python3 ~/carla/0.10.0/PythonAPI/examples/manual_control.py \
  --filter 'vehicle.lincoln.mkz*' \
  --sync \
  --fixed-delta-seconds 0.05 \
  --external-control
```

Then launch navigation with the desired WGS84 coordinate:

```bash
source /opt/ros/humble/setup.bash
source ~/terrain_nav_ws/install/setup.bash
ros2 launch launch_pkg terrain_navigation.launch.py \
  goal_enabled:=true \
  goal_latitude:=DESTINATION_LATITUDE \
  goal_longitude:=DESTINATION_LONGITUDE \
  goal_altitude:=DESTINATION_ALTITUDE
```

`gps_go_to_goal_controller_node` consumes the goal vector, goal distance, and
vehicle yaw and publishes `/cmd_vel_navigation`. Defaults are intentionally
conservative: maximum 3.0 m/s (10.8 km/h), slowdown inside 12 m, and stop inside
3 m.
While `manual_control.py` is running, press `Y` to pause ROS control, position
the car manually at A, and press `Y` again to hand control back to ROS.

`lidar_emergency_stop_node` is the only node that publishes the final
`/cmd_vel`. It stops immediately when at least 20 non-ground returns are found
inside the 2.7 m-wide corridor up to 6 m ahead. It resumes only after the 8 m
corridor is clear for three consecutive LiDAR scans. Missing LiDAR or command
data also produces a stop. Returns closer than 2.5 m to the roof LiDAR are
excluded as the Lincoln's own hood/ego footprint.

Safety diagnostics are published on:

- `/safety/state`
- `/safety/emergency_stop`
- `/safety/nearest_obstacle_distance`
- `/safety/obstacle_points`

Every run writes a flushed timeline to:

```text
~/terrain_nav_data/logs/safety/run_YYYYMMDD_HHMMSS_microseconds/safety.csv
```

The emergency-stop layer itself only stops; steering detours are produced by
the local-avoidance layer below.

## Smooth Ackermann waypoint routes

The F7/F9 waypoint route is converted from WGS84 to the same local ENU frame
as the vehicle. Internal corners are rounded inside the waypoint polyline and
remain within 2 m of each waypoint, inside the default 3 m arrival radius.
The controller tracks `/navigation/smoothed_path` with forward-only Pure
Pursuit and limits curvature to the Lincoln MKZ wheelbase and steering-angle
parameters. Consequently, the vehicle follows a continuous curve instead of
aiming at each waypoint as an independent straight segment.

Route construction occurs once when START ROUTE is pressed. At 0.5 m path
spacing, normal routes contain only a few hundred points; each 10 Hz control
cycle searches a small forward window. The cost is negligible compared with
CARLA rendering and 3D LiDAR processing. In RViz, add a `Path` display for
`/navigation/smoothed_path` to inspect the exact reference curve.

## Single-obstacle local avoidance

`local_avoidance_node` sits between GNSS guidance and the emergency-stop gate:

```text
/cmd_vel_navigation
  -> local_avoidance_node
  -> /cmd_vel_avoidance
  -> lidar_emergency_stop_node
  -> /cmd_vel
```

The avoidance path is an odometry-anchored smooth S-curve tracked with pure
pursuit. Each left/right candidate checks the complete swept corridor using the
configured vehicle width plus lateral clearance. The pass phase is locked until
odometry confirms the requested lateral offset for repeated control cycles.
Downstream emergency-stop time is excluded from the active detour timeout.
Obstacles located beyond the remaining GNSS stopping point are ignored so they
do not trigger an unnecessary detour immediately before arrival.

It checks centre, left, and right corridors out to 17 m. When the centre is
blocked, it locks one open side and shifts to the configured clearance line.
It does not assume an obstacle length or a fixed passing distance. The roof
LiDAR instead anchors the initially detected obstacle in the odometry-aligned
detour frame and expands its longitudinal extent only with spatially connected
returns. This prevents unrelated buildings, poles, and kerbs farther along the
road from extending the pass phase. Once enough returns from the tracked object
cross behind the rear boundary, its measured extent is frozen so later city
clutter cannot be appended to it. The vehicle holds the offset until the frozen
rear extent moves behind the rear bumper plus the configured clearance and the
return corridor stays clear for three consecutive scans. It then
creates the smooth return portion and rejoins navigation guidance. For an
F7/F9 route, recovery uses the nearest forward segment and tangent on
`/navigation/smoothed_path`; a single F6 goal retains the straight goal-line
fallback. Pressing START ROUTE clears any unfinished avoidance state before
the new path begins, so an old recovery line cannot steer a new route. The
downstream 6 m emergency stop remains active throughout the manoeuvre. If both
sides are
blocked, the local node publishes zero velocity. A manoeuvre timeout also stops
the vehicle instead of returning without LiDAR passage confirmation.

Useful diagnostics are:

- `/avoidance/state`
- `/avoidance/selected_side`
- `/avoidance/center_points`
- `/avoidance/left_points`
- `/avoidance/right_points`
- `/avoidance/path`
- `/avoidance/actual_lateral_offset`

The avoidance CSV also records the current phase, points that have not passed
the rear boundary, points confirmed behind it, the consecutive-clear count,
the return-corridor point count, and the tracked obstacle's minimum/maximum
progress. These fields distinguish a genuine LiDAR pass confirmation from a
timeout or emergency stop.

Runs are logged to:

```text
~/terrain_nav_data/logs/avoidance/run_*/avoidance.csv
```

The launch starts avoidance by default. To test the same command chain without
modifying GNSS commands, use `avoidance_enabled:=false`; do not stop the local
node itself because the downstream safety gate intentionally treats a missing
command as unsafe.

Every run writes a flushed navigation timeline to:

```text
~/terrain_nav_data/logs/navigation/run_YYYYMMDD_HHMMSS_microseconds/
  navigation.csv
  run_metadata.json
```

The CSV includes raw GNSS, local ENU position, odometry, speed, destination,
distance, bearing, arrival state, sensor ages, and status. It is safe to
inspect while the node is running.

## Nav2 LiDAR obstacle and clearing clouds

`lidar_obstacle_filter_node` publishes two complementary clouds:

- `/lidar/nav2_obstacles`: locally ground-fitted obstacle returns used only
  for costmap marking.
- `/lidar/nav2_clearing`: a sparse full-scan cloud used only for costmap
  raytracing/clearing. Separating these sources prevents a transient road
  false-positive from remaining in the costmap after it disappears from the
  obstacle-filtered cloud.

The local ground estimator fits a small plane around each XY grid cell. It
therefore follows road pitch and crown while retaining abrupt curb/step
returns. Returns consistent with that surface are removed even when an
uphill road rises above the legacy sensor-frame fixed-Z threshold.

## RViz navigation diagnostics

`navigation_visualization_node` is started by
`terrain_navigation.launch.py`. It does not modify any control command. It
translates the `map`-frame reference route into `odom` and records a
distance-sampled driven path for visualization:

- `/navigation/smoothed_path_odom`: blue global reference route
- `/avoidance/path`: red currently selected local trajectory
- `/navigation/driven_path`: white accumulated vehicle track
- `/lidar/points`: height-coloured live 3D LiDAR scan

Open the saved top-down layout in a second terminal:

```bash
ros2 launch launch_pkg terrain_navigation_rviz.launch.py
```

The RViz fixed frame is `odom`, and the camera follows `base_link`. Mouse-wheel
zoom can be used to switch between the local avoidance view and the complete
building route. A new F9 waypoint route clears the accumulated green path.

`F9 / OPEN ROUTE` drives the saved targets in the order
`WP1 -> WP2 -> ... -> WPn` and stops at the final waypoint. `F10 / CLOSED LAP`
treats WP1 as the start/finish marker and drives
`WP2 -> ... -> WPn -> WP1`, then stops after one lap. Both modes use the same
two-target rolling horizon, so the mapless local costmap only has to plan the
current target and one preview target instead of validating the entire distant
route at once. F10 deliberately does not use the route protocol's infinite
`loop` flag, so a closed-lap test cannot continue circling unattended.

For both rolling modes, a two-pose preview is used on long segments. When the
vehicle comes within `rolling_waypoint_focus_distance_m` (6 m by default) of
the current intermediate waypoint, the bridge temporarily replaces that
preview with a current-waypoint-only action. This prevents the 3.5 m RPP
lookahead from cutting the corner toward the following waypoint outside the
strict 1 m capture/crossing corridor. Passing the waypoint immediately starts
the next two-pose rolling window, so the long-range controller settings remain
unchanged.

If Nav2 aborts a two-pose rolling preview because the second, future waypoint
is not yet connected in the observed rolling costmap, the bridge automatically
retries the same current waypoint as a one-pose action. This prevents an
unobservable or temporarily blocked preview waypoint from stopping progress to
the reachable current waypoint. A one-pose retry is attempted only once for
that waypoint; failure of the retry remains a real navigation failure.

The committed avoidance side is not released merely because the centre
corridor is briefly clear. The vehicle must first achieve the configured
odometry-measured lateral shift (`trajectory_side_lock_minimum_lateral_m`).
This prevents an early turn back into the same obstacle.

## Nav2 long-obstacle preview mode

The legacy GNSS/Pure-Pursuit/local-trajectory chain remains the default. A
separate Nav2 launch is provided for building-scale planning so comparison and
rollback do not require editing the working launch file.

Preview first (Nav2 is physically disconnected from CARLA `/cmd_vel`):

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=false
```

With CARLA sensors and TF running, press F9 for an open rolling route or F10
for one closed rolling lap. Inspect `/global_costmap/costmap`, `/local_costmap/costmap`,
`/plan`, `/lidar/nav2_obstacles`, and `/navigation/nav2_goal` in the existing
terrain-navigation RViz configuration. The preview launch may publish
`/cmd_vel_nav2`, but no node forwards that topic to the vehicle.

Only after the costmaps and Smac path are visibly correct, enable driving:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true
```

Do not run `terrain_navigation.launch.py` at the same time. In Nav2 drive
mode, commands follow this explicit boundary:

```text
Smac Hybrid-A* -> MPPI Ackermann -> velocity smoother (/cmd_vel_nav2)
  -> lidar_emergency_stop_node -> CARLA (/cmd_vel)
```

`lidar_obstacle_filter_node` removes ground/sky returns in the roof-LiDAR
frame and voxel-downsamples the scan before both costmaps consume it. The
global costmap is a 100 x 100 m rolling online map, so it supports the current
container/building test and manually supplied waypoint corridors without a
pre-authored map. A destination farther than this planning window should be
split into intermediate GNSS waypoints. Persistent campus-scale mapping or a
rolling subgoal manager is intentionally a later stage.

The planner uses the Lincoln footprint, 0.35 m padding, and a 4.1 m minimum
turning radius. The custom behavior tree replans at 1 Hz but contains no
in-place Spin recovery, since that motion is impossible for the real car.

## FAR-inspired long-range guide mode

The original `direct` bridge remains the default so the previously verified
F9/F10 behavior is not silently changed. An experimental two-level mode is
available for routes whose current GNSS waypoint is difficult for Smac Hybrid
to solve as one long Ackermann search:

```text
F9/F10 GNSS waypoint
  -> online coarse occupancy-grid search
  -> line-of-sight visibility vertices (long guide)
  -> next 12 m subgoal with guide tangent
  -> Smac Hybrid-A* short Ackermann path
  -> RPP controller -> safety gates -> vehicle
```

This first implementation is **FAR-inspired**, not a claim that the official
FAR Planner repository has been copied verbatim. The official system expects
its own terrain-cloud/intensity contract. Here, the existing Nav2 rolling
costmap is used directly so the architecture can be tested without replacing
the verified LiDAR, TF, command, and safety interfaces. Unknown cells remain
searchable with a penalty; observed free space is preferred. Smac Hybrid and
the independent safety layers still have final authority over actual motion.

Build and test it with the same saved F7 waypoints:

```bash
cd ~/terrain_nav_ws
colcon build --symlink-install \
  --packages-select terrain_navigation_pkg config_pkg launch_pkg
source install/setup.bash

ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  guide_mode:=far
```

Then press F9 for an open route or F10 for a closed lap. In RViz:

- green `/navigation/far_guide_path`: long online guide
- orange `/navigation/far_subgoal`: the current short Smac target
- magenta `/plan`: the actual kinematically feasible Smac Hybrid path

Useful diagnostics are:

```bash
ros2 topic echo /navigation/far_guide_status
ros2 topic echo /navigation/far_subgoal
```

The proposed stability profile keeps a still-hard-valid active Hybrid
segment sticky. A newly revealed rolling-costmap guide cannot cancel the
active action merely because it is shorter; persistent hard blockage and
Nav2 action failure still trigger replanning. The former optimization remains
available for an ablation run:

```bash
ros2 param set /far_nav2_guide_node \
  active_replan_allow_valid_path_optimization true
```

Restart the launch or set the parameter back to `false` before evaluating the
proposed profile.

Each run is logged under
`~/terrain_nav_data/logs/far_guide/run_*/far_guide.csv`. To return to the
previous behavior for regression comparison, use `guide_mode:=direct` or omit
the argument entirely.

The coarse guide treats OccupancyGrid values 99 and 100 as hard collision
space. Lower inflation values remain traversable soft costs. This preserves
the vehicle's inscribed collision boundary without sealing a usable corridor
merely because it lies inside Nav2's preferred-clearance halo.

## YAML mission routes

Routes measured beforehand can be written directly as WGS84 coordinates in a
YAML file. The loader reads and validates that file, converts it in memory to
the existing `/navigation/waypoint_route` JSON message, and publishes it. It
does not copy anything into Pygame's
`~/.config/terrain_navigation/waypoints.json`, so F7/F9/F10 and YAML missions
do not overwrite each other's saved files.

Copy the installed template or edit the source template:

```text
~/terrain_nav_ws/src/config_pkg/config/routes/mission_route_template.yaml
```

With the Nav2 launch already running, publish and start a mission in another
terminal:

```bash
ros2 launch launch_pkg terrain_mission_route.launch.py \
  route_file:=/absolute/path/to/my_route.yaml \
  start_route:=true
```

Alternatively, start Nav2 and the route loader together:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  mission_route_enabled:=true \
  mission_route_file:=/absolute/path/to/my_route.yaml \
  mission_route_start:=true
```

`start_route:=false` only loads the route. `start_route:=true` overrides a
false `auto_start` value in YAML without editing the file. The loader remains
alive after its one latched publication so late subscribers can still receive
the route. `/navigation/mission_route_status` reports the loaded/published/
started state. Only one source should actively start a route at a time; a
later F9/F10 press intentionally replaces the currently loaded runtime route.

The `temporary_waypoints` block is reserved for automatic 25--30 m mission
subdivision. It is validated now but not yet applied to navigation.

## Rolling 3D terrain mapper

`terrain_mapping_node` transforms every complete `/lidar/points` revolution
into `odom`, removes returns inside the ego-vehicle footprint, and fuses
repeated observations in a bounded 3D voxel map. It publishes:

- `/terrain/map_points`: persistent fused 3D voxels
- `/terrain/slope_costmap`: slope magnitude encoded from 0 to 100
- `/terrain/traversability_costmap`: combined slope and step cost

Both costmaps use `-1` for unknown cells. A cost of 0 is locally flat, and 100
meets or exceeds either the configured lethal slope or lethal step threshold.
The default map retains voxels within 40 m of the vehicle, rather than using
RViz's temporary PointCloud2 decay buffer.

Run the mapper by itself:

```bash
ros2 run terrain_navigation_pkg terrain_mapping_node \
  --ros-args \
  --params-file ~/terrain_nav_ws/src/config_pkg/config/params.yaml
```

In RViz, keep `Fixed Frame` at `odom`, set `/terrain/map_points` to
`Decay Time = 0`, and add both OccupancyGrid topics. The node itself owns the
map lifetime, so RViz must only display the latest published map.

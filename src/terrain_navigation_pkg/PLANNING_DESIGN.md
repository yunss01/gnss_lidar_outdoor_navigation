# GNSS and 3D LiDAR navigation policy

The system separates mission guidance, collision-aware planning, control, and
the final safety gate.

## Mission guidance

- Human-authored WGS84 waypoints define the mission-level route.
- F9 executes an open route and stops at the final waypoint.
- F10 executes the saved loop and returns to its starting area.
- A two-target rolling horizon exposes only the current and next waypoint to
  Nav2. This allows long missions without a prebuilt global occupancy map.
- Intermediate waypoint completion accepts arrival, safe pass-through, or a
  confirmed along-track overshoot so a forward-only vehicle is not forced to
  reverse for a narrowly missed point.

## Online obstacle map and planning

- The 3D LiDAR cloud is filtered into obstacle points in the odometry frame.
- Nav2 rolling local costmaps contain only currently observed geometry and its
  configured vehicle-clearance inflation; they are not a prior city map.
- The planner generates a collision-free path through the observed local map.
- The controller tracks that path with Ackermann-compatible forward motion.
- The map and plan are updated continuously as new surfaces become visible.

## Optional geometric terrain map

`terrain_mapping_node` computes slope, step height, and geometric roughness
from 3D LiDAR elevations. These are deterministic geometry measurements, not
camera/IMU learned roughness labels. The node remains optional until curb and
ground-traversability experiments resume.

## Final safety contract

The independent LiDAR safety node is the only component allowed to publish the
final `/cmd_vel` command. A stale command, stale or invalid point cloud,
collision latch, or occupied swept path produces zero velocity. Planning
success can never override this final stop decision.

The simulation and real-vehicle backends consume the same safety-gated command:

```text
GNSS mission -> Nav2 planner/controller -> /cmd_vel_nav2
             -> LiDAR safety gate -> /cmd_vel
             -> CARLA external control or Arduino vehicle interface
```

## Development order

1. Revalidate F9 and F10 at the conservative 4 km/h simulation speed.
2. Load repeatable WGS84 missions from YAML.
3. Add automatic temporary waypoint subdivision for long A-to-B missions.
4. Calibrate the real vehicle footprint, wheelbase, steering limits, GNSS,
   IMU, and 3D LiDAR extrinsics before powered outdoor tests.

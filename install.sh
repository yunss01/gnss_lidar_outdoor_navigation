#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
  exit 1
fi

source /opt/ros/humble/setup.bash

echo "Installing declared ROS dependencies from ${WORKSPACE_DIR}/src"
rosdep install \
  --from-paths "${WORKSPACE_DIR}/src" \
  --ignore-src \
  --rosdistro humble \
  --skip-keys carla \
  -r -y

echo "Building terrain_nav_ws"
cd "${WORKSPACE_DIR}"
colcon build --symlink-install

echo "Done. Run: source ${WORKSPACE_DIR}/install/setup.bash"

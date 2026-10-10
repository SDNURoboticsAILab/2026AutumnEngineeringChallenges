#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-cyberdog_sim_level2}"

exec sudo docker exec -it "$CONTAINER" bash -lc '
set -eo pipefail
export AMENT_TRACE_SETUP_FILES=
source /opt/ros/galactic/setup.bash
source /home/cyberdog_ws/install/setup.bash
cd /home/cyberdog_ws
exec ros2 run motion_manager motion_manager
'

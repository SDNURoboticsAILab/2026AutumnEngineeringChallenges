#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-cyberdog_sim_level2}"
CHECK_SCRIPT="/tmp/cyberdog_check_camera.py"

sudo docker cp "$SCRIPT_DIR/check_camera.py" "$CONTAINER:$CHECK_SCRIPT"
sudo docker exec "$CONTAINER" bash -lc '
set -eo pipefail
export AMENT_TRACE_SETUP_FILES=
source /opt/ros/galactic/setup.bash
source /home/cyberdog_sim/install/setup.bash
timeout 20 python3 /tmp/cyberdog_check_camera.py
'

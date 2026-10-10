#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../config/gazebo.xacro.level4"
CONTAINER="${CONTAINER:-cyberdog_sim_level2}"
TARGET_FILE="/home/cyberdog_sim/src/cyberdog_simulator/cyberdog_robot/cyberdog_description/xacro/gazebo.xacro"
BACKUP_FILE="${TARGET_FILE}.bak.level4"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "相机配置文件不存在：$CONFIG_FILE" >&2
  exit 1
fi

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" != "true" ]]; then
  echo "容器 $CONTAINER 未运行，请先运行 scripts/start_container.sh。" >&2
  exit 1
fi

if ! sudo docker exec "$CONTAINER" test -f "$BACKUP_FILE"; then
  sudo docker exec "$CONTAINER" cp -a "$TARGET_FILE" "$BACKUP_FILE"
  echo "已备份原始配置：$BACKUP_FILE"
fi

sudo docker cp "$CONFIG_FILE" "$CONTAINER:$TARGET_FILE"
sudo docker exec "$CONTAINER" chmod 644 "$TARGET_FILE"
sudo docker exec "$CONTAINER" bash -lc '
set -eo pipefail
export AMENT_TRACE_SETUP_FILES=
source /opt/ros/galactic/setup.bash
source /home/cyberdog_sim/install/setup.bash
cd /home/cyberdog_sim
xacro src/cyberdog_simulator/cyberdog_robot/cyberdog_description/xacro/robot.xacro USE_LIDAR:=true > /tmp/cyberdog_level4.urdf
grep -q "race_rgb_camera" /tmp/cyberdog_level4.urdf
grep -q "libgazebo_ros_camera.so" /tmp/cyberdog_level4.urdf
echo "相机配置已应用，xacro 校验通过。"
'

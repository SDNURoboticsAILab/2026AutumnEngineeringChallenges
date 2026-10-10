#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-cyberdog_sim_level2}"
DURATION="${1:-8}"

if ! [[ "$DURATION" =~ ^[1-9][0-9]*$ ]]; then
  echo "用法：$0 [前进持续时间秒数，默认 8]" >&2
  exit 1
fi

sudo docker exec -it "$CONTAINER" bash -lc "
set -eo pipefail
export AMENT_TRACE_SETUP_FILES=
source /opt/ros/galactic/setup.bash
source /home/cyberdog_ws/install/setup.bash

echo '=== 1. 激活状态机 ==='
ros2 service call /motion_managermachine_service protocol/srv/FsMachine \"{target_state: 'Active'}\"
sleep 2

echo '=== 2. 恢复站立姿态：motion_id=111 ==='
ros2 service call /motion_result_cmd protocol/srv/MotionResultCmd '{motion_id: 111}'
sleep 3

echo '=== 3. 慢速前进：motion_id=303，持续 ${DURATION} 秒 ==='
timeout ${DURATION} ros2 topic pub -r 20 /motion_servo_cmd protocol/msg/MotionServoCmd '{motion_id: 303, vel_des: [0.2, 0.0, 0.0]}' || true
sleep 1

echo '=== 4. 快速前进：motion_id=305，持续 4 秒 ==='
timeout 4 ros2 topic pub -r 10 /motion_servo_cmd protocol/msg/MotionServoCmd '{motion_id: 305, vel_des: [0.6, 0.0, 0.0]}' || true
sleep 1

echo '=== 5. 再次恢复站立姿态 ==='
ros2 service call /motion_result_cmd protocol/srv/MotionResultCmd '{motion_id: 111}'
"

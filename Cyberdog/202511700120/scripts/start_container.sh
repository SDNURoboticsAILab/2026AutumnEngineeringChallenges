#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cyberdog_sim:v2026}"
CONTAINER="${CONTAINER:-cyberdog_sim_level2}"
DISPLAY_VALUE="${DISPLAY:-:0}"

if ! sudo docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "镜像 $IMAGE_NAME 不存在，请先运行 scripts/load_image.sh。" >&2
  exit 1
fi

if ! command -v xhost >/dev/null 2>&1; then
  echo "未找到 xhost，请先安装 x11-xserver-utils。" >&2
  exit 1
fi

xhost +SI:localuser:root >/dev/null

if sudo docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  STATE="$(sudo docker inspect -f '{{.State.Status}}' "$CONTAINER")"
  echo "容器 $CONTAINER 已存在，当前状态：$STATE"
  if [[ "$STATE" != "running" ]]; then
    sudo docker start "$CONTAINER" >/dev/null
  fi
  exec sudo docker exec -it "$CONTAINER" bash
fi

echo "创建并进入容器：$CONTAINER"
exec sudo docker run -it \
  --name "$CONTAINER" \
  --shm-size="1g" \
  --privileged=true \
  -e DISPLAY="$DISPLAY_VALUE" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "$IMAGE_NAME"

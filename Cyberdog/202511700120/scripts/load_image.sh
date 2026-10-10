#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cyberdog_sim:v2026}"
SOURCE_TAG="${SOURCE_TAG:-cyberdog_sim:v2}"
IMAGE_TAR="${1:-}"

if [[ -z "$IMAGE_TAR" ]]; then
  echo "用法：$0 /path/to/cyberdog_raceV2.tar" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_TAR" ]]; then
  echo "镜像文件不存在：$IMAGE_TAR" >&2
  exit 1
fi

if sudo docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "镜像 $IMAGE_NAME 已存在，跳过导入。"
elif sudo docker image inspect "$SOURCE_TAG" >/dev/null 2>&1; then
  echo "镜像原始标签 $SOURCE_TAG 已存在，补充 $IMAGE_NAME 标签。"
  sudo docker tag "$SOURCE_TAG" "$IMAGE_NAME"
else
  echo "正在导入镜像，文件较大，可能需要数分钟：$IMAGE_TAR"
  sudo docker load -i "$IMAGE_TAR"
  if ! sudo docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 && \
     sudo docker image inspect "$SOURCE_TAG" >/dev/null 2>&1; then
    sudo docker tag "$SOURCE_TAG" "$IMAGE_NAME"
  fi
fi

sudo docker image inspect "$IMAGE_NAME" >/dev/null
sudo docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}'

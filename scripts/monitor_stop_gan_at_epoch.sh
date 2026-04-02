#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/autodl-tmp/CNNProject1}"
TARGET_EPOCH="${2:-3}"
cd "${PROJECT_DIR}"

while true; do
  if grep -q "第 ${TARGET_EPOCH} 轮训练完成" "gan——results/train.log" 2>/dev/null; then
    pkill -f "python3 scripts/train_gan.py" || true
    echo "已在第 ${TARGET_EPOCH} 轮训练完成后停止 GAN 训练" > stop_marker.log
    exit 0
  fi
  sleep 20
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/autodl-tmp/CNNProject1}"
cd "${PROJECT_DIR}"

while true; do
  if grep -q "第 7 轮训练完成" "gan——results/train.log" 2>/dev/null || \
     grep -q "第 8 轮训练完成" "gan——results/train.log" 2>/dev/null; then
    pkill -f "python3 scripts/train_gan.py" || true
    sleep 3
    ts="$(date +%Y%m%d_%H%M%S)"
    if [ -d "gan——results" ]; then
      mv "gan——results" "gan——results_stage1_stop_${ts}"
    fi
    mkdir -p "gan——results"
    export PATH=/root/miniconda3/bin:$PATH
    nohup env PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      MPLCONFIGDIR="${PROJECT_DIR}/.mplconfig" \
      python3 scripts/train_gan.py \
      --epochs 80 \
      --batch-size 64 \
      --image-size 256 \
      --latent-dim 256 \
      --generator-feature-maps 96 \
      --discriminator-feature-maps 96 \
      --learning-rate 0.00005 \
      --discriminator-steps 1 \
      --grad-clip-norm 5.0 \
      --defect-multiplier 3.0 \
      --sample-interval 10 \
      --output-dir "gan——results" \
      > "gan——results/train.log" 2>&1 < /dev/null &
    echo "已在第7-8轮后切换到残次品强化训练：${ts}" > restart_marker.log
    exit 0
  fi
  sleep 30
done

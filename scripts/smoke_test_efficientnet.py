#!/usr/bin/env python3
"""小规模冒烟测试，验证训练/评估/推理链路可用。"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    """运行命令并在失败时抛出异常。"""
    result = subprocess.run(command, cwd=ROOT_DIR, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main() -> None:
    output_dir = ROOT_DIR / "outputs" / "smoke_efficientnet_b0"
    train_cmd = [
        "python3",
        "scripts/train_efficientnet.py",
        "--model-name",
        "b0",
        "--image-size",
        "160",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--per-class-limit",
        "8",
        "--max-train-batches",
        "2",
        "--max-val-batches",
        "1",
        "--output-dir",
        str(output_dir.relative_to(ROOT_DIR)),
    ]
    eval_cmd = [
        "python3",
        "scripts/evaluate_efficientnet.py",
        "--checkpoint",
        str(output_dir / "best_model.pt"),
        "--split",
        "test",
        "--max-batches",
        "1",
    ]
    sample_image = next((ROOT_DIR / "data" / "casting_data" / "casting_data" / "test").rglob("*.jpeg"))
    infer_cmd = [
        "python3",
        "scripts/infer_efficientnet.py",
        "--checkpoint",
        str(output_dir / "best_model.pt"),
        "--image",
        str(sample_image),
    ]
    run_command(train_cmd)
    run_command(eval_cmd)
    run_command(infer_cmd)


if __name__ == "__main__":
    main()

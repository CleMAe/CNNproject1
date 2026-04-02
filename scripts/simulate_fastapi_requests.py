#!/usr/bin/env python3
"""模拟 FastAPI 接口访问并绘制中文运行曲线。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import httpx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.datasets import load_metadata  # noqa: E402


def configure_plot_style() -> None:
    """设置中文绘图风格。"""
    font_candidates = [
        "Songti SC",
        "Hiragino Sans GB",
        "Heiti TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
    ]
    selected_font_name = "DejaVu Sans"
    for candidate in font_candidates:
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
            font_manager.fontManager.addfont(path)
            selected_font_name = font_manager.FontProperties(fname=path).get_name()
            break
        except Exception:  # noqa: BLE001
            continue
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = selected_font_name
    plt.rcParams["font.sans-serif"] = [selected_font_name]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="模拟 FastAPI 接口访问程序")
    parser.add_argument("--base-url", default="http://127.0.0.1:9053", help="FastAPI 服务地址")
    parser.add_argument("--requests-per-minute", type=int, default=20, help="每分钟请求数")
    parser.add_argument("--duration-minutes", type=int, default=10, help="持续运行分钟数")
    parser.add_argument("--output-dir", default="web_outputs/load_test", help="压测结果输出目录")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="单次请求超时时间")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def choose_sample_images(total_requests: int, seed: int) -> list[Path]:
    """从测试集中随机抽取用于压测的图片。"""
    random.seed(seed)
    metadata = load_metadata(drop_cross_split_duplicates=True)
    test_df = metadata[metadata["split"] == "测试集"].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError("测试集为空，无法执行接口模拟访问")

    sampled_paths = random.choices(test_df["file_path"].tolist(), k=total_requests)
    return [ROOT_DIR / relative_path for relative_path in sampled_paths]


def save_curve(history_df: pd.DataFrame, output_dir: Path) -> Path:
    """绘制接口运行曲线。"""
    configure_plot_style()
    minute_summary = (
        history_df.groupby("minute_index")
        .agg(
            请求数=("request_id", "count"),
            成功率=("is_success", "mean"),
            平均响应时间毫秒=("elapsed_ms", "mean"),
            P95响应时间毫秒=("elapsed_ms", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].plot(minute_summary["minute_index"], minute_summary["请求数"], marker="o", label="每分钟请求数")
    axes[0].plot(minute_summary["minute_index"], minute_summary["成功率"] * 100, marker="s", label="成功率(%)")
    axes[0].set_title("接口请求成功情况曲线")
    axes[0].set_xlabel("分钟序号")
    axes[0].set_ylabel("数值")
    axes[0].legend()

    axes[1].plot(minute_summary["minute_index"], minute_summary["平均响应时间毫秒"], marker="o", label="平均响应时间")
    axes[1].plot(minute_summary["minute_index"], minute_summary["P95响应时间毫秒"], marker="s", label="P95响应时间")
    axes[1].set_title("接口响应耗时曲线")
    axes[1].set_xlabel("分钟序号")
    axes[1].set_ylabel("毫秒")
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / "接口运行曲线.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    """执行接口模拟访问。"""
    args = parse_args()
    total_requests = args.requests_per_minute * args.duration_minutes
    interval_seconds = 60.0 / args.requests_per_minute
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_images = choose_sample_images(total_requests=total_requests, seed=args.seed)
    records: list[dict] = []
    predict_url = args.base_url.rstrip("/") + "/predict"

    with httpx.Client(timeout=args.request_timeout) as client:
        start_time = time.perf_counter()
        for request_index, image_path in enumerate(sample_images, start=1):
            target_time = start_time + (request_index - 1) * interval_seconds
            sleep_seconds = target_time - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            request_begin = time.perf_counter()
            try:
                with image_path.open("rb") as image_file:
                    response = client.post(
                        predict_url,
                        files={"file": (image_path.name, image_file, "image/jpeg")},
                    )
                request_elapsed_ms = (time.perf_counter() - request_begin) * 1000
                response_json = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                is_success = response.status_code == 200
                predicted_label = response_json.get("predicted_label", "")
                confidence = response_json.get("confidence", None)
                error_message = response_json.get("detail", "") if not is_success else ""
            except Exception as exc:  # noqa: BLE001
                request_elapsed_ms = (time.perf_counter() - request_begin) * 1000
                is_success = False
                predicted_label = ""
                confidence = None
                error_message = str(exc)
                response = None

            elapsed_total_seconds = time.perf_counter() - start_time
            minute_index = min(math.ceil(elapsed_total_seconds / 60.0), args.duration_minutes)
            minute_index = max(minute_index, 1)
            records.append(
                {
                    "request_id": request_index,
                    "image_path": str(image_path),
                    "status_code": response.status_code if response is not None else -1,
                    "is_success": int(is_success),
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "elapsed_ms": round(request_elapsed_ms, 3),
                    "minute_index": minute_index,
                    "error_message": error_message,
                }
            )
            print(
                f"第 {request_index}/{total_requests} 个请求完成 | "
                f"状态码: {records[-1]['status_code']} | "
                f"耗时: {records[-1]['elapsed_ms']:.3f} ms | "
                f"预测类别: {predicted_label or '失败'}"
            )

    history_df = pd.DataFrame(records)
    csv_path = output_dir / "接口请求明细.csv"
    history_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    curve_path = save_curve(history_df=history_df, output_dir=output_dir)

    summary = {
        "总请求数": int(len(history_df)),
        "成功请求数": int(history_df["is_success"].sum()),
        "成功率": round(float(history_df["is_success"].mean() * 100), 3),
        "平均响应时间毫秒": round(float(history_df["elapsed_ms"].mean()), 3),
        "P95响应时间毫秒": round(float(history_df["elapsed_ms"].quantile(0.95)), 3),
        "请求明细文件": str(csv_path),
        "运行曲线文件": str(curve_path),
    }
    summary_path = output_dir / "接口压测摘要.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("接口访问模拟完成：")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

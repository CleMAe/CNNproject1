"""FastAPI Web 应用入口。"""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from cnnproject1.inference_service import EfficientNetInferenceService  # noqa: E402


service: EfficientNetInferenceService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """在服务启动时加载模型。"""
    global service
    checkpoint_path = os.getenv("MODEL_CHECKPOINT")
    service = EfficientNetInferenceService(checkpoint_path=checkpoint_path)
    yield
    service = None


app = FastAPI(
    title="铸件缺陷检测服务",
    version="1.0.0",
    description="基于 EfficientNet 的铸件图像缺陷检测 HTTP 服务",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """返回可视化首页。"""
    return FileResponse(ROOT_DIR / "app" / "templates" / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    """返回服务健康状态。"""
    if service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "模型尚未加载完成"},
        )

    return JSONResponse(
        content={
            "status": "ok",
            "message": "服务运行正常",
            "model_checkpoint": str(service.checkpoint_path),
            "device": str(service.device),
            "model_name": service.config["model_name"],
            "image_size": int(service.config["image_size"]),
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """接收图像文件并返回 JSON 推理结果。"""
    if service is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件上传")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        result = service.predict_from_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"图片解析或推理失败：{exc}") from exc

    return JSONResponse(
        content={
            "request_id": uuid.uuid4().hex,
            "filename": file.filename,
            "predicted_index": result.predicted_index,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence, 6),
            "probabilities": {label: round(prob, 6) for label, prob in result.probabilities.items()},
            "elapsed_ms": round(result.elapsed_ms, 3),
            "original_image_size": {
                "width": result.image_size[0],
                "height": result.image_size[1],
            },
            "model": {
                "name": service.config["model_name"],
                "checkpoint": str(service.checkpoint_path),
                "device": str(service.device),
            },
        }
    )

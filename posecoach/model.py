from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Callable

from .config import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_MODEL_PATH,
    MIN_MODEL_BYTES,
    MODEL_URL,
    EXPECTED_MODEL_SHA256,
)


class ModelError(RuntimeError):
    pass


def model_is_ready(model_path: Path = DEFAULT_MODEL_PATH) -> bool:
    return model_path.is_file() and model_path.stat().st_size >= MIN_MODEL_BYTES


def ensure_pose_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if model_is_ready(model_path):
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".download")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "PoseCoach/1.0 MediaPipe model downloader"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            total_bytes = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            with temporary_path.open("wb") as destination:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    destination.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_bytes:
                        progress_callback(
                            min(downloaded / total_bytes, 1.0),
                            "正在下载姿态识别模型",
                        )
        if temporary_path.stat().st_size < MIN_MODEL_BYTES:
            raise ModelError("模型文件大小异常，请检查网络后重试。")
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        if digest != EXPECTED_MODEL_SHA256:
            raise ModelError("模型文件完整性校验失败，请重新下载。")
        os.replace(temporary_path, model_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise ModelError(
            "姿态识别模型未就绪。请联网重试，或手动将 "
            f"pose_landmarker_lite.task 放到 {model_path.parent}。"
        ) from exc
    return model_path


def create_pose_landmarker(
    model_path: Path = DEFAULT_MODEL_PATH,
    detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
):
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ModelError("未找到 MediaPipe，请先按 requirements.txt 安装依赖。") from exc

    if not model_is_ready(model_path):
        raise ModelError(f"模型文件不存在或已损坏：{model_path}")

    # MediaPipe's native layer can fail on non-ASCII Windows paths. Passing the
    # model bytes avoids that limitation and also works when the project sits
    # under a folder such as this repository's Chinese path.
    model_bytes = model_path.read_bytes()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(detection_confidence),
        min_pose_presence_confidence=float(detection_confidence),
        min_tracking_confidence=float(detection_confidence),
    )
    try:
        return mp, vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        raise ModelError(
            "MediaPipe 模型加载失败。请确认模型与当前 MediaPipe 版本兼容。"
        ) from exc

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import (
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_MAX_DURATION_S,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_VISIBILITY_THRESHOLD,
)
from .data_types import AnalysisResult, FrameObservation, VideoMetadata
from .metrics import (
    detect_contact_events,
    make_observation,
    summarize_observations,
)
from .model import ModelError, create_pose_landmarker, ensure_pose_model
from .recommendations import generate_recommendations
from .render import draw_pose_overlay


ProgressCallback = Callable[[float, str], None]


class AnalysisError(RuntimeError):
    pass


def analyze_video(
    input_path: Path,
    output_dir: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    analysis_fps: int = DEFAULT_ANALYSIS_FPS,
    detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    max_duration_s: int = DEFAULT_MAX_DURATION_S,
    max_width: int = DEFAULT_MAX_WIDTH,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    target_cadence: tuple[float, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress_callback, 0.0, "正在准备分析")

    if not input_path.is_file():
        raise AnalysisError("上传的视频文件不存在。")
    if analysis_fps < 4:
        raise AnalysisError("分析帧率过低，至少需要 4 帧/秒。")

    def model_progress(ratio: float, message: str) -> None:
        _report(progress_callback, ratio * 0.08, message)

    try:
        ensure_pose_model(model_path, progress_callback=model_progress)
    except ModelError as exc:
        raise AnalysisError(str(exc)) from exc

    capture = cv2.VideoCapture(str(input_path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not capture.isOpened():
        capture.release()
        raise AnalysisError(
            "OpenCV 无法读取该视频。请转换为 H.264 编码的 MP4 后重试。"
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not math.isfinite(source_fps) or not 1.0 <= source_fps <= 240.0:
        source_fps = 30.0
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration_s = (
        source_frame_count / source_fps if source_frame_count > 0 else 0.0
    )
    frame_step = max(1, int(math.ceil(source_fps / max(1, analysis_fps))))
    effective_analysis_fps = source_fps / frame_step
    output_fps = effective_analysis_fps
    estimated_source_frames = min(
        source_frame_count or int(source_fps * max_duration_s),
        int(source_fps * max_duration_s) + 1,
    )

    _report(progress_callback, 0.09, "正在加载姿态识别模型")
    landmarker = None
    writer = None
    writer_path: Path | None = None
    observations: list[FrameObservation] = []
    width = height = 0
    processed_source_frames = 0
    last_timestamp_ms = -1
    last_analysis_time = 0.0
    first_analysis_time: float | None = None
    truncated = False
    completed = False

    try:
        mediapipe, landmarker = create_pose_landmarker(
            model_path, detection_confidence
        )
        _report(progress_callback, 0.12, "正在逐帧识别跑步姿态")

        frame_index = -1
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            frame_index += 1
            processed_source_frames = frame_index + 1
            timestamp_s = frame_index / source_fps
            if timestamp_s > max_duration_s:
                truncated = True
                break
            if frame_index % frame_step:
                continue

            retrieved, frame = capture.retrieve()
            if not retrieved or frame is None:
                continue
            frame = _resize_to_width(frame, max_width)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            media_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB,
                data=rgb,
            )
            timestamp_ms = int(round(timestamp_s * 1000))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(media_image, timestamp_ms)
            landmark_array = _first_pose_landmarks(result)

            observation = make_observation(
                timestamp_s=timestamp_s,
                landmarks=landmark_array,
                width=width,
                height=height,
                visibility_threshold=visibility_threshold,
            )
            observations.append(observation)
            if first_analysis_time is None:
                first_analysis_time = timestamp_s
            last_analysis_time = timestamp_s

            if writer is None:
                writer, writer_path = _open_browser_video_writer(
                    output_dir, output_fps, (width, height)
                )
            if writer is None or writer_path is None:
                raise AnalysisError("无法创建结果视频，请检查磁盘空间和写入权限。")

            rendered = draw_pose_overlay(
                frame,
                landmark_array,
                observation.left_knee_deg,
                observation.torso_lean_deg,
                visibility_threshold,
            )
            writer.write(rendered)

            if estimated_source_frames > 0:
                ratio = min(processed_source_frames / estimated_source_frames, 1.0)
                _report(
                    progress_callback,
                    0.12 + ratio * 0.76,
                    f"正在分析 {timestamp_s:.1f} 秒 / 已识别 {len(observations)} 帧",
                )

        if not observations or writer is None or writer_path is None:
            raise AnalysisError("视频中没有可读取的画面。")

        _report(progress_callback, 0.91, "正在计算步频和关节指标")
        contacts = detect_contact_events(observations, visibility_threshold)
        analyzed_duration_s = max(
            0.0,
            last_analysis_time
            - (first_analysis_time if first_analysis_time is not None else 0.0)
            + 1.0 / max(effective_analysis_fps, 1.0),
        )
        metrics, quality = summarize_observations(
            observations, contacts, analyzed_duration_s
        )
        recommendations = generate_recommendations(
            metrics, quality, target_cadence=target_cadence
        )
        metadata = VideoMetadata(
            source_fps=round(source_fps, 3),
            analysis_fps=round(effective_analysis_fps, 3),
            output_fps=round(output_fps, 3),
            width=width,
            height=height,
            source_frame_count=source_frame_count,
            analyzed_frame_count=len(observations),
            source_duration_s=round(source_duration_s, 3),
            analyzed_duration_s=round(analyzed_duration_s, 3),
            truncated=truncated,
        )
        _report(progress_callback, 1.0, "分析完成")
        result = AnalysisResult(
            run_id=output_dir.name,
            input_path=input_path,
            output_video_path=writer_path,
            metadata=metadata,
            observations=observations,
            contacts=contacts,
            metrics=metrics,
            quality=quality,
            recommendations=recommendations,
        )
        completed = True
        return result
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"分析过程中出现错误：{exc}") from exc
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if landmarker is not None:
            landmarker.close()
        if not completed and writer_path is not None:
            writer_path.unlink(missing_ok=True)


def _first_pose_landmarks(result) -> np.ndarray | None:
    pose_sets = getattr(result, "pose_landmarks", None)
    if not pose_sets or not pose_sets[0]:
        return None
    landmarks = pose_sets[0]
    return np.asarray(
        [
            [
                float(item.x),
                float(item.y),
                float(item.z),
                float(getattr(item, "visibility", 1.0)),
                float(getattr(item, "presence", 1.0)),
            ]
            for item in landmarks
        ],
        dtype=np.float32,
    )


def _resize_to_width(frame: np.ndarray, maximum_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= maximum_width:
        return frame
    scale = maximum_width / float(width)
    return cv2.resize(
        frame,
        (maximum_width, max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _open_browser_video_writer(
    output_dir: Path,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter | None, Path | None]:
    webm_path = output_dir / "pose_overlay.webm"
    for codec in ("VP80", "VP90"):
        writer = cv2.VideoWriter(
            str(webm_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer, webm_path
        writer.release()
        webm_path.unlink(missing_ok=True)

    mp4_path = output_dir / "pose_overlay.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if writer.isOpened():
        return writer, mp4_path
    writer.release()
    mp4_path.unlink(missing_ok=True)
    return None, None


def _report(
    callback: ProgressCallback | None,
    ratio: float,
    message: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(ratio, 0.0, 1.0)), message)

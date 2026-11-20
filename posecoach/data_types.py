from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameObservation:
    timestamp_s: float
    landmarks: np.ndarray | None = None
    left_knee_deg: float | None = None
    right_knee_deg: float | None = None
    left_hip_deg: float | None = None
    right_hip_deg: float | None = None
    torso_lean_deg: float | None = None
    left_foot_vertical: float | None = None
    right_foot_vertical: float | None = None
    mean_visibility: float | None = None


@dataclass(slots=True)
class VideoMetadata:
    source_fps: float
    analysis_fps: float
    output_fps: float
    width: int
    height: int
    source_frame_count: int
    analyzed_frame_count: int
    source_duration_s: float
    analyzed_duration_s: float
    truncated: bool


@dataclass(slots=True)
class ContactEvent:
    timestamp_s: float
    side: str
    strike_type: str
    strength: float


@dataclass(slots=True)
class QualitySummary:
    level: str
    label: str
    score: int
    pose_coverage: float
    mean_visibility: float
    message: str


@dataclass(slots=True)
class Recommendation:
    kind: str
    title: str
    body: str


@dataclass(slots=True)
class AnalysisResult:
    run_id: str
    input_path: Path
    output_video_path: Path
    metadata: VideoMetadata
    observations: list[FrameObservation]
    contacts: list[ContactEvent]
    metrics: dict[str, Any]
    quality: QualitySummary
    recommendations: list[Recommendation] = field(default_factory=list)

    def time_series_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for observation in self.observations:
            rows.append(
                {
                    "时间(s)": round(observation.timestamp_s, 3),
                    "左膝角度(°)": _rounded(observation.left_knee_deg),
                    "右膝角度(°)": _rounded(observation.right_knee_deg),
                    "左髋角度(°)": _rounded(observation.left_hip_deg),
                    "右髋角度(°)": _rounded(observation.right_hip_deg),
                    "躯干前倾(°)": _rounded(observation.torso_lean_deg),
                }
            )
        return rows

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video": {
                "input_path": str(self.input_path),
                "output_path": str(self.output_video_path),
                **asdict(self.metadata),
            },
            "metrics": _json_safe(self.metrics),
            "quality": asdict(self.quality),
            "contacts": [asdict(contact) for contact in self.contacts],
            "recommendations": [asdict(item) for item in self.recommendations],
            "notices": [
                "本报告用于运动教学和比赛原型演示，不构成医疗诊断。",
                "单相机二维关键点结果会受机位、遮挡、服装和拍摄帧率影响。",
            ],
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

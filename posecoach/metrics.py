from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .config import (
    MAX_CADENCE_SPM,
    MIN_CADENCE_SPM,
    MIN_RELIABLE_DURATION_S,
)
from .data_types import (
    ContactEvent,
    FrameObservation,
    QualitySummary,
)
from .geometry import (
    classify_strike_points,
    joint_angle_deg,
    relative_foot_height,
    torso_lean_deg,
)
from .landmarks import (
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    KEY_MEASUREMENT_LANDMARKS,
)


def make_observation(
    timestamp_s: float,
    landmarks: np.ndarray | None,
    width: int,
    height: int,
    visibility_threshold: float,
) -> FrameObservation:
    if landmarks is None or landmarks.shape[0] < 33:
        return FrameObservation(timestamp_s=timestamp_s)

    left_knee = joint_angle_deg(
        landmarks,
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
        width,
        height,
        visibility_threshold,
    )
    right_knee = joint_angle_deg(
        landmarks,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
        width,
        height,
        visibility_threshold,
    )
    left_hip = joint_angle_deg(
        landmarks,
        LEFT_SHOULDER,
        LEFT_HIP,
        LEFT_KNEE,
        width,
        height,
        visibility_threshold,
    )
    right_hip = joint_angle_deg(
        landmarks,
        RIGHT_SHOULDER,
        RIGHT_HIP,
        RIGHT_KNEE,
        width,
        height,
        visibility_threshold,
    )
    torso_lean, body_scale_px = torso_lean_deg(
        landmarks,
        width,
        height,
        visibility_threshold,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    )
    left_foot_vertical = relative_foot_height(
        landmarks,
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
        width,
        height,
        visibility_threshold,
        body_scale_px,
    )
    right_foot_vertical = relative_foot_height(
        landmarks,
        (LEFT_HIP, RIGHT_HIP),
        (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX),
        width,
        height,
        visibility_threshold,
        body_scale_px,
    )

    scores = [
        float(max(landmarks[index, 3], landmarks[index, 4]))
        for index in KEY_MEASUREMENT_LANDMARKS
        if landmarks.shape[1] > 4
    ]
    return FrameObservation(
        timestamp_s=timestamp_s,
        landmarks=landmarks,
        left_knee_deg=left_knee,
        right_knee_deg=right_knee,
        left_hip_deg=left_hip,
        right_hip_deg=right_hip,
        torso_lean_deg=torso_lean,
        left_foot_vertical=left_foot_vertical,
        right_foot_vertical=right_foot_vertical,
        mean_visibility=float(np.mean(scores)) if scores else None,
    )


def detect_contact_events(
    observations: list[FrameObservation],
    visibility_threshold: float = 0.45,
) -> list[ContactEvent]:
    if len(observations) < 6:
        return []

    fps = _effective_fps(observations)
    times = np.array([item.timestamp_s for item in observations], dtype=float)
    left_values = np.array(
        [np.nan if item.left_foot_vertical is None else item.left_foot_vertical for item in observations],
        dtype=float,
    )
    right_values = np.array(
        [np.nan if item.right_foot_vertical is None else item.right_foot_vertical for item in observations],
        dtype=float,
    )

    candidates: list[tuple[float, str, int, float]] = []
    candidates.extend(_side_peaks(times, left_values, "left", fps))
    candidates.extend(_side_peaks(times, right_values, "right", fps))
    candidates.sort(key=lambda item: item[0])

    merged: list[tuple[float, str, int, float]] = []
    for candidate in candidates:
        if merged and candidate[0] - merged[-1][0] < 0.16:
            if candidate[3] > merged[-1][3]:
                merged[-1] = candidate
            continue
        merged.append(candidate)

    spaced: list[tuple[float, str, int, float]] = []
    for candidate in merged:
        if spaced and candidate[0] - spaced[-1][0] < 0.20:
            if candidate[3] > spaced[-1][3]:
                spaced[-1] = candidate
            continue
        spaced.append(candidate)

    events: list[ContactEvent] = []
    for timestamp, side, index, strength in spaced:
        strike_type = classify_strike(
            observations[index].landmarks,
            side,
            width=0,
            height=0,
            visibility_threshold=visibility_threshold,
            precomputed_normalized=True,
        )
        events.append(
            ContactEvent(
                timestamp_s=float(timestamp),
                side=side,
                strike_type=strike_type,
                strength=float(strength),
            )
        )
    return events


def _side_peaks(
    times: np.ndarray,
    values: np.ndarray,
    side: str,
    fps: float,
) -> list[tuple[float, str, int, float]]:
    valid = np.isfinite(values)
    if valid.sum() < max(6, int(fps * 1.5)):
        return []

    interpolated = np.interp(
        np.arange(len(values)), np.flatnonzero(valid), values[valid]
    )
    smoothing_radius = max(1, int(round(fps * 0.07)))
    kernel = np.ones(smoothing_radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(interpolated, kernel, mode="same")

    p10, p40, p90 = np.percentile(smooth, [10, 40, 90])
    dynamic_range = float(p90 - p10)
    if dynamic_range < 0.012:
        return []

    local_radius = max(2, int(round(fps * 0.12)))
    raw_candidates: list[tuple[float, int, float]] = []
    for index in range(local_radius, len(smooth) - local_radius):
        window = smooth[index - local_radius : index + local_radius + 1]
        if smooth[index] < p40 or smooth[index] < float(window.max()) - 1e-9:
            continue
        raw_candidates.append((float(times[index]), index, float(smooth[index])))

    grouped: list[tuple[float, int, float]] = []
    for candidate in raw_candidates:
        if grouped and candidate[0] - grouped[-1][0] < 0.38:
            if candidate[2] > grouped[-1][2]:
                grouped[-1] = candidate
            continue
        grouped.append(candidate)

    half_span = max(3, int(round(fps * 0.45)))
    minimum_prominence = max(0.015, dynamic_range * 0.12)
    accepted: list[tuple[float, str, int, float]] = []
    for _, index, peak_value in grouped:
        before = smooth[max(0, index - half_span) : index]
        after = smooth[index + 1 : min(len(smooth), index + half_span + 1)]
        if not len(before) or not len(after):
            continue
        prominence = peak_value - max(float(before.min()), float(after.min()))
        if prominence < minimum_prominence:
            continue
        accepted.append((float(times[index]), side, index, float(prominence)))
    return accepted


def estimate_cadence(events: list[ContactEvent]) -> tuple[float | None, str]:
    if len(events) < 4:
        return None, "识别到的触地次数不足"

    side_times: dict[str, list[float]] = defaultdict(list)
    for event in events:
        side_times[event.side].append(event.timestamp_s)

    stride_intervals: list[float] = []
    for timestamps in side_times.values():
        ordered = sorted(timestamps)
        stride_intervals.extend(
            later - earlier
            for earlier, later in zip(ordered, ordered[1:])
            if 0.40 <= later - earlier <= 1.40
        )

    cadence: float | None = None
    source = ""
    if len(stride_intervals) >= 2:
        stride_s = float(np.median(stride_intervals))
        cadence = 120.0 / stride_s
        source = "双侧步幅周期"
    else:
        ordered_times = sorted(event.timestamp_s for event in events)
        step_intervals = [
            later - earlier
            for earlier, later in zip(ordered_times, ordered_times[1:])
            if 0.22 <= later - earlier <= 0.90
        ]
        if len(step_intervals) >= 3:
            cadence = 60.0 / float(np.median(step_intervals))
            source = "相邻触地间隔"

    if cadence is None:
        return None, "触地节律不稳定"
    if not MIN_CADENCE_SPM <= cadence <= MAX_CADENCE_SPM:
        return None, "步频超出可解释范围"
    return float(cadence), source


def classify_strike(
    landmarks: np.ndarray | None,
    side: str,
    width: int,
    height: int,
    visibility_threshold: float,
    precomputed_normalized: bool = False,
) -> str:
    if landmarks is None or landmarks.shape[0] < 33:
        return "unknown"
    if side == "left":
        indices = (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX)
    else:
        indices = (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX)
    if precomputed_normalized:
        width = 1
        height = 1
    return classify_strike_points(
        landmarks[indices[0]],
        landmarks[indices[1]],
        landmarks[indices[2]],
        width,
        height,
        visibility_threshold,
    )


def summarize_observations(
    observations: list[FrameObservation],
    contacts: list[ContactEvent],
    analyzed_duration_s: float,
) -> tuple[dict, QualitySummary]:
    detected = [item for item in observations if item.landmarks is not None]
    coverage = len(detected) / max(1, len(observations))
    visibility_values = [
        item.mean_visibility for item in detected if item.mean_visibility is not None
    ]
    mean_visibility = (
        float(np.mean(visibility_values)) if visibility_values else 0.0
    )
    quality_score = int(
        round(np.clip(coverage * 70.0 + mean_visibility * 30.0, 0, 100))
    )

    if not detected or analyzed_duration_s < 1.0:
        level, label, message = (
            "failed",
            "未识别到完整人体",
            "请确认视频中人物全身入镜，并重新拍摄侧面视频。",
        )
    elif analyzed_duration_s < MIN_RELIABLE_DURATION_S or coverage < 0.35:
        level, label, message = (
            "low",
            "数据可信度较低",
            "有效姿态帧偏少，建议检查全身入镜、光线和遮挡后重拍。",
        )
    elif coverage < 0.65 or mean_visibility < 0.62:
        level, label, message = (
            "medium",
            "数据可信度中等",
            "主要趋势可参考，角度和着地方式请结合原视频复核。",
        )
    else:
        level, label, message = (
            "high",
            "数据可信度较高",
            "关键关节持续可见，可作为本次跑姿复盘的参考。",
        )

    knees = [
        value
        for item in detected
        for value in (item.left_knee_deg, item.right_knee_deg)
        if value is not None
    ]
    left_knees = [
        item.left_knee_deg for item in detected if item.left_knee_deg is not None
    ]
    right_knees = [
        item.right_knee_deg for item in detected if item.right_knee_deg is not None
    ]
    hips = [
        value
        for item in detected
        for value in (item.left_hip_deg, item.right_hip_deg)
        if value is not None
    ]
    torso = [
        item.torso_lean_deg
        for item in detected
        if item.torso_lean_deg is not None
    ]
    contact_knees = _contact_knee_angles(observations, contacts)

    cadence, cadence_source = estimate_cadence(contacts)
    valid_strikes = [item.strike_type for item in contacts if item.strike_type != "unknown"]
    strike_counts = Counter(valid_strikes)
    strike_distribution = {
        key: round(value / len(valid_strikes) * 100, 1)
        for key, value in strike_counts.items()
    } if valid_strikes else {}
    dominant_strike = (
        strike_counts.most_common(1)[0][0] if strike_counts else "unknown"
    )

    knee_stats = _stats(knees)
    hip_stats = _stats(hips)
    torso_stats = _stats(torso)
    metrics = {
        "cadence_spm": _round(cadence),
        "cadence_source": cadence_source,
        "contact_count": len(contacts),
        "contact_knee_mean_deg": _round(_mean(contact_knees)),
        "contact_knee_min_deg": _round(_safe_min(contact_knees)),
        "contact_knee_max_deg": _round(_safe_max(contact_knees)),
        "knee_mean_deg": _round(knee_stats["mean"]),
        "knee_min_deg": _round(knee_stats["min"]),
        "knee_max_deg": _round(knee_stats["max"]),
        "knee_p10_deg": _round(knee_stats["p10"]),
        "knee_p90_deg": _round(knee_stats["p90"]),
        "left_knee_mean_deg": _round(_mean(left_knees)),
        "right_knee_mean_deg": _round(_mean(right_knees)),
        "hip_mean_deg": _round(hip_stats["mean"]),
        "hip_min_deg": _round(hip_stats["min"]),
        "hip_max_deg": _round(hip_stats["max"]),
        "torso_lean_mean_deg": _round(torso_stats["mean"]),
        "torso_lean_p10_deg": _round(torso_stats["p10"]),
        "torso_lean_p90_deg": _round(torso_stats["p90"]),
        "strike_distribution_pct": strike_distribution,
        "dominant_strike": dominant_strike,
        "pose_coverage_pct": round(coverage * 100, 1),
        "mean_visibility_pct": round(mean_visibility * 100, 1),
        "analyzed_frames": len(observations),
        "frames_with_pose": len(detected),
    }
    quality = QualitySummary(
        level=level,
        label=label,
        score=quality_score,
        pose_coverage=coverage,
        mean_visibility=mean_visibility,
        message=message,
    )
    return metrics, quality


def _effective_fps(observations: list[FrameObservation]) -> float:
    if len(observations) < 2:
        return 12.0
    differences = np.diff([item.timestamp_s for item in observations])
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if not len(differences):
        return 12.0
    return float(np.clip(1.0 / np.median(differences), 1.0, 60.0))


def _contact_knee_angles(
    observations: list[FrameObservation],
    contacts: list[ContactEvent],
) -> list[float]:
    if not observations:
        return []
    times = np.array([item.timestamp_s for item in observations])
    values: list[float] = []
    for contact in contacts:
        index = int(np.abs(times - contact.timestamp_s).argmin())
        item = observations[index]
        angle = item.left_knee_deg if contact.side == "left" else item.right_knee_deg
        if angle is not None:
            values.append(angle)
    return values


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_min(values: list[float]) -> float | None:
    return float(np.min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 1)

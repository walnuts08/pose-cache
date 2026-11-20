from __future__ import annotations

import math

import numpy as np


def visibility_score(landmark: np.ndarray) -> float:
    if landmark.shape[0] > 4:
        return float(max(landmark[3], landmark[4]))
    if landmark.shape[0] > 3:
        return float(landmark[3])
    return 1.0


def pixel_point(
    landmark: np.ndarray,
    width: int,
    height: int,
    visibility_threshold: float,
) -> np.ndarray | None:
    if visibility_score(landmark) < visibility_threshold:
        return None
    return np.array([landmark[0] * width, landmark[1] * height], dtype=np.float64)


def visible_mean_landmark(
    landmarks: np.ndarray,
    indices: tuple[int, ...],
    visibility_threshold: float,
) -> np.ndarray | None:
    points = [
        landmarks[index]
        for index in indices
        if visibility_score(landmarks[index]) >= visibility_threshold
    ]
    if len(points) != len(indices):
        return None
    return np.mean(np.stack(points), axis=0)


def joint_angle_deg(
    landmarks: np.ndarray,
    first_index: int,
    middle_index: int,
    last_index: int,
    width: int,
    height: int,
    visibility_threshold: float,
) -> float | None:
    first = pixel_point(
        landmarks[first_index], width, height, visibility_threshold
    )
    middle = pixel_point(
        landmarks[middle_index], width, height, visibility_threshold
    )
    last = pixel_point(landmarks[last_index], width, height, visibility_threshold)
    if first is None or middle is None or last is None:
        return None

    vector_a = first - middle
    vector_b = last - middle
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator < 1e-6:
        return None
    cosine = float(np.dot(vector_a, vector_b) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def torso_lean_deg(
    landmarks: np.ndarray,
    width: int,
    height: int,
    visibility_threshold: float,
    left_shoulder: int,
    right_shoulder: int,
    left_hip: int,
    right_hip: int,
) -> tuple[float | None, float | None]:
    shoulders = visible_mean_landmark(
        landmarks, (left_shoulder, right_shoulder), visibility_threshold
    )
    hips = visible_mean_landmark(
        landmarks, (left_hip, right_hip), visibility_threshold
    )
    if shoulders is None or hips is None:
        return None, None

    shoulder_point = np.array(
        [shoulders[0] * width, shoulders[1] * height], dtype=np.float64
    )
    hip_point = np.array([hips[0] * width, hips[1] * height], dtype=np.float64)
    trunk = shoulder_point - hip_point
    vertical = np.array([0.0, -1.0])
    norm = float(np.linalg.norm(trunk))
    if norm < 1e-6:
        return None, None
    cross_product = (
        vertical[0] * trunk[1] - vertical[1] * trunk[0]
    )
    sine = float(np.clip(cross_product / norm, -1.0, 1.0))
    lean = abs(math.degrees(math.asin(sine)))
    return lean, norm


def classify_strike_points(
    ankle: np.ndarray,
    heel: np.ndarray,
    toe: np.ndarray,
    width: int,
    height: int,
    visibility_threshold: float,
) -> str:
    ankle_point = pixel_point(ankle, width, height, visibility_threshold)
    heel_point = pixel_point(heel, width, height, visibility_threshold)
    toe_point = pixel_point(toe, width, height, visibility_threshold)
    if ankle_point is None or heel_point is None or toe_point is None:
        return "unknown"

    foot_length = float(np.linalg.norm(toe_point - ankle_point))
    if foot_length < 1e-6:
        return "unknown"

    heel_minus_toe = (heel_point[1] - toe_point[1]) / foot_length
    if heel_minus_toe > 0.10:
        return "rearfoot"
    if heel_minus_toe < -0.10:
        return "forefoot"
    return "midfoot"


def relative_foot_height(
    landmarks: np.ndarray,
    hip_indices: tuple[int, int],
    foot_indices: tuple[int, int, int],
    width: int,
    height: int,
    visibility_threshold: float,
    body_scale_px: float | None,
) -> float | None:
    if not body_scale_px or body_scale_px < 1e-6:
        return None
    hips = visible_mean_landmark(landmarks, hip_indices, visibility_threshold)
    if hips is None:
        return None

    visible_feet = [
        landmarks[index]
        for index in foot_indices
        if visibility_score(landmarks[index]) >= visibility_threshold
    ]
    if not visible_feet:
        return None

    lowest_foot_y = max(float(item[1]) for item in visible_feet) * height
    hip_y = float(hips[1]) * height
    return (lowest_foot_y - hip_y) / body_scale_px

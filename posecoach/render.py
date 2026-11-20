from __future__ import annotations

import cv2
import numpy as np

from .landmarks import (
    LEFT_ARM_CONNECTIONS,
    LEFT_LEG_CONNECTIONS,
    RIGHT_ARM_CONNECTIONS,
    RIGHT_LEG_CONNECTIONS,
    TORSO_CONNECTIONS,
)


TORSO_COLOR = (116, 214, 174)
LEFT_COLOR = (93, 212, 255)
RIGHT_COLOR = (255, 178, 104)
JOINT_COLOR = (244, 247, 238)
TEXT_COLOR = (240, 248, 244)
HUD_COLOR = (20, 45, 42)


def draw_pose_overlay(
    frame: np.ndarray,
    landmarks: np.ndarray | None,
    left_knee_deg: float | None,
    torso_lean_deg: float | None,
    visibility_threshold: float,
) -> np.ndarray:
    output = frame.copy()
    if landmarks is None:
        _draw_status(output, "NO POSE", (64, 77, 235))
        return output

    height, width = output.shape[:2]
    points = [
        (
            int(round(float(landmark[0]) * width)),
            int(round(float(landmark[1]) * height)),
        )
        for landmark in landmarks
    ]
    visibility = [
        float(max(landmark[3], landmark[4])) for landmark in landmarks
    ]

    line_groups = (
        (TORSO_CONNECTIONS, TORSO_COLOR, 5),
        (LEFT_ARM_CONNECTIONS, LEFT_COLOR, 3),
        (RIGHT_ARM_CONNECTIONS, RIGHT_COLOR, 3),
        (LEFT_LEG_CONNECTIONS, LEFT_COLOR, 5),
        (RIGHT_LEG_CONNECTIONS, RIGHT_COLOR, 5),
    )
    for connections, color, thickness in line_groups:
        for start, end in connections:
            if (
                visibility[start] >= visibility_threshold
                and visibility[end] >= visibility_threshold
            ):
                cv2.line(
                    output,
                    points[start],
                    points[end],
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    for index, (x, y) in enumerate(points):
        if visibility[index] < visibility_threshold:
            continue
        radius = 6 if index in (11, 12, 23, 24, 25, 26, 27, 28) else 4
        cv2.circle(output, (x, y), radius + 2, HUD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), radius, JOINT_COLOR, -1, cv2.LINE_AA)

    if left_knee_deg is not None:
        knee_x, knee_y = points[25]
        cv2.putText(
            output,
            f"KNEE {left_knee_deg:.0f}",
            (min(width - 120, knee_x + 12), max(24, knee_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    if torso_lean_deg is not None:
        _draw_status(output, f"LEAN {torso_lean_deg:.1f} DEG", TORSO_COLOR)
    else:
        _draw_status(output, "POSE TRACKING", TORSO_COLOR)
    return output


def _draw_status(frame: np.ndarray, text: str, accent: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    scale = max(0.48, min(0.72, width / 1200))
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    width_padding = max(12, int(width * 0.015))
    height_padding = max(8, int(height * 0.012))
    x1 = width_padding
    y1 = height_padding
    x2 = min(width - width_padding, x1 + text_width + 38)
    y2 = y1 + text_height + 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), HUD_COLOR, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.circle(
        frame,
        (x1 + 14, y1 + (y2 - y1) // 2),
        5,
        accent,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 27, y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )

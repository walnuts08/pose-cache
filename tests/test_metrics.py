from __future__ import annotations

import math
import unittest

import numpy as np

from posecoach.data_types import FrameObservation
from posecoach.geometry import classify_strike_points, joint_angle_deg
from posecoach.metrics import detect_contact_events, estimate_cadence


class GeometryTests(unittest.TestCase):
    def test_straight_joint_has_180_degree_angle(self) -> None:
        landmarks = np.zeros((33, 5), dtype=np.float32)
        landmarks[:, 3:5] = 1.0
        landmarks[23] = (0.5, 0.3, 0, 1, 1)
        landmarks[25] = (0.5, 0.5, 0, 1, 1)
        landmarks[27] = (0.5, 0.7, 0, 1, 1)
        angle = joint_angle_deg(
            landmarks, 23, 25, 27, width=1080, height=1920, visibility_threshold=0.4
        )
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=3)

    def test_rearfoot_is_detected_when_heel_is_lower(self) -> None:
        ankle = np.array([0.5, 0.60, 0, 1, 1], dtype=np.float32)
        heel = np.array([0.48, 0.66, 0, 1, 1], dtype=np.float32)
        toe = np.array([0.55, 0.64, 0, 1, 1], dtype=np.float32)
        label = classify_strike_points(ankle, heel, toe, 1, 1, 0.4)
        self.assertEqual(label, "rearfoot")


class CadenceTests(unittest.TestCase):
    def test_synthetic_180_spm_sequence(self) -> None:
        cadence = 180.0
        step_interval = 60.0 / cadence
        stride_interval = step_interval * 2.0
        fps = 20
        times = np.arange(0.0, 12.0, 1 / fps)
        left_centers = np.arange(0.5, 12.0, stride_interval)
        right_centers = left_centers + step_interval

        def pulse(centers: np.ndarray) -> np.ndarray:
            values = np.full_like(times, 0.25, dtype=float)
            for center in centers:
                values += 0.35 * np.exp(-0.5 * ((times - center) / 0.055) ** 2)
            return values

        left = pulse(left_centers)
        right = pulse(right_centers)
        observations = [
            FrameObservation(
                timestamp_s=float(timestamp),
                left_foot_vertical=float(left[index]),
                right_foot_vertical=float(right[index]),
            )
            for index, timestamp in enumerate(times)
        ]
        events = detect_contact_events(observations)
        estimated, _ = estimate_cadence(events)
        self.assertIsNotNone(estimated)
        self.assertGreater(len(events), 20)
        self.assertAlmostEqual(estimated, cadence, delta=8.0)
        self.assertTrue(math.isfinite(estimated))


if __name__ == "__main__":
    unittest.main()

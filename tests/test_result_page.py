from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from posecoach.config import PROJECT_ROOT
from posecoach.data_types import (
    AnalysisResult,
    FrameObservation,
    QualitySummary,
    Recommendation,
    VideoMetadata,
)


def _render_result_harness(result) -> None:
    from app import _render_result

    _render_result(result)


class ResultPageTests(unittest.TestCase):
    def test_result_page_renders_all_tabs(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "overlay.webm"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"VP80"),
                5,
                (320, 240),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 210, dtype=np.uint8))
            writer.release()

            result = AnalysisResult(
                run_id="test-run",
                input_path=root / "input.mp4",
                output_video_path=video_path,
                metadata=VideoMetadata(
                    source_fps=15,
                    analysis_fps=10,
                    output_fps=15,
                    width=320,
                    height=240,
                    source_frame_count=45,
                    analyzed_frame_count=2,
                    source_duration_s=3,
                    analyzed_duration_s=2,
                    truncated=False,
                ),
                observations=[
                    FrameObservation(
                        timestamp_s=0.0,
                        left_knee_deg=158,
                        right_knee_deg=151,
                        left_hip_deg=146,
                        right_hip_deg=142,
                        torso_lean_deg=8.4,
                    ),
                    FrameObservation(
                        timestamp_s=0.1,
                        left_knee_deg=160,
                        right_knee_deg=153,
                        left_hip_deg=148,
                        right_hip_deg=144,
                        torso_lean_deg=8.7,
                    ),
                ],
                contacts=[],
                metrics={
                    "cadence_spm": 176.0,
                    "cadence_source": "双侧步幅周期",
                    "contact_knee_mean_deg": 156.0,
                    "left_knee_mean_deg": 159.0,
                    "right_knee_mean_deg": 152.0,
                    "knee_p10_deg": 151.0,
                    "knee_p90_deg": 160.0,
                    "hip_mean_deg": 145.0,
                    "torso_lean_mean_deg": 8.5,
                    "torso_lean_p10_deg": 8.4,
                    "torso_lean_p90_deg": 8.7,
                    "pose_coverage_pct": 91.0,
                    "contact_count": 6,
                    "dominant_strike": "midfoot",
                    "strike_distribution_pct": {"midfoot": 66.7, "rearfoot": 33.3},
                },
                quality=QualitySummary(
                    level="high",
                    label="数据可信度较高",
                    score=91,
                    pose_coverage=0.91,
                    mean_visibility=0.9,
                    message="关键关节持续可见。",
                ),
                recommendations=[
                    Recommendation(
                        kind="technique",
                        title="保持当前节奏",
                        body="用同条件视频追踪两周变化。",
                    )
                ],
            )

            app = AppTest.from_function(
                _render_result_harness,
                kwargs={"result": result},
                default_timeout=10,
            )
            app.run()
            self.assertFalse(app.exception, [item.value for item in app.exception])


if __name__ == "__main__":
    unittest.main()

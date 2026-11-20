from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from posecoach.config import DEFAULT_MODEL_PATH, PROJECT_ROOT
from posecoach.model import model_is_ready
from posecoach.pipeline import analyze_video


@unittest.skipUnless(
    model_is_ready(DEFAULT_MODEL_PATH),
    "MediaPipe 模型尚未下载",
)
class PipelineSmokeTests(unittest.TestCase):
    def test_blank_video_runs_end_to_end(self) -> None:
        # Keep the smoke test inside the Chinese-named project path to cover
        # the same OpenCV/Windows path behavior as the actual application.
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "blank.mp4"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (320, 480),
            )
            self.assertTrue(writer.isOpened())
            frame = np.full((480, 320, 3), 235, dtype=np.uint8)
            for index in range(50):
                cv2.line(
                    frame,
                    (20, 30 + index * 4),
                    (300, 30 + index * 4),
                    (110, 130, 120),
                    1,
                )
                writer.write(frame)
            writer.release()

            result = analyze_video(
                input_path,
                root / "result",
                analysis_fps=5,
                max_duration_s=5,
                max_width=320,
            )
            self.assertTrue(result.output_video_path.is_file())
            self.assertGreater(result.output_video_path.stat().st_size, 0)
            self.assertGreater(result.metadata.analyzed_frame_count, 0)
            self.assertEqual(result.quality.level, "failed")


if __name__ == "__main__":
    unittest.main()

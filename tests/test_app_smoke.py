from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from posecoach.config import PROJECT_ROOT


class AppSmokeTests(unittest.TestCase):
    def test_initial_page_renders_without_exception(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=10)
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(app.file_uploader[0].label, "上传跑步视频")


if __name__ == "__main__":
    unittest.main()

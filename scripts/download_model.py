from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from posecoach.config import DEFAULT_MODEL_PATH  # noqa: E402
from posecoach.model import ModelError, ensure_pose_model  # noqa: E402


def main() -> int:
    try:
        path = ensure_pose_model(
            DEFAULT_MODEL_PATH,
            progress_callback=lambda ratio, message: print(
                f"\r{message}: {ratio * 100:5.1f}%", end="", flush=True
            ),
        )
    except ModelError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n模型已就绪：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

DEFAULT_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MIN_MODEL_BYTES = 1_000_000
EXPECTED_MODEL_SHA256 = (
    "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
)

SUPPORTED_VIDEO_TYPES = ("mp4", "mov", "m4v", "avi", "webm", "mkv")
DEFAULT_ANALYSIS_FPS = 12
DEFAULT_MAX_DURATION_S = 90
DEFAULT_MAX_WIDTH = 1280
DEFAULT_DETECTION_CONFIDENCE = 0.5
DEFAULT_VISIBILITY_THRESHOLD = 0.45

CADENCE_REFERENCE = (160.0, 190.0)
CONTACT_KNEE_REFERENCE = (140.0, 175.0)
TORSO_LEAN_REFERENCE = (3.0, 18.0)

MAX_CADENCE_SPM = 260.0
MIN_CADENCE_SPM = 90.0
MIN_RELIABLE_DURATION_S = 4.0

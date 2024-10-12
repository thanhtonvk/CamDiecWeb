# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.34"

from ultralytics_v10.data.explorer.explorer import Explorer
from ultralytics_v10.models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from ultralytics_v10.models.fastsam import FastSAM
from ultralytics_v10.models.nas import NAS
from ultralytics_v10.utils import ASSETS, SETTINGS as settings
from ultralytics_v10.utils.checks import check_yolo as checks
from ultralytics_v10.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10"
)
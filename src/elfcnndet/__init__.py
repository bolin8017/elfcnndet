"""1D-CNN ELF malware detector with multi-GPU DataParallel."""

from .config import ElfCnnDetectorConfig
from .detector import ElfCnnDetector

__all__ = ["ElfCnnDetector", "ElfCnnDetectorConfig"]
__version__ = "0.1.1"

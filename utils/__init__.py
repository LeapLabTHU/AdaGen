import sys
from pathlib import Path

_UTILS_ROOT = Path(__file__).resolve().parent
if str(_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(_UTILS_ROOT))

__all__ = []

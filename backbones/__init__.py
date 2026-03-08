import sys
from pathlib import Path

_LIBS_ROOT = Path(__file__).resolve().parent / "libs"
if str(_LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBS_ROOT))

__all__ = []

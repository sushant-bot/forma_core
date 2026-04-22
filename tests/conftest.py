import sys
from pathlib import Path


# Make FormaCore modules importable in tests (e.g., "from core.grid import Grid").
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMACORE_ROOT = PROJECT_ROOT / "FormaCore"
if str(FORMACORE_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMACORE_ROOT))

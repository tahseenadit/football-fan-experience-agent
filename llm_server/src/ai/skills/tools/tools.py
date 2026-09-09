import sys
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parents[2]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from skills.simple_skill import simple_test

TOOLS = {
    "simple_test": simple_test,
}
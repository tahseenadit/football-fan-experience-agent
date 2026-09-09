import sys
from pathlib import Path

_AI_SKILLS_DIR = Path(__file__).resolve().parent[1]
if str(AI_SKILLS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SKILLS_DIR))

from skills.simple_skill import simple_test

TOOLS = {
    "simple_test": simple_test,
}
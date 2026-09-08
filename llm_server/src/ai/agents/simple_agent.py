import subprocess
import sys
from pathlib import Path

# Allow both: `python agents/simple_agent.py` and `python -m agents.simple_agent`
_AI_DIR = Path(__file__).resolve().parents[1]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from config.config import MODEL_PATH


def generate(prompt: str) -> str:
    result = subprocess.run(
        [
            "llama-cli",
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", "64",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


if __name__ == "__main__":
    response = generate("Reply with exactly: hello from python")
    print(response)

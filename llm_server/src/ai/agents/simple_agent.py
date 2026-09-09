import subprocess
import sys
import json
from pathlib import Path

# Allow both: `python agents/simple_agent.py` and `python -m agents.simple_agent`
_AI_DIR = Path(__file__).resolve().parents[1]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from config.config import MODEL_PATH
from skills.tools.tools import TOOLS
from engine.llama.inference import localLLM
from utils.prompts.llama import SYSTEM_PROMPT
from utils.llama_utils import extract_json

def run_agent(
    llm: localLLM,
    user_message: str,
) -> str:
    conversation = f"""
{SYSTEM_PROMPT}

USER:
{user_message}

ASSISTANT:
"""

    while True:
        # -----------------------------
        # Ask the LLM what to do
        # -----------------------------

        response = llm.generate(conversation)

        print("LLM OUTPUT:")
        print(response)

        decision = extract_json(response)

        # -----------------------------
        # Normal answer
        # -----------------------------

        if decision["action"] == "answer":
            return decision["content"]

        # -----------------------------
        # Tool call
        # -----------------------------

        if decision["action"] == "tool":
            tool_name = decision["name"]
            tool_arguments = decision.get(
                "arguments",
                {},
            )

            # Find the tool function
            tool_function = TOOLS.get(tool_name)
            if not tool_function:
                raise ValueError(f"Tool {tool_name} not found")

            # Call the tool function
            tool_result = tool_function(**tool_arguments)

            print(
                f"TOOL RESULT: {tool_result}"
            )

            # -----------------------------
            # Give result back to LLM
            # -----------------------------
            conversation += f"""
    {response}

    TOOL RESULT:
    {tool_result}

    Now decide what to do next.

    ASSISTANT:
    """
            continue
        
        raise ValueError(f"Invalid action: {decision['action']}")

def main():
    llm = localLLM()

    try:
        result = run_agent(
            llm,
            "Please run the simple test.",
        )

        print("\nFINAL ANSWER:")
        print(result)
    finally:
        llm.close()

if __name__ == "__main__":
    main()

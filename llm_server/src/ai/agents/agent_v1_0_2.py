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


def _usable_tool_arguments(arguments) -> dict:
    """Drop schema fragments the model copied from TOOLS_DESCRIPTIONS."""
    if not isinstance(arguments, dict):
        return {}
    usable = {}
    for key, value in arguments.items():
        if isinstance(value, dict) and "type" in value and "description" in value:
            continue
        usable[key] = value
    return usable


def run_agent(
    llm: localLLM,
    user_message: str,
) -> str:
    agent_state = {}
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
        print(conversation)
        response = llm.generate(conversation)

        print("LLM OUTPUT:")
        print(response)

        # Return if response is empty
        if not response:
            return "No response from the LLM. Last conversation: " + conversation

        decision = extract_json(response)
        action = decision.get("action")

        # -----------------------------
        # Normal answer
        # -----------------------------

        if action == "answer":
            return decision["content"]

        # -----------------------------
        # Tool call
        # -----------------------------
        # Canonical: {"action": "tool", "name": "get_user_input", ...}
        # Small models often emit {"action": "get_user_input", ...} instead.

        tool_name = decision.get("name") if action == "tool" else action
        tool_function = TOOLS.get(tool_name)
        if tool_function:
            tool_arguments = _usable_tool_arguments(
                decision.get(
                    "arguments",
                    {},
                ),
            )

            try:

                result = tool_function(
                    **tool_arguments
                )

                tool_result = {
                    "success": True,
                    "tool": tool_name,
                    "result": result,
                }

            except Exception as exc:

                tool_result = {
                    "success": False,
                    "tool": tool_name,
                    "error": str(exc),
                }


            conversation += f"""

        Previous LLM response:
        {response}

        TOOL RESULT:
        {json.dumps(tool_result, indent=2)}

        Now decide what to do next.

        If success is true, use the tool result to continue the task.

        If success is false because required information is missing,
        use another appropriate tool to obtain that information.

        Do not invent file paths, URIs, user input, or tool results.

        ASSISTANT:
        """

            continue

        raise ValueError(f"Invalid action: {action}")

def main():
    llm = localLLM()

    try:
        result = run_agent(
            llm,
            "Ask the user for the image URI input. Then parse the text in the image and return the text.",
        )

        print("\nFINAL ANSWER:")
        print(result)
    finally:
        llm.close()

if __name__ == "__main__":
    main()

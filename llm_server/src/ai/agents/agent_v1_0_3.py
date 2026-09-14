import sys
import json
from pathlib import Path


# ---------------------------------------------------------
# Allow both:
# python agents/agent_v1.py
# python -m agents.agent_v1
# ---------------------------------------------------------

_AI_DIR = Path(__file__).resolve().parents[1]

if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))


from skills.tools.tools import TOOLS
from engine.llama.inference import localLLM
from utils.prompts.llama import SYSTEM_PROMPT
from utils.llama_utils import extract_json


MAX_AGENT_STEPS = 10


def _usable_tool_arguments(arguments) -> dict:
    """
    Remove tool-schema fragments accidentally copied by the LLM.

    Example of a bad model output:

    {
        "prompt": {
            "type": "string",
            "description": "..."
        }
    }

    We do not want to pass that dictionary to the actual Python skill.
    """

    if not isinstance(arguments, dict):
        return {}

    usable = {}

    for key, value in arguments.items():
        print(key, value)
        if (
            isinstance(value, dict)
            and "type" in value
            and "description" in value
        ):
            continue

        usable[key] = value

    return usable


def _execute_tool(
    tool_name: str,
    tool_arguments: dict,
) -> dict:
    """
    Execute one tool and always return a structured result.
    """

    tool_function = TOOLS.get(tool_name)

    if tool_function is None:
        return {
            "tool": tool_name,
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }

    # -----------------------------------------------------
    # Small-model correction:
    # Qwen sometimes copies the schema instead of generating
    # the actual prompt string.
    # -----------------------------------------------------

    if tool_name == "get_user_input":

        prompt = tool_arguments.get("prompt")

        if not isinstance(prompt, str):
            tool_arguments["prompt"] = "Enter the image file path: "

    try:

        result = tool_function(**tool_arguments)

        # -------------------------------------------------
        # Skill already returned our structured format
        # -------------------------------------------------

        if isinstance(result, dict) and "success" in result:

            return {
                "tool": tool_name,
                **result,
            }

        # -------------------------------------------------
        # Skill returned a normal raw value
        # -------------------------------------------------

        return {
            "tool": tool_name,
            "success": True,
            "result": result,
        }

    except Exception as exc:

        return {
            "tool": tool_name,
            "success": False,
            "error": str(exc),
        }


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

    for step in range(MAX_AGENT_STEPS):

        print(
            f"\n========== AGENT STEP {step + 1} ==========\n"
        )

        # -------------------------------------------------
        # 1. Ask the LLM what to do
        # -------------------------------------------------

        response = llm.generate(conversation)

        print("\nLLM OUTPUT:")
        print(response)

        if not response:
            return (
                "No response from the LLM.\n\n"
                f"Last conversation:\n{conversation}"
            )

        # -------------------------------------------------
        # 2. Parse model output
        # -------------------------------------------------

        try:
            decision = extract_json(response)

        except Exception as exc:

            conversation += f"""

The previous response could not be parsed as valid JSON.

ERROR:
{exc}

Return exactly one valid JSON object.

ASSISTANT:
"""

            continue

        action = decision.get("action")

        # -------------------------------------------------
        # 3. Final answer
        # -------------------------------------------------

        if action == "answer":

            content = decision.get("content")

            if isinstance(content, str):
                return content

            conversation += """

Your answer action did not contain a valid string in "content".

Return:

{
  "action": "answer",
  "content": "..."
}

ASSISTANT:
"""

            continue

        # -------------------------------------------------
        # 4. Resolve tool name
        #
        # Canonical:
        #
        # {
        #   "action": "tool",
        #   "name": "get_user_input"
        # }
        #
        # Small-model fallback:
        #
        # {
        #   "action": "get_user_input"
        # }
        # -------------------------------------------------

        if action == "tool":
            tool_name = decision.get("name")
        else:
            tool_name = action

        # -------------------------------------------------
        # 5. Clean arguments
        # -------------------------------------------------

        tool_arguments = _usable_tool_arguments(
            decision.get("arguments", {})
        )

        # -------------------------------------------------
        # 6. Execute tool
        # -------------------------------------------------

        tool_result = _execute_tool(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

        print("\nTOOL RESULT:")
        print(
            json.dumps(
                tool_result,
                indent=2,
                ensure_ascii=False,
            )
        )

        # -------------------------------------------------
        # 7. Add model decision + real tool result
        #    back into the conversation.
        # -------------------------------------------------

        conversation = f"""

{SYSTEM_PROMPT}

USER:
{user_message}

You have just taken the following action:
TOOL RESULT:
{json.dumps(
    tool_result,
    indent=2,
    ensure_ascii=False,
)}

Look at the tool result. It is the last taken action.
If the tool result is the final result of the task, call action="answer" and return the result.
Otherwise, continue the given task by identifying the next action to take using the TOOL RESULT to accomplish the task.

Rules:
- You can only take one single action per turn.
- If success is false, the tool failed.
- If success is true, use the returned result if needed by the next action.
- Local file path can be anything, it can even be just the filename.
- Do not take the argument name by heart. Focus on which action to take next and what values to pass as arguments from the TOOL RESULT.

WRONG ASSUMPTION:
The user has requested to identify the text in the local image file path 'test.png'. However, the user has not provided the actual image file path.
RIGHT ASSUMPTION:
The user has provided the image file path as 'test.png'. It can be used if needed by the next action.

ASSISTANT:
"""

    # -----------------------------------------------------
    # Protection against infinite agent loops
    # -----------------------------------------------------

    return (
        f"Agent stopped after {MAX_AGENT_STEPS} steps "
        "without completing the task."
    )


def main():

    llm = localLLM()

    try:

        result = run_agent(
            llm,
            (
                "Ask the user for the local image file path. "
                "Then parse the text from the image and return "
                "the extracted text."
            ),
        )

        print("\nFINAL ANSWER:")
        print(result)

    finally:
        llm.close()


if __name__ == "__main__":
    main()
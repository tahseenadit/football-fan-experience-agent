"""
Task: Parse text from image.
"""

import json
from engine.llama.inference import localLLM
from utils.prompts.llama import SYSTEM_PROMPT
from utils.llama_utils import extract_json
from utils.agent_utils import usable_tool_arguments, execute_tool, apply_chat_template

MAX_AGENT_STEPS = 10

def perform_tasks(
    llm: localLLM,
    user_message: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    for step in range(MAX_AGENT_STEPS):

        print(
            f"\n========== AGENT STEP {step + 1} ==========\n"
        )

        # -------------------------------------------------
        # 1. Ask the LLM what to do
        # -------------------------------------------------

        prompt = apply_chat_template(
            messages,
            add_generation_prompt = True,
            enable_thinking = False,
        )
        response = llm.generate(prompt)

        print("\nLLM OUTPUT:")
        print(response)

        if not response:
            return (
                "No response from the LLM.\n\n"
                f"Last conversation:\n{prompt}"
            )

        # -------------------------------------------------
        # 2. Parse model output
        # -------------------------------------------------

        try:
            decision = extract_json(response)

        except Exception as exc:
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    "The previous response could not be parsed as valid JSON.\n"
                    f"ERROR:\n{exc}\n"
                    "Return exactly one valid JSON object."
                ),
            })

            continue

        action = decision.get("action")

        # -------------------------------------------------
        # 3. Final answer
        # -------------------------------------------------

        if action == "answer":

            content = decision.get("content")

            if isinstance(content, str):
                return content

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    "Your answer action did not contain a valid string in \"content\".\n"
                    "Return:\n"
                    '{"action": "answer", "content": "..."}'
                ),
            })

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

        tool_arguments = usable_tool_arguments(
            decision.get("arguments", {})
        )

        # -------------------------------------------------
        # 6. Execute tool
        # -------------------------------------------------

        tool_result = execute_tool(
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

        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": (
                "You have just taken the following action:\n"
                "TOOL RESULT:\n"
                f"{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n\n"
                "Look at the tool result. It is the last taken action.\n"
                "If the tool result is the final result of the task, "
                'call action="answer" and return the result.\n'
                "Otherwise, continue the given task by identifying the next "
                "action to take using the TOOL RESULT to accomplish the task.\n\n"
                "Rules:\n"
                "- You can only take one single action per turn.\n"
                "- If success is false, the tool failed.\n"
                "- If success is true, use the returned result if needed by the next action.\n"
                "- Local file path can be anything, it can even be just the filename.\n"
                "- Do not take the argument name by heart. Focus on which action "
                "to take next and what values to pass as arguments from the TOOL RESULT.\n\n"
                "WRONG ASSUMPTION:\n"
                "The user has requested to identify the text in the local image "
                "file path 'test.png'. However, the user has not provided the actual image file path.\n"
                "RIGHT ASSUMPTION:\n"
                "The user has provided the image file path as 'test.png'. "
                "It can be used if needed by the next action."
            ),
        })

    # -----------------------------------------------------
    # Protection against infinite agent loops
    # -----------------------------------------------------

    return (
        f"Agent stopped after {MAX_AGENT_STEPS} steps "
        "without completing the task."
    )
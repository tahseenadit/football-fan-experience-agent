from skills.tools.tools import TOOLS

def usable_tool_arguments(arguments) -> dict:
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


def execute_tool(
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

def apply_chat_template(
    messages: list,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False
) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(
            f"<|im_start|>{role}\n{content}<|im_end|>\n"
        )

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
        if enable_thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")
    
    return "".join(parts)
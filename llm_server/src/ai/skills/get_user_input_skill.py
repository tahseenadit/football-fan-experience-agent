def get_user_input(
    prompt: str = "Enter your input: "
) -> dict:
    """
    Ask the human for input through the terminal.
    """

    value = input(prompt)

    return {
        "success": True,
        "result": value.strip(),
    }

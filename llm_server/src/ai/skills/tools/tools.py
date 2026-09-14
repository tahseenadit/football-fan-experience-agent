import sys
from pathlib import Path
import json

_AI_DIR = Path(__file__).resolve().parents[2]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from skills.simple_skill import simple_test
from skills.get_user_input_skill import get_user_input
from skills.swedish_language_skills.skills import parse_text_from_image_of_swedish_document

def get_all_tools_description() -> str:
    """
    Get all tools descriptions
    """
    return json.dumps(TOOLS_DESCRIPTIONS, indent=2, default=str)

TOOLS = {
    "simple_test": simple_test,
    "get_user_input": get_user_input,
    "parse_text_from_image_of_swedish_document": parse_text_from_image_of_swedish_document,
}

TOOLS_DESCRIPTIONS = {
    "simple_test": {
        "description": "Runs a simple test.",
        "arguments": {},
    },
    "get_user_input": {
        "description": (
            "Ask the human in the terminal and return what they type. "
            "Use this whenever you need a file path, URI, or any other value "
            "you do not already have. Never guess or invent that value."
        ),
        "arguments": {
            "prompt": {
                "type": "string",
                "description": "Short question to show, e.g. Enter the image file path:",
            },
        },
    },
    "parse_text_from_image_of_swedish_document": {
        "description": (
            "OCR a local image of a Swedish document. "
            "document_image_uri must be a local file path from get_user_input. "
            "Do not invent paths or internet URLs."
        ),
        "arguments": {
            "document_image_uri": {
                "type": "string",
                "description": "The URI of the image of the Swedish document.",
            },
        },
    },
}
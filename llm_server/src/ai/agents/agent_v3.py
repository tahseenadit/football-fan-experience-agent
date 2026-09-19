"""
Agent V3:
- Task: Parse text from image.
"""

import sys
from pathlib import Path


# ---------------------------------------------------------
# Allow both:
# python agents/agent_v1.py
# python -m agents.agent_v1
# ---------------------------------------------------------

_AI_DIR = Path(__file__).resolve().parents[1]

if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from utils.tasks.utils import perform_tasks
from engine.llama.inference import localLLM


def main():

    llm = localLLM()

    try:

        result = perform_tasks(
            llm,
            (
                "Ask the user for the local image file path. "
                "Then parse the text from the image and return "
                "the extracted text."
            ),
        )

        print("\nPARSED TEXT:")
        print(result)

        final_result = perform_tasks(
            llm,
            (
                """
                You are a semantic information extraction agent.

Your task is to analyze the provided text and convert it into structured JSON based entirely on the meaning of the text.

Instructions:

1. Identify the distinct pieces of information present in the text.
2. Determine what each piece of information represents.
3. Create clear and descriptive JSON keys for those pieces of information.
4. Do not assume a predefined schema. The JSON structure must be derived dynamically from the provided text.
5. Group related information together when appropriate.
6. Use nested JSON objects when information naturally belongs to a larger concept.
7. Use JSON arrays when multiple values belong to the same category.
8. Preserve specific values, names, dates, numbers, identifiers, organizations, and other important information from the original text.
9. When explanatory text describes what something means or represents, summarize its meaning concisely rather than simply copying the entire sentence.
10. Do not invent, infer, or add information that is not supported by the provided text.
11. Do not create fields for information that is not present.
12. Choose JSON key names that clearly describe the meaning of the extracted information.
13. Return valid JSON only.
14. Do not include Markdown, code fences, comments, explanations, or any text outside the JSON.

Here is your input TEXT:""" + result + """

EXAMPLE OF THE OUTPUT JSON STRUCTURE:
{
"document_type": "<type of document>",
"authority": "<organization or authority>",
"page_title": "<title>",
"date": "<date>",
"time": "<time>",
"sections": {
"<descriptive_section_name>": "<what this section indicates>",
"<descriptive_section_name>": [
"<item>",
"<item>"
]
}
}"""
            ),
        )

        print("\nFINAL ANSWER:")
        print(final_result)

    finally:
        llm.close()


if __name__ == "__main__":
    main()
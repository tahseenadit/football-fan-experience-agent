"""
Swedish language skills:

- parse text from an image
"""

from PIL import Image
import pytesseract

def ocr_image(image: Image) -> str:
    """
    Image to text conversion
    """
    return pytesseract.image_to_string(image, lang="swe")

def clean_text(text: str) -> str:
    """
    Clean the text to get the text

    - keep only non-empty stripped lines.
    """
    return "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )

def parse_text_from_image_of_swedish_document(document_image_uri: str) -> str:
    """
    Parse text from an image of a Swedish document
    """
    if document_image_uri.startswith(("http://", "https://")):
        raise ValueError(
            "document_image_uri must be a local file path from get_user_input, "
            "not an internet URL."
        )

    text_from_image = ""
    # 1. image to text
    text_from_image = ocr_image(Image.open(document_image_uri))
    # 2. Clean the text to get the text
    text_from_image = clean_text(text_from_image)
    # 3. Return the text
    return text_from_image
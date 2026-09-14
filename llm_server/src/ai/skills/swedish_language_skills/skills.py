from pathlib import Path

from PIL import Image
import pytesseract
import os
from config.config import ASSET_DIR

def ocr_image(image: Image.Image) -> str:
    """
    Convert a Swedish document image to text.
    """
    return pytesseract.image_to_string(
        image,
        lang="swe",
    )


def clean_text(text: str) -> str:
    """
    Keep only non-empty stripped lines.
    """
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def parse_text_from_image_of_swedish_document(
    document_image_uri: str,
) -> dict:
    """
    Parse text from a local image of a Swedish document.
    """

    # 1. Reject internet URLs
    if document_image_uri.startswith(
        ("http://", "https://")
    ):
        return {
            "success": False,
            "error": (
                "document_image_uri must be a local file path, "
                "not an internet URL."
            ),
        }

    # 2. Normalize the path
    image_path = Path(os.path.join(ASSET_DIR, 'images', document_image_uri)).resolve()

    print(image_path)
    # 3. Check that the path exists
    if not image_path.exists():
        return {
            "success": False,
            "error": f"File does not exist: {image_path}",
        }

    # 4. Check that it is actually a file
    if not image_path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {image_path}",
        }

    try:
        # 5. Open image
        image = Image.open(image_path)

        # 6. OCR
        text = ocr_image(image)

        # 7. Clean OCR output
        text = clean_text(text)

    except Exception as exc:
        return {
            "success": False,
            "error": f"OCR failed: {exc}",
        }

    # 8. OCR succeeded but found nothing
    if not text:
        return {
            "success": False,
            "error": "OCR completed but no text was found in the image.",
        }

    # 9. Successful result
    return {
        "success": True,
        "text": text,
    }
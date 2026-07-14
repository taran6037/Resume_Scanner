
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from config.pipeline_config import OCR_RENDER_DPI, TESSERACT_CONFIG, TESSERACT_LANG


def extract_ocr_text(path: str | Path) -> tuple[str, int]:
    path = Path(path)

    _check_tesseract()

    try:
        import fitz 
    except ImportError:
        raise OCRError("PyMuPDF (fitz) is required for OCR. Run: pip install pymupdf")

    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise OCRError(
            "pytesseract and Pillow are required for OCR. "
            "Run: pip install pytesseract pillow"
        )

    pages_text = []

    try:
        doc = fitz.open(str(path))
        page_count = len(doc)

        logger.info(
            f"Starting OCR on {path.name} — "
            f"{page_count} pages at {OCR_RENDER_DPI} DPI"
        )

        for page_num, page in enumerate(doc):
            logger.debug(f"OCR processing page {page_num + 1}/{page_count}")
            mat    = fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pixmap.tobytes("png")
            image     = Image.open(io.BytesIO(img_bytes))
            image = _preprocess_image(image)
            try:
                page_text = pytesseract.image_to_string(
                    image,
                    lang=TESSERACT_LANG,
                    config=TESSERACT_CONFIG,
                )
                page_text = page_text.strip()

                if page_text:
                    pages_text.append(
                        f"--- Page {page_num + 1} (OCR) ---\n{page_text}"
                    )
                else:
                    logger.warning(
                        f"OCR returned empty text for page {page_num + 1} "
                        f"of {path.name}"
                    )

            except Exception as e:
                logger.warning(
                    f"Tesseract failed on page {page_num + 1} "
                    f"of {path.name}: {e}"
                )
                continue

        doc.close()

    except Exception as e:
        raise OCRError(f"OCR failed on {path.name}: {e}")

    full_text = "\n\n".join(pages_text)

    logger.info(
        f"OCR complete: {len(full_text)} chars extracted "
        f"from {page_count} pages of {path.name}"
    )

    return full_text, page_count



def _preprocess_image(image):
    from PIL import ImageEnhance, ImageFilter

    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image    = enhancer.enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image

def _check_tesseract() -> None:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        raise OCRError(
            "Tesseract OCR is not installed or not on PATH. "
        )
class OCRError(Exception):
    """Raised when OCR fails or Tesseract is not available."""
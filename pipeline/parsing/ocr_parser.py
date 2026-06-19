# pipeline/parsing/ocr_parser.py
#
# OCR fallback for scanned PDFs — PDFs that are just images with no text layer.
# Only called by router.py when pdf_parser returns sparse/empty text.
#
# Strategy:
#   1. Render each PDF page to a high-resolution image (300 DPI)
#   2. Pre-process the image for better OCR accuracy (grayscale, contrast)
#   3. Run Tesseract OCR on each page image
#   4. Combine page results into a single text block
#
# Tesseract config:
#   --oem 3  → LSTM neural net engine (most accurate)
#   --psm 1  → automatic page segmentation with OSD
#              (best for mixed-layout documents like resumes)
#
# Note: OCR accuracy depends heavily on scan quality.
# Blurry, rotated, or low-DPI scans will give poor results.
# The router stores is_scanned=True so the audit trail reflects this.
#
# Returns: (raw_text, page_count)
# Called by: pipeline/ingestion/router.py

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

from config.pipeline_config import OCR_RENDER_DPI, TESSERACT_CONFIG, TESSERACT_LANG


def extract_ocr_text(path: str | Path) -> tuple[str, int]:
    """
    Extracts text from a scanned PDF using Tesseract OCR.

    Args:
        path: Path to the scanned PDF file.

    Returns:
        (raw_text, page_count)

    Raises:
        OCRError if Tesseract is not installed or fails completely.
    """
    path = Path(path)

    # Check Tesseract is available before trying
    _check_tesseract()

    try:
        import fitz   # PyMuPDF — for rendering pages to images
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

            # ── Render page to image ───────────────────────────────────────
            mat    = fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)

            # Convert pixmap to PIL Image via bytes
            img_bytes = pixmap.tobytes("png")
            image     = Image.open(io.BytesIO(img_bytes))

            # ── Pre-process for better OCR ─────────────────────────────────
            image = _preprocess_image(image)

            # ── Run Tesseract ──────────────────────────────────────────────
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


# ─── Image pre-processing ─────────────────────────────────────────────────────

def _preprocess_image(image):
    """
    Applies basic image processing to improve Tesseract accuracy.
    Converts to grayscale and increases contrast.
    More aggressive processing (deskew, denoise) can be added if needed.
    """
    from PIL import ImageEnhance, ImageFilter

    # Convert to grayscale — Tesseract works better on grayscale
    image = image.convert("L")

    # Increase contrast slightly — helps with faded scans
    enhancer = ImageEnhance.Contrast(image)
    image    = enhancer.enhance(1.5)

    # Sharpen slightly — helps with blurry scans
    image = image.filter(ImageFilter.SHARPEN)

    return image


# ─── Tesseract availability check ────────────────────────────────────────────

def _check_tesseract() -> None:
    """
    Checks if Tesseract is installed and callable.
    Raises OCRError with a clear install message if not found.
    """
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        raise OCRError(
            "Tesseract OCR is not installed or not on PATH. "
            "Install it with:\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  macOS:         brew install tesseract\n"
            "  Windows:       https://github.com/UB-Mannheim/tesseract/wiki"
        )


# ─── Custom exceptions ────────────────────────────────────────────────────────

class OCRError(Exception):
    """Raised when OCR fails or Tesseract is not available."""
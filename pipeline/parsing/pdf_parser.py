import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from config.pipeline_config import MIN_CHARS_PER_PAGE, PDF_X_TOLERANCE, PDF_Y_TOLERANCE


def extract_pdf_text(path: str | Path) -> tuple[str, int]:
    path = Path(path)

    try:
        pymupdf_text, page_count = _extract_pymupdf(path)
        chars_per_page = len(pymupdf_text) / max(page_count, 1)

        if chars_per_page >= MIN_CHARS_PER_PAGE:
            logger.info(
                f"PyMuPDF extracted {len(pymupdf_text)} chars "
                f"over {page_count} pages from {path.name}"
            )
            return pymupdf_text, page_count

        logger.info(
            f"PyMuPDF returned sparse text ({chars_per_page:.0f} chars/page). "
            f"Trying pdfplumber."
        )

    except Exception as e:
        logger.warning(f"PyMuPDF failed on {path.name}: {e}. Trying pdfplumber.")
        pymupdf_text = ""
        page_count   = 0

    try:
        plumber_text, plumber_pages = _extract_pdfplumber(path)
        page_count = plumber_pages or page_count

        if len(plumber_text) >= len(pymupdf_text):
            logger.info(
                f"pdfplumber extracted {len(plumber_text)} chars "
                f"from {path.name} (better than PyMuPDF's {len(pymupdf_text)})"
            )
            return plumber_text, page_count
        else:
            logger.info(
                f"PyMuPDF result kept ({len(pymupdf_text)} chars) "
                f"over pdfplumber ({len(plumber_text)} chars)"
            )
            return pymupdf_text, page_count

    except Exception as e:
        logger.warning(f"pdfplumber also failed on {path.name}: {e}")
        return pymupdf_text, page_count

def _extract_pymupdf(path: Path) -> tuple[str, int]:
    import fitz   
    pages_text = []
    with fitz.open(str(path)) as doc:
        page_count = len(doc)
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if not text.strip():
                blocks = page.get_text("blocks")
                if blocks:
                    blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
                    text = "\n".join(
                        b[4].strip() for b in blocks
                        if b[4].strip()
                    )

            if text.strip():
                pages_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

    full_text = "\n\n".join(pages_text)
    return full_text, page_count

def _extract_pdfplumber(path: Path) -> tuple[str, int]:
    import pdfplumber

    pages_text = []

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            page_parts = []
            text = page.extract_text(
                x_tolerance=PDF_X_TOLERANCE,
                y_tolerance=PDF_Y_TOLERANCE,
                layout=True,
                y_density=13,
            )
            if text and text.strip():
                page_parts.append(text.strip())
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(
                        cell.strip() for cell in row
                        if cell and cell.strip()
                    )
                    if row_text:
                        page_parts.append(row_text)

            if page_parts:
                combined = "\n".join(page_parts)
                pages_text.append(f"--- Page {page_num + 1} ---\n{combined}")

    full_text = "\n\n".join(pages_text)
    return full_text, page_count
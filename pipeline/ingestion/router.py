import logging
from dataclasses import dataclass, field
from pathlib import Path
from pipeline.ingestion.validator import ValidationResult

logger = logging.getLogger(__name__)

from config.pipeline_config import OCR_FALLBACK_THRESHOLD

@dataclass
class RouterResult:
    raw_text:      str
    parser_used:   str
    page_count:    int  = 0
    is_scanned:    bool = False
    warnings:      list[str] = field(default_factory=list)
    char_count:    int  = 0
    word_count:    int  = 0


def route_file(validation: ValidationResult) -> RouterResult:
    if not validation.is_valid:
        raise ValueError(
            f"Cannot route an invalid file. "
            f"Validation error was: {validation.error}"
        )

    ext      = validation.extension
    path     = validation.file_path
    warnings = []

    logger.info(f"Routing {validation.file_name} ({ext})")

    if ext == ".txt":
        raw_text = _read_txt(path)
        return _build_result(raw_text, "txt", warnings=warnings)

    if ext in (".docx", ".doc"):
        raw_text, page_count = _parse_docx(path)
        return _build_result(raw_text, "docx", page_count=page_count, warnings=warnings)

    if ext == ".pdf":
        return _route_pdf(path, warnings)

    raise RouterError(f"Unhandled extension: {ext}")

def _route_pdf(path: Path, warnings: list) -> RouterResult:
    from pipeline.parsing.pdf_parser import extract_pdf_text
    from pipeline.parsing.ocr_parser import extract_ocr_text

    logger.info(f"Trying PDF text extraction: {path.name}")
    pdf_text, page_count = extract_pdf_text(path)

    text_length = len(pdf_text.strip())
    logger.debug(f"PDF text extraction returned {text_length} characters.")

    if text_length >= OCR_FALLBACK_THRESHOLD:
        logger.info(f"PDF has text layer ({text_length} chars). Using pdf_parser.")
        return _build_result(
            pdf_text, "pdf",
            page_count=page_count,
            is_scanned=False,
            warnings=warnings
        )

    logger.info(
        f"PDF text too sparse ({text_length} chars < {OCR_FALLBACK_THRESHOLD}). "
        f"Falling back to OCR."
    )
    warnings.append(
        f"PDF appears to be scanned (only {text_length} chars from text layer). "
        "OCR was used — accuracy may vary."
    )

    ocr_text, page_count = extract_ocr_text(path)
    ocr_length = len(ocr_text.strip())

    if ocr_length < OCR_FALLBACK_THRESHOLD:
        raise RouterError(
            f"Both PDF text extraction ({text_length} chars) and OCR "
            f"({ocr_length} chars) returned too little text. "
            "The file may be corrupted, password-protected, or contain only images."
        )

    logger.info(f"OCR extracted {ocr_length} characters.")
    return _build_result(
        ocr_text, "ocr",
        page_count=page_count,
        is_scanned=True,
        warnings=warnings
    )

def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path.name} — trying latin-1.")
        return path.read_text(encoding="latin-1")


def _parse_docx(path: Path) -> tuple[str, int]:
    from pipeline.parsing.docx_parser import extract_docx_text
    return extract_docx_text(path)

def _build_result(
    raw_text:    str,
    parser_used: str,
    page_count:  int  = 0,
    is_scanned:  bool = False,
    warnings:    list = None,
) -> RouterResult:
    warnings = warnings or []
    text     = raw_text.strip()

    if not text:
        raise RouterError(
            f"Parser '{parser_used}' returned empty text. "
            "The file may be corrupted, password-protected, or contain no readable content."
        )

    result = RouterResult(
        raw_text=text,
        parser_used=parser_used,
        page_count=page_count,
        is_scanned=is_scanned,
        warnings=warnings,
        char_count=len(text),
        word_count=len(text.split()),
    )

    logger.info(
        f"Router complete: parser={parser_used}, "
        f"chars={result.char_count}, words={result.word_count}, "
        f"scanned={is_scanned}, warnings={len(warnings)}"
    )

    return result
class RouterError(Exception):
    """Raised when routing fails or all parsers return unusable output."""
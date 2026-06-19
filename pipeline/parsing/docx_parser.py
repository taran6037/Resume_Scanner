import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_docx_text(path: str | Path) -> tuple[str, int]:
    path = Path(path)
    try:
        from docx import Document
        doc = Document(str(path))
    except Exception as e:
        raise DocxParseError(f"Cannot open DOCX file {path.name}: {e}")
    parts = []
    header_text = _extract_headers_footers(doc)
    if header_text:
        parts.append(header_text)
    header_text = _extract_headers_footers(doc)
    if header_text:
        parts.append(header_text)

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            from docx.oxml.ns import qn
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                parts.append(text)

        elif tag == "tbl":
            table_text = _extract_table(element, doc)
            if table_text:
                parts.append(table_text)

    footer_text = _extract_headers_footers(doc, footers=True)
    if footer_text:
        parts.append(footer_text)

    textbox_text = _extract_textboxes(doc)
    if textbox_text:
        parts.append(textbox_text)
    textbox_text = _extract_textboxes(doc)
    if textbox_text:
        parts.append(textbox_text)
    if not parts:
        raise DocxParseError(
            f"No text could be extracted from {path.name}. "
            "The file may be empty, corrupted, or use unsupported formatting."
        )
    full_text  = "\n".join(parts)
    page_count = _estimate_page_count(full_text)

    logger.info(
        f"DOCX extracted {len(full_text)} chars, "
        f"~{page_count} pages from {path.name}"
    )

    return full_text, page_count


def _extract_table(table_element, doc) -> str:
    from docx.oxml.ns import qn

    rows_text = []
    for row in table_element.findall(f".//{{{_w_ns()}}}tr"):
        cells = []
        for cell in row.findall(f".//{{{_w_ns()}}}tc"):
            cell_text_parts = []
            for para in cell.findall(f".//{{{_w_ns()}}}p"):
                para_text = "".join(
                    run.text for run in para.findall(f".//{{{_w_ns()}}}t")
                ).strip()

                if para_text:
                    cell_text_parts.append(para_text)
            cell_text = " ".join(cell_text_parts).strip()
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows_text.append(" | ".join(cells))

    return "\n".join(rows_text)


def _extract_headers_footers(doc, footers: bool = False) -> str:
    parts = []
    try:
        sections = doc.sections
        for section in sections:

            container = section.footer if footers else section.header
            for para in container.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)

    except Exception:
        pass

    return "\n".join(parts)

def _extract_textboxes(doc) -> str:
    parts = []
    try:
        body = doc.element.body
        for txbx in body.iter(f"{{{_wps_ns()}}}txbx"):
            for para in txbx.iter(f"{{{_w_ns()}}}p"):
                text = "".join(
                    run.text for run in para.iter(f"{{{_w_ns()}}}t")
                ).strip()
                if text:
                    parts.append(text)

    except Exception:
        pass

    return "\n".join(parts)

def _estimate_page_count(text: str) -> int:
    word_count = len(text.split())

    return max(1, round(word_count / 450))

def _w_ns() -> str:
    return "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _wps_ns() -> str:
    return "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"

class DocxParseError(Exception):
    """
    Raised when a DOCX file cannot be opened or parsed.
    Using a custom exception instead of a generic one means callers
    can catch specifically DocxParseError and know exactly what failed.
    """
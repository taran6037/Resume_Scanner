import re
import logging
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from config.pipeline_config import MAX_BLANK_LINES, MIN_GARBAGE_LINE_LENGTH

@dataclass
class CleanResult:  
    clean_text:          str
    original_length:     int
    cleaned_length:      int
    injection_flags:     list[str]
    was_ocr:             bool

def clean_text(raw_text: str, was_ocr: bool = False) -> CleanResult:
    original_length  = len(raw_text)
    injection_flags  = []
    text             = raw_text
    text, flags = _strip_injection_attempts(text)
    injection_flags.extend(flags)
    text = _strip_invisible_unicode(text)
    text = _fix_encoding(text)
    text = _remove_page_markers(text)
    text = _normalize_bullets(text)
    text = _fix_broken_lines(text)
    if was_ocr:
        text = _fix_ocr_artifacts(text)
    text = _normalize_whitespace(text)
    text = text.strip()

    cleaned_length = len(text)

    if injection_flags:
        logger.warning(
            f"Prompt injection flags found and removed: {injection_flags}"
        )

    logger.info(
        f"Cleaning complete: {original_length} → {cleaned_length} chars "
        f"(removed {original_length - cleaned_length}). "
        f"Injection flags: {len(injection_flags)}"
    )

    return CleanResult(
        clean_text=text,
        original_length=original_length,
        cleaned_length=cleaned_length,
        injection_flags=injection_flags,
        was_ocr=was_ocr,
    )

_INJECTION_PATTERNS = [
    (r"(?i)ignore\s+(all\s+)?(previous\s+)?instructions?", "ignore_instructions"),
    (r"(?i)you\s+are\s+now\s+a", "persona_override"),
    (r"(?i)system\s*:\s*", "system_prompt_injection"),
    (r"(?i)rank\s+(me|this\s+candidate)\s+(as\s+)?(#\s*1|first|top)", "rank_manipulation"),
    (r"(?i)disregard\s+(the\s+)?(above|previous|prior)", "disregard_injection"),
    (r"(?i)your\s+new\s+instructions?\s+are", "instruction_override"),
    (r"(?i)forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)", "memory_wipe_attempt"),
    (r"(?i)<\s*prompt\s*>.*?<\s*/\s*prompt\s*>", "html_prompt_tag"),
    (r"(?i)\[INST\].*?\[/INST\]", "llama_instruction_tag"),
]


def _strip_injection_attempts(text: str) -> tuple[str, list[str]]:
    flags = []
    for pattern, label in _INJECTION_PATTERNS:
        new_text, count = re.subn(pattern, " ", text, flags=re.DOTALL)
        if count > 0:
            flags.append(f"{label} (x{count})")
            text = new_text
    return text, flags


def _strip_invisible_unicode(text: str) -> str:
    invisible = [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u00ad",
    ]
    for char in invisible:
        text = text.replace(char, "")

    direction_overrides = [
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2066", "\u2067", "\u2068", "\u2069",
    ]
    for char in direction_overrides:
        text = text.replace(char, "")

    return text

def _fix_encoding(text: str) -> str:
    replacements = {
        "\u2019": "'",    "\u2018": "'",
        "\u201c": '"',    "\u201d": '"',
        "\u2013": "-",    "\u2014": "-",
        "\u2022": "-",    "\u2023": "-",
        "\u25cf": "-",    "\u25aa": "-",
        "\u00a0": " ",
        "\u2026": "...",
        "\u00b7": "-",
        "\u2212": "-",
        "\r\n":   "\n",
        "\r":     "\n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _remove_page_markers(text: str) -> str:
    text = re.sub(r"---\s*Page\s+\d+\s*(\(OCR\)\s*)?---\n?", "", text)
    return text


def _normalize_bullets(text: str) -> str:
    text = re.sub(r"^[\•\◦\▪\▸\►\✓\✔\→\-\*]\s+", "- ", text, flags=re.MULTILINE)
    return text


def _fix_broken_lines(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"([a-z,])\n([a-z])", r"\1 \2", text)
    return text


def _fix_ocr_artifacts(text: str) -> str:
    fixes = [
        (r"\s\|\s", " | "), 
        (rf"^[^a-zA-Z0-9\n]{{{MIN_GARBAGE_LINE_LENGTH},}}$", ""),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(rf"\n{{{MAX_BLANK_LINES + 1},}}", "\n" * MAX_BLANK_LINES, text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)
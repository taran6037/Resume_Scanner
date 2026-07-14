import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from config.pipeline_config import (
    MIN_JD_LENGTH_CHARS, MIN_JD_LENGTH_WORDS,
    MAX_JD_LENGTH_CHARS, JD_TRUNCATION_SENTENCE_SEARCH,
)
MIN_JD_LENGTH   = MIN_JD_LENGTH_CHARS     
MAX_JD_LENGTH   = MAX_JD_LENGTH_CHARS   
MIN_JD_WORDS    = MIN_JD_LENGTH_WORDS     

@dataclass
class JDReadResult:
    clean_text:   str
    word_count:   int
    char_count:   int
    was_truncated: bool
    warnings:     list[str]  


def read_jd(raw_text: str) -> JDReadResult:
    warnings = []

    if not raw_text or not raw_text.strip():
        raise JDEmptyError("Job description text is empty.")

    if len(raw_text.strip()) < MIN_JD_LENGTH:
        raise JDTooShortError(
            f"Job description is too short ({len(raw_text.strip())} chars). "
            f"Minimum is {MIN_JD_LENGTH} characters."
        )

    text = _fix_encoding(raw_text)
    text = _normalize_whitespace(text)
    text, removed = _remove_boilerplate(text)
    if removed:
        warnings.append(f"Removed boilerplate sections: {removed}")

    word_count = len(text.split())
    if word_count < MIN_JD_WORDS:
        raise JDTooShortError(
            f"Job description has too few words ({word_count}). "
            f"Minimum is {MIN_JD_WORDS} words."
        )

    was_truncated = False
    if len(text) > MAX_JD_LENGTH:
        text = text[:MAX_JD_LENGTH]
        last_period = text.rfind(".")
        if last_period > MAX_JD_LENGTH * 0.8:
            text = text[:last_period + 1]
        was_truncated = True
        warnings.append(
            f"JD was truncated to {MAX_JD_LENGTH} characters to fit model context."
        )
        logger.warning(f"JD truncated from original length to {len(text)} chars.")

    final_word_count = len(text.split())
    logger.info(
        f"JD read complete: {final_word_count} words, "
        f"truncated={was_truncated}, warnings={len(warnings)}"
    )

    return JDReadResult(
        clean_text=text,
        word_count=final_word_count,
        char_count=len(text),
        was_truncated=was_truncated,
        warnings=warnings,
    )


def _fix_encoding(text: str) -> str:
    replacements = {
        "\u2019": "'",    
        "\u2018": "'",    
        "\u201c": '"',    
        "\u201d": '"',    
        "\u2013": "-",  
        "\u2014": "-",  
        "\u2022": "-",  
        "\u00a0": " ",  
        "\u2026": "...",  
        "\r\n":   "\n",   
        "\r":     "\n",  
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text  = "\n".join(lines)

    return text.strip()


def _remove_boilerplate(text: str) -> tuple[str, list[str]]:
    removed = []
    boilerplate_patterns = [
        (r"(?i)(we are an equal opportunity employer.*?)($|\n\n)", "equal_opportunity"),
        (r"(?i)(this job description is not exhaustive.*?)($|\n\n)", "legal_disclaimer"),
        (r"(?i)(we offer a competitive.*?benefits.*?)($|\n\n)", "benefits_filler"),
        (r"(?i)(to apply.*?send.*?resume.*?)($|\n\n)", "application_instructions"),
    ]

    for pattern, label in boilerplate_patterns:
        new_text, count = re.subn(pattern, "", text, flags=re.DOTALL)
        if count > 0:
            text = new_text
            removed.append(label)

    return text.strip(), removed


class JDReadError(Exception):
    """Base class for JD reading errors."""

class JDEmptyError(JDReadError):
    """JD text is empty."""

class JDTooShortError(JDReadError):
    """JD text is too short to extract meaningful criteria."""

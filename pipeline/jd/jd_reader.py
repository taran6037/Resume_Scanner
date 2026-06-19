# pipeline/jd/jd_reader.py
#
# Takes raw job description text pasted by a recruiter (Job.raw_jd_text)
# and prepares it for the JD extractor.
#
# Responsibilities:
#   - Validate the text is usable (not empty, not too short)
#   - Light cleaning: fix encoding, normalize whitespace
#   - Truncate if too long (8B models degrade on very long input)
#   - Return clean text ready for jd_extractor.py

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
from config.pipeline_config import (
    MIN_JD_LENGTH_CHARS, MIN_JD_LENGTH_WORDS,
    MAX_JD_LENGTH_CHARS, JD_TRUNCATION_SENTENCE_SEARCH,
)
MIN_JD_LENGTH   = MIN_JD_LENGTH_CHARS     # characters — anything shorter is probably not a real JD
MAX_JD_LENGTH   = MAX_JD_LENGTH_CHARS   # characters — truncate beyond this to stay within context
MIN_JD_WORDS    = MIN_JD_LENGTH_WORDS     # word count sanity check


# ─── Result object ────────────────────────────────────────────────────────────

@dataclass
class JDReadResult:
    clean_text:   str
    word_count:   int
    char_count:   int
    was_truncated: bool
    warnings:     list[str]   # non-fatal issues found during reading


# ─── Main function ────────────────────────────────────────────────────────────

def read_jd(raw_text: str) -> JDReadResult:
    """
    Cleans and validates raw JD text.

    Args:
        raw_text: The job description text pasted by the recruiter.

    Returns:
        JDReadResult with clean text and metadata.

    Raises:
        JDTooShortError  — if text is too short to be a real JD
        JDEmptyError     — if text is empty or whitespace only
    """
    warnings = []

    # ── Step 1: basic validation ───────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        raise JDEmptyError("Job description text is empty.")

    if len(raw_text.strip()) < MIN_JD_LENGTH:
        raise JDTooShortError(
            f"Job description is too short ({len(raw_text.strip())} chars). "
            f"Minimum is {MIN_JD_LENGTH} characters."
        )

    # ── Step 2: fix encoding artifacts ────────────────────────────────────
    text = _fix_encoding(raw_text)

    # ── Step 3: normalize whitespace ──────────────────────────────────────
    text = _normalize_whitespace(text)

    # ── Step 4: remove boilerplate patterns ───────────────────────────────
    # Things like "Equal Opportunity Employer" blocks at the end add noise
    # without any extractable criteria
    text, removed = _remove_boilerplate(text)
    if removed:
        warnings.append(f"Removed boilerplate sections: {removed}")

    # ── Step 5: word count check ───────────────────────────────────────────
    word_count = len(text.split())
    if word_count < MIN_JD_WORDS:
        raise JDTooShortError(
            f"Job description has too few words ({word_count}). "
            f"Minimum is {MIN_JD_WORDS} words."
        )

    # ── Step 6: truncate if too long ──────────────────────────────────────
    was_truncated = False
    if len(text) > MAX_JD_LENGTH:
        text = text[:MAX_JD_LENGTH]
        # Don't cut mid-sentence
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fix_encoding(text: str) -> str:
    """Fix common encoding artifacts from copy-paste."""
    replacements = {
        "\u2019": "'",    # right single quotation mark
        "\u2018": "'",    # left single quotation mark
        "\u201c": '"',    # left double quotation mark
        "\u201d": '"',    # right double quotation mark
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2022": "-",    # bullet point
        "\u00a0": " ",    # non-breaking space
        "\u2026": "...",  # ellipsis
        "\r\n":   "\n",   # Windows line endings
        "\r":     "\n",   # old Mac line endings
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _normalize_whitespace(text: str) -> str:
    """
    Collapse multiple blank lines into one.
    Normalize multiple spaces into one.
    Preserve single newlines (section structure matters for extraction).
    """
    # Multiple spaces → single space (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # More than 2 consecutive newlines → 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text  = "\n".join(lines)

    return text.strip()


def _remove_boilerplate(text: str) -> tuple[str, list[str]]:
    """
    Removes standard boilerplate blocks that add length but no extractable criteria.
    Returns (cleaned_text, list_of_removed_labels).
    """
    removed = []

    boilerplate_patterns = [
        # Equal opportunity statements
        (r"(?i)(we are an equal opportunity employer.*?)($|\n\n)", "equal_opportunity"),
        # Legal disclaimers
        (r"(?i)(this job description is not exhaustive.*?)($|\n\n)", "legal_disclaimer"),
        # Generic company culture filler
        (r"(?i)(we offer a competitive.*?benefits.*?)($|\n\n)", "benefits_filler"),
        # Application instructions
        (r"(?i)(to apply.*?send.*?resume.*?)($|\n\n)", "application_instructions"),
    ]

    for pattern, label in boilerplate_patterns:
        new_text, count = re.subn(pattern, "", text, flags=re.DOTALL)
        if count > 0:
            text = new_text
            removed.append(label)

    return text.strip(), removed


# ─── Custom exceptions ────────────────────────────────────────────────────────

class JDReadError(Exception):
    """Base class for JD reading errors."""

class JDEmptyError(JDReadError):
    """JD text is empty."""

class JDTooShortError(JDReadError):
    """JD text is too short to extract meaningful criteria."""
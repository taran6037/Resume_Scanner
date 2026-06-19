# pipeline/jd/jd_extractor.py
#
# Takes clean JD text from jd_reader.py and produces a validated
# StructuredCriteria object — ready to be stored in Job.structured_criteria.
#
# Flow:
#   raw JD text
#     → jd_reader.read_jd()       — clean + validate text
#     → load jd_extraction_v1.txt — inject text into prompt template
#     → call_ollama_for_schema()   — Qwen call + JSON extraction + retry
#     → StructuredCriteria         — validated Pydantic model
#     → dict                       — ready for Job.structured_criteria column

import logging
from pathlib import Path
from backend.schemas.job import StructuredCriteria, ScoringWeights
from pipeline.jd.jd_reader import read_jd, JDReadError
from pipeline.utils.ollama_client import (
    call_ollama_for_schema,
    check_ollama_health,
    ExtractionFailedError,
)

logger = logging.getLogger(__name__)

# ─── Prompt template path ─────────────────────────────────────────────────────

PROMPT_PATH = Path(__file__).parent.parent.parent / "data" / "prompts" / "jd_extraction_v1.txt"

SYSTEM_PROMPT = (
    "You are a precise job description parser. "
    "You extract structured hiring criteria from job descriptions. "
    "You always respond with a single valid JSON object and nothing else."
)


# ─── Main function ────────────────────────────────────────────────────────────

def extract_jd(
    raw_jd_text: str,
    custom_weights: dict | None = None,
) -> dict:
    """
    Full pipeline: raw JD text → validated StructuredCriteria → dict.

    Args:
        raw_jd_text:    The job description text pasted by the recruiter.
                        Stored in Job.raw_jd_text.
        custom_weights: Optional dict to override default scoring weights.
                        Example: {"skills": 0.5, "experience": 0.2, "llm": 0.2, "education": 0.1}
                        If None, uses defaults: skills=0.40, exp=0.25, llm=0.25, edu=0.10

    Returns:
        dict — validated StructuredCriteria as a dict.
               Store directly in Job.structured_criteria.

    Raises:
        JDReadError        — if the JD text is empty or too short
        ExtractionFailedError — if Qwen fails after all retries
        PromptNotFoundError   — if the prompt template file is missing
    """
    # ── Step 1: clean and validate the JD text ────────────────────────────
    logger.info("Starting JD extraction...")
    read_result = read_jd(raw_jd_text)

    if read_result.warnings:
        for w in read_result.warnings:
            logger.warning(f"JD reader warning: {w}")

    logger.info(
        f"JD ready: {read_result.word_count} words, "
        f"truncated={read_result.was_truncated}"
    )

    # ── Step 2: load and fill prompt template ─────────────────────────────
    prompt = _load_prompt(read_result.clean_text)

    # ── Step 3: call Qwen with retry ──────────────────────────────────────
    logger.info("Calling Qwen for JD extraction...")
    criteria, confidence = call_ollama_for_schema(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        schema=StructuredCriteria,
        context_label="JD extraction",
    )

    # ── Step 4: apply scoring weights ─────────────────────────────────────
    # Use custom weights if provided, otherwise defaults from ScoringWeights
    if custom_weights:
        try:
            weights = ScoringWeights(**custom_weights)
            if not weights.validate_sum():
                logger.warning(
                    "Custom weights do not sum to 1.0 — falling back to defaults."
                )
                weights = ScoringWeights()
        except Exception as e:
            logger.warning(f"Invalid custom weights ({e}) — using defaults.")
            weights = ScoringWeights()
    else:
        weights = ScoringWeights()

    criteria.scoring_weights = weights

    # ── Step 5: attach metadata ───────────────────────────────────────────
    criteria.extraction_confidence = confidence
    criteria.jd_word_count         = read_result.word_count

    # ── Step 6: return as dict for DB storage ─────────────────────────────
    result = criteria.model_dump()

    logger.info(
        f"JD extraction complete. "
        f"Role: {criteria.role_type}, Seniority: {criteria.seniority}, "
        f"Required skills: {len(criteria.skills.required)}, "
        f"Confidence: {confidence}"
    )

    return result


# ─── Prompt loader ────────────────────────────────────────────────────────────

def _load_prompt(clean_jd_text: str) -> str:
    """
    Loads the prompt template and injects the clean JD text.
    Raises PromptNotFoundError if the template file is missing.
    """
    if not PROMPT_PATH.exists():
        raise PromptNotFoundError(
            f"Prompt template not found at: {PROMPT_PATH}. "
            "Make sure data/prompts/jd_extraction_v1.txt exists."
        )

    template = PROMPT_PATH.read_text(encoding="utf-8")

    if "{jd_text}" not in template:
        raise PromptNotFoundError(
            "Prompt template is missing the {jd_text} placeholder."
        )

    return template.replace("{jd_text}", clean_jd_text)


# ─── Convenience: extract + print summary ─────────────────────────────────────

def extract_jd_summary(raw_jd_text: str) -> str:
    """
    Returns a human-readable summary of what was extracted.
    Useful for debugging and the run_pipeline.py test harness.
    """
    result = extract_jd(raw_jd_text)

    skills_req  = result.get("skills", {}).get("required", [])
    skills_pref = result.get("skills", {}).get("preferred", [])
    exp_years   = result.get("experience_years")
    role        = result.get("role_type")
    seniority   = result.get("seniority")
    confidence  = result.get("extraction_confidence")

    lines = [
        "─── JD Extraction Result ───────────────────",
        f"  Role type  : {role}",
        f"  Seniority  : {seniority}",
        f"  Exp needed : {exp_years} years",
        f"  Required   : {', '.join(skills_req) if skills_req else 'none found'}",
        f"  Preferred  : {', '.join(skills_pref) if skills_pref else 'none found'}",
        f"  Confidence : {confidence}",
        "────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ─── Custom exceptions ────────────────────────────────────────────────────────

class PromptNotFoundError(Exception):
    """Prompt template file is missing or malformed."""
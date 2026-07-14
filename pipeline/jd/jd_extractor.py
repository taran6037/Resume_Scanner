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

PROMPT_PATH = Path(__file__).parent.parent.parent / "data" / "prompts" / "jd_extraction_v1.txt"

SYSTEM_PROMPT = (
    "You are a precise job description parser. "
    "You extract structured hiring criteria from job descriptions. "
    "You always respond with a single valid JSON object and nothing else."
)


def extract_jd(
    raw_jd_text: str,
    custom_weights: dict | None = None,
) -> dict:

    logger.info("Starting JD extraction...")
    read_result = read_jd(raw_jd_text)

    if read_result.warnings:
        for w in read_result.warnings:
            logger.warning(f"JD reader warning: {w}")

    logger.info(
        f"JD ready: {read_result.word_count} words, "
        f"truncated={read_result.was_truncated}"
    )

    prompt = _load_prompt(read_result.clean_text)

    logger.info("Calling Qwen for JD extraction...")
    criteria, confidence = call_ollama_for_schema(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        schema=StructuredCriteria,
        context_label="JD extraction",
    )

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

    criteria.extraction_confidence = confidence
    criteria.jd_word_count         = read_result.word_count

    result = criteria.model_dump()

    logger.info(
        f"JD extraction complete. "
        f"Role: {criteria.role_type}, Seniority: {criteria.seniority}, "
        f"Required skills: {len(criteria.skills.required)}, "
        f"Confidence: {confidence}"
    )

    return result


def _load_prompt(clean_jd_text: str) -> str:
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


def extract_jd_summary(raw_jd_text: str) -> str:
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

class PromptNotFoundError(Exception):
    """Prompt template file is missing or malformed."""

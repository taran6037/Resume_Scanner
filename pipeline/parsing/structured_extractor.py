import re
import logging
from typing import Optional
from pydantic import BaseModel

from backend.schemas.candidate import (
    ParsedProfile, Skills, ExperienceEntry,
    EducationEntry, ProjectEntry,
)
from pipeline.parsing.ner_extractor import NERResult
from pipeline.utils.ollama_client import (
    call_ollama_for_schema,
    ExtractionFailedError,
)
from config.pipeline_config import (
    RESUME_TEXT_LIMIT,        
    PARSER_VERSION,           
    CONTACT_STRIP_NAME_LINES, 
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a resume parser. "
    "Extract structured data from resume text. "
    "Respond with a single valid JSON object only. "
    "No explanations. No markdown. No text outside JSON."
)

EXTRACTION_PROMPT_TEMPLATE = """Extract all professional information from this resume text.
Figure out the structure yourself — no section headers needed.

RESUME:
{resume_text}

Return ONLY this JSON. null for missing fields. [] for empty lists.
Set is_current true if end_date is Present or missing.
Calculate total_experience_years from the dates.

{{
  "summary": null,
  "skills": {{
    "technical": [],
    "tools": [],
    "soft": []
  }},
  "experience": [
    {{
      "company": null,
      "title": null,
      "location": null,
      "start_date": null,
      "end_date": null,
      "duration_months": null,
      "responsibilities": [],
      "achievements": [],
      "is_current": false
    }}
  ],
  "education": [
    {{
      "degree": null,
      "field": null,
      "institution": null,
      "graduation_year": null,
      "gpa": null
    }}
  ],
  "projects": [
    {{
      "name": null,
      "description": null,
      "tech_used": [],
      "url": null
    }}
  ],
  "certifications": [],
  "total_experience_years": null
}}"""

def _strip_contact_lines(text: str, ner_result: NERResult) -> str:
    lines         = text.split("\n")
    filtered      = []
    contact       = ner_result.contact
    removed_count = 0
    contact_signals = set()

    if contact.email:
        contact_signals.add(contact.email.lower())

    if contact.phone:
        contact_signals.add(re.sub(r"[\s\-\(\)\+]", "", contact.phone))

    if contact.linkedin:
        contact_signals.add("linkedin.com")

    if contact.github:
        contact_signals.add("github.com")

    contact_name = contact.name.lower() if contact.name else None

    for i, line in enumerate(lines):
        line_lower      = line.lower().strip()
        line_bare_phone = re.sub(r"[\s\-\(\)\+]", "", line_lower)
        is_contact = any(
            s in line_lower or s in line_bare_phone
            for s in contact_signals
        )
        if (not is_contact                        
                and i < CONTACT_STRIP_NAME_LINES   
                and contact_name                   
                and contact_name in line_lower
                and len(line.strip().split()) <= 5):  
            is_contact = True

        if is_contact:
            removed_count += 1
        else:
            filtered.append(line)

    logger.info(f"Contact stripping: removed {removed_count} lines, {len(filtered)} remain.")
    return "\n".join(filtered).strip()

class _LLMExtractionOutput(BaseModel):
    summary:                Optional[str]         = None
    skills:                 Skills                = Skills()
    experience:             list[ExperienceEntry] = []
    education:              list[EducationEntry]  = []
    projects:               list[ProjectEntry]    = []
    certifications:         list[str]             = []
    total_experience_years: Optional[float]       = None

    class Config:
        extra = "ignore"

def extract_structured_profile(
    ner_result:     NERResult,
    clean_text:     str,
    parser_version: str = PARSER_VERSION,
) -> ParsedProfile:
    
    safe_text = _strip_contact_lines(clean_text, ner_result)
    if not safe_text.strip():
        logger.warning("All text was stripped as contact lines — using full text as fallback.")
        safe_text = clean_text
    if len(safe_text) > RESUME_TEXT_LIMIT:
        safe_text = safe_text[:RESUME_TEXT_LIMIT]
        logger.info(f"Resume text truncated to {RESUME_TEXT_LIMIT} chars.")

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(resume_text=safe_text)
    logger.info("Calling Qwen for structured resume extraction...")
    llm_result, confidence = call_ollama_for_schema(
        prompt        = prompt,
        system_prompt = SYSTEM_PROMPT,
        schema        = _LLMExtractionOutput,
        context_label = "resume extraction",
    )

    total_years = llm_result.total_experience_years or _compute_experience_years(
        llm_result.experience
    )

    profile = ParsedProfile(
        contact                = ner_result.contact,
        summary                = llm_result.summary,
        skills                 = llm_result.skills,
        experience             = llm_result.experience,
        education              = llm_result.education,
        projects               = llm_result.projects,
        certifications         = llm_result.certifications,
        total_experience_years = total_years,

        raw_sections           = {},
        parser_version         = parser_version,
        extraction_confidence  = confidence,
    )

    logger.info(
        f"Extraction complete — "
        f"{len(profile.skills.technical)} technical skills, "
        f"{len(profile.experience)} jobs, "
        f"{profile.total_experience_years} years experience, "
        f"confidence {confidence}"
    )

    return profile

def _compute_experience_years(experience: list[ExperienceEntry]) -> Optional[float]:
    if not experience:
        return None   

    total_months = 0
    has_data = False

    for entry in experience:
        if entry.duration_months:
            total_months += entry.duration_months
            has_data = True

        elif entry.start_date and entry.end_date:
            sy = _extract_year(entry.start_date)
            ey = _extract_year(
                "2025" if str(entry.end_date).lower() in
                ("present", "current", "now") else entry.end_date
            )

            if sy and ey:
                total_months += max(0, (ey - sy) * 12)
                has_data = True
    return round(total_months / 12, 1) if has_data else None


def _extract_year(date_str: str) -> Optional[int]:
    match = re.search(r"\b(19\d{2}|20[0-3]\d)\b", str(date_str))
    return int(match.group(1)) if match else None
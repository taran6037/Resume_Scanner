import re
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from pipeline.utils.ollama_client import call_ollama_for_schema, ExtractionFailedError
from config.pipeline_config import SEMANTIC_MATCHING_PROMPT_PATH, SEMANTIC_SCORE_MIN_THRESHOLD

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technical hiring expert. "
    "You compare candidate profiles against job descriptions and return structured match analysis. "
    "Respond with a single valid JSON object only. No explanations. No markdown."
)

SCORE_FLOOR = 8


class SemanticMatchResult(BaseModel):
    semantic_score:       int
    skill_matches:        list[str] = []
    skill_gaps:           list[str] = []
    experience_relevance: int
    reasoning:            str
    is_transferable:      bool = False
    education_match:      Optional[bool] = None
    education_notes:      Optional[str]  = None

    class Config:
        extra = "ignore"


def _has_education_requirement(education: dict) -> bool:
    if not education:
        return False
    return bool(
        education.get("minimum_degree")
        or education.get("minimum_gpa")
        or education.get("preferred_fields")
        or education.get("preferred_institutions")
    )


def _has_experience_requirement(criteria: dict) -> bool:
    return bool(criteria.get("experience_years") or criteria.get("responsibilities"))

def _has_location_requirement(criteria: dict) -> bool:
    return bool(criteria.get("location") or criteria.get("work_mode"))

def _parse_gpa_value(gpa_str) -> Optional[tuple]:
    if not gpa_str:
        return None
    s = str(gpa_str).strip()
    m = re.search(r"\d+(\.\d+)?", s)
    if not m:
        return None
    value = float(m.group())
    is_percentage = "%" in s
    return (value, is_percentage)


_DEGREE_LEVELS = {
    "bachelor": 1, "b.tech": 1, "btech": 1, "b.e": 1, "be": 1,
    "b.sc": 1, "bsc": 1, "bca": 1, "undergraduate": 1,
    "master": 2, "m.tech": 2, "mtech": 2, "m.e": 2, "me": 2,
    "m.sc": 2, "msc": 2, "mca": 2, "mba": 2, "postgraduate": 2,
    "phd": 3, "doctorate": 3, "ph.d": 3,
}


def _degree_level(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    for key, level in _DEGREE_LEVELS.items():
        if key in t:
            return level
    return None


_KNOWN_INSTITUTION_EXPANSIONS = {
    "iit":  "indian institute of technology",
    "nit":  "national institute of technology",
    "iiit": "indian institute of information technology",
    "bits": "birla institute of technology",
    "iisc": "indian institute of science",
    "iim":  "indian institute of management",
}

def _field_or_institution_match(candidate_value: str, required_list: list[str]) -> bool:
    if not candidate_value or not required_list:
        return False
    cv = candidate_value.lower()
    for req in required_list:
        req_clean = req.strip().lower()
        if req_clean in cv:
            return True
        expansion = _KNOWN_INSTITUTION_EXPANSIONS.get(req_clean)
        if expansion and expansion in cv:
            return True
    return False


def _select_best_education_entry(candidate_education: list[dict], min_degree_req: Optional[str]) -> dict:
    if len(candidate_education) == 1:
        return candidate_education[0]

    req_level = _degree_level(min_degree_req) if min_degree_req else None

    def sort_key(entry):
        level = _degree_level(entry.get("degree") or "")
        return level if level is not None else -1

    best = max(candidate_education, key=sort_key)
    if req_level is not None and sort_key(best) < req_level:
        for entry in candidate_education:
            if sort_key(entry) >= req_level:
                    return entry
    return best


def _any_entry_matches(candidate_education: list[dict], preferred_list: list[str], key: str) -> tuple:
    for entry in candidate_education:
        value = entry.get(key) or ""
        degree_value = entry.get("degree") or ""
        if _field_or_institution_match(value, preferred_list) or _field_or_institution_match(degree_value, preferred_list):
            return (True, value or degree_value)
    return (False, None)


def _deterministic_education_check(profile: dict, criteria: dict) -> Optional[tuple]:
    education_req = criteria.get("education", {})
    if not _has_education_requirement(education_req):
        return None

    candidate_education = profile.get("education", [])
    if not candidate_education:
        return (False, "No education records found on the resume to verify the requirement.")

    min_degree_req   = education_req.get("minimum_degree")
    preferred_fields = education_req.get("preferred_fields", [])
    preferred_insts  = education_req.get("preferred_institutions", [])
    min_gpa          = education_req.get("minimum_gpa")

    best_entry = _select_best_education_entry(candidate_education, min_degree_req)
    degree  = best_entry.get("degree") or ""
    gpa_raw = best_entry.get("gpa")

    if min_degree_req:
        req_level  = _degree_level(min_degree_req)
        cand_level = _degree_level(degree)
        if req_level and cand_level is not None and cand_level < req_level:
            return (False, f"Candidate's highest degree ({degree or 'unspecified'}) is below the minimum required degree ({min_degree_req}).")

    institution_match, institution_value = _any_entry_matches(candidate_education, preferred_insts, "institution")
    field_match, field_value             = _any_entry_matches(candidate_education, preferred_fields, "field")
    institution = institution_value or (best_entry.get("institution") or "")
    field       = field_value or (best_entry.get("field") or "")

    gpa_result = None
    if min_gpa:
        parsed = _parse_gpa_value(gpa_raw)
        if parsed is not None:
            value, is_percentage = parsed
            if is_percentage:
                estimated_cgpa = round(value / 9.5, 2)
                gpa_result = estimated_cgpa >= min_gpa
                gpa_note = f"GPA {value}% (~{estimated_cgpa}/10 estimated) vs required {min_gpa}"
            else:
                gpa_result = value >= min_gpa
                gpa_note = f"GPA {value} vs required {min_gpa}"
        else:
            gpa_note = "GPA not disclosed on resume"
    else:
        gpa_note = None

    if gpa_result is False:
        return (False, f"{gpa_note}. Does not meet the minimum GPA requirement, regardless of institution or field.")

    if gpa_result is True:
        note = f"{gpa_note}. Meets the GPA requirement."
        if institution_match:
            note += f" Also attended a preferred institution ({institution})."
        if field_match:
            note += f" Field of study ({field or degree}) matches preferred fields."
        return (True, note)

    if institution_match or field_match:
        reasons = []
        if institution_match:
            reasons.append(f"institution ({institution}) is in the preferred list")
        if field_match:
            reasons.append(f"field of study ({field or degree}) matches preferred fields")
        gpa_part = f" {gpa_note}." if gpa_note else ""
        return (True, f"Candidate's {' and '.join(reasons)}.{gpa_part} GPA requirement could not be independently verified but other education criteria are satisfied.")

    gpa_part = f" {gpa_note}." if gpa_note else ""
    return (False, f"Candidate's institution and field of study do not match the preferred criteria.{gpa_part} Could not verify GPA either.")


def _collect_candidate_searchable_text(profile: dict) -> str:
    parts = []
    skills = profile.get("skills", {})
    parts.extend(skills.get("technical", []))
    parts.extend(skills.get("tools", []))
    parts.extend(skills.get("soft", []))
    parts.extend(profile.get("certifications", []))
    for exp in profile.get("experience", []):
        if exp.get("title"):
            parts.append(exp["title"])
        parts.extend(exp.get("responsibilities", []))
        parts.extend(exp.get("achievements", []))
    for proj in profile.get("projects", []):
        if proj.get("name"):
            parts.append(proj["name"])
        if proj.get("description"):
            parts.append(proj["description"])
        parts.extend(proj.get("tech_used", []))
    return " ".join(parts).lower()


_GENERIC_SKILL_WORDS = {
    "architecture", "design", "pipelines", "pipeline", "skills", "skill",
    "development", "framework", "frameworks", "technologies", "technology",
    "tools", "tool", "systems", "system", "services", "service",
    "engineering", "management", "administration", "platform", "platforms",
}


def _core_tokens(skill: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", skill.lower())
    tokens = [t for t in cleaned.split() if t]
    core = [t for t in tokens if t not in _GENERIC_SKILL_WORDS]
    return core if core else tokens


def _skill_present_in_text(skill: str, searchable: str) -> bool:
    tokens = _core_tokens(skill)
    if not tokens:
        return False
    for token in tokens:
        if not re.search(r"\b" + re.escape(token) + r"\b", searchable):
            return False
    return True


def _validate_skill_matches(skill_matches: list[str], skill_gaps: list[str], profile: dict) -> tuple:
    searchable = _collect_candidate_searchable_text(profile)

    verified_matches = []
    demoted_to_gaps = []
    for skill in skill_matches:
        if _skill_present_in_text(skill, searchable):
            verified_matches.append(skill)
        else:
            demoted_to_gaps.append(skill)
            logger.warning(f"Skill '{skill}' was claimed as matched but not found in candidate data. Moving to gaps.")

    promoted_to_matches = []
    final_gaps = []
    for skill in skill_gaps:
        if skill in verified_matches or skill in promoted_to_matches:
            continue
        if _skill_present_in_text(skill, searchable):
            promoted_to_matches.append(skill)
            logger.warning(f"Skill '{skill}' was claimed as a gap but found in candidate data. Moving to matches.")
        else:
            final_gaps.append(skill)

    for skill in demoted_to_gaps:
        if skill not in final_gaps:
            final_gaps.append(skill)

    final_matches = verified_matches + [s for s in promoted_to_matches if s not in verified_matches]
    return final_matches, final_gaps


def _build_candidate_text(profile: dict, criteria: dict) -> str:
    parts = []

    summary = profile.get("summary")
    if summary:
        parts.append(summary)

    skills = profile.get("skills", {})
    technical = skills.get("technical", [])
    tools     = skills.get("tools", [])
    if technical:
        parts.append("Technical skills: " + ", ".join(technical))
    if tools:
        parts.append("Tools: " + ", ".join(tools))

    certifications = profile.get("certifications", [])
    if certifications:
        parts.append("Certifications: " + ", ".join(certifications))

    if _has_experience_requirement(criteria):
        total_years = profile.get("total_experience_years")
        if total_years:
            parts.append(f"Total experience: {total_years} years")
        for exp in profile.get("experience", []):
            title   = exp.get("title")
            company = exp.get("company")
            if title:
                line = title
                if company:
                    line += f" at {company}"
                parts.append(line)
            responsibilities = exp.get("responsibilities", [])
            if responsibilities:
                parts.append(" ".join(responsibilities[:5]))

    if _has_education_requirement(criteria.get("education", {})):
        for edu in profile.get("education", []):
            degree      = edu.get("degree")
            field       = edu.get("field")
            institution = edu.get("institution")
            gpa         = edu.get("gpa")
            if degree or institution:
                line = "Education: "
                if degree:
                    line += degree
                if field:
                    line += f" in {field}"
                if institution:
                    line += f" from {institution}"
                if gpa:
                    line += f" with GPA {gpa}"
                parts.append(line)

    if _has_location_requirement(criteria):
        location = profile.get("contact", {}).get("location")
        if location:
            parts.append(f"Location: {location}")

    if not parts:
        raise ValueError("parsed_profile has no embeddable content for this JD.")

    text = ". ".join(p.strip() for p in parts if p.strip())
    logger.debug(f"Candidate text ({len(text)} chars): {text[:100]}...")
    return text


def _build_jd_text(criteria: dict) -> str:
    parts = []

    role_type = criteria.get("role_type")
    seniority = criteria.get("seniority")
    if role_type:
        parts.append(f"Role: {role_type}")
    if seniority:
        parts.append(f"Seniority: {seniority}")

    skills = criteria.get("skills", {})
    required  = skills.get("required", [])
    preferred = skills.get("preferred", [])
    if required:
        parts.append("Required skills: " + ", ".join(required))
    if preferred:
        parts.append("Preferred skills: " + ", ".join(preferred))

    experience_years = criteria.get("experience_years")
    if experience_years:
        parts.append(f"Minimum experience: {experience_years} years")

    responsibilities = criteria.get("responsibilities", [])
    if responsibilities:
        parts.append("Responsibilities: " + ". ".join(responsibilities[:5]))

    education = criteria.get("education", {})
    has_edu_req = _has_education_requirement(education)
    if has_edu_req:
        min_degree   = education.get("minimum_degree")
        pref_fields  = education.get("preferred_fields", [])
        min_gpa      = education.get("minimum_gpa")
        pref_instits = education.get("preferred_institutions", [])
        is_mandatory = education.get("is_mandatory", False)

        line = "Education requirement: "
        if min_degree:
            line += f"minimum degree {min_degree}"
        if pref_fields:
            line += f", preferred fields {', '.join(pref_fields)}"
        if min_gpa:
            line += f", minimum GPA {min_gpa}"
        if pref_instits:
            line += f", preferred institutions {', '.join(pref_instits)}"
        if is_mandatory:
            line += " (mandatory)"
        parts.append(line)

    if _has_location_requirement(criteria):
        location  = criteria.get("location")
        work_mode = criteria.get("work_mode")
        if location:
            parts.append(f"Location: {location}")
        if work_mode:
            parts.append(f"Work mode: {work_mode}")

    if not parts:
        raise ValueError("structured_criteria has no embeddable content.")

    text = ". ".join(p.strip() for p in parts if p.strip())
    logger.debug(f"JD text ({len(text)} chars): {text[:100]}...")
    return text, has_edu_req


def semantic_match(
    parsed_profile:      dict,
    structured_criteria: dict,
) -> SemanticMatchResult:
    candidate_text = _build_candidate_text(parsed_profile, structured_criteria)
    jd_text, has_edu_req = _build_jd_text(structured_criteria)

    prompt = _load_prompt(candidate_text, jd_text, has_edu_req)

    logger.info("Calling Qwen for semantic match analysis...")
    result, confidence = call_ollama_for_schema(
        prompt        = prompt,
        system_prompt = SYSTEM_PROMPT,
        schema        = SemanticMatchResult,
        context_label = "semantic matching",
    )

    result.semantic_score       = max(SCORE_FLOOR, min(100, result.semantic_score))
    result.experience_relevance = max(SCORE_FLOOR, min(100, result.experience_relevance))

    verified_matches, verified_gaps = _validate_skill_matches(
        result.skill_matches, result.skill_gaps, parsed_profile
    )
    result.skill_matches = verified_matches
    result.skill_gaps     = verified_gaps

    overlap = set(result.skill_matches) & set(result.skill_gaps)
    if overlap:
        result.skill_gaps = [s for s in result.skill_gaps if s not in overlap]

    if has_edu_req:
        deterministic = _deterministic_education_check(parsed_profile, structured_criteria)
        if deterministic is not None:
            meets, note = deterministic
            if result.education_match != meets:
                logger.warning(
                    f"Qwen said education_match={result.education_match}, "
                    f"deterministic check says {meets}. Overriding with deterministic result."
                )
            result.education_match = meets
            result.education_notes = f"[Verified] {note}"

    if result.semantic_score <= SCORE_FLOOR:
        logger.warning(
            f"Semantic score hit the floor ({SCORE_FLOOR}). "
            f"Gaps: {result.skill_gaps}"
        )

    logger.info(
        f"Semantic match done — score: {result.semantic_score}, "
        f"matches: {len(result.skill_matches)}, gaps: {len(result.skill_gaps)}, "
        f"transferable: {result.is_transferable}, education_match: {result.education_match}"
    )
    return result


def _load_prompt(candidate_text: str, jd_text: str, has_edu_req: bool) -> str:
    path = Path(SEMANTIC_MATCHING_PROMPT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic matching prompt not found at {path}. "
            "Make sure data/prompts/semantic_matching_v1.txt exists."
        )
    template = path.read_text(encoding="utf-8")

    education_instruction = (
        "This job has specific education requirements. Evaluate the candidate's degree, "
        "field of study, institution, and GPA against the requirement. "
        "Set education_match to true if the candidate meets or exceeds the requirement, "
        "false if they do not. Explain your decision in education_notes."
        if has_edu_req else
        "This job has no specific education requirement. Set education_match to null "
        "and education_notes to null."
    )

    return (
        template
        .replace("{candidate_text}", candidate_text)
        .replace("{jd_text}", jd_text)
        .replace("{education_instruction}", education_instruction)
    )
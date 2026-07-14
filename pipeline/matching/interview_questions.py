import json
import logging
from pipeline.utils.ollama_client import call_ollama_for_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior technical interviewer with 15+ years of experience. "
    "You read resumes deeply and ask precise, probing questions that expose "
    "whether a candidate truly understands what they claim to have built. "
    "Never ask basic definitions. Always anchor questions to the candidate's "
    "actual experience, projects, and claimed achievements. "
    "Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text."
)


class QuestionBank(BaseModel):
    technical_depth:          list[str] = []
    skill_gap_probing:        list[str] = []
    experience_leadership:    list[str] = []
    culture_problem_solving:  list[str] = []

    class Config:
        extra = "ignore"


def _format_experience(experience_entries: list) -> str:
    if not experience_entries:
        return "No experience entries extracted."
    lines = []
    for e in experience_entries:
        company  = e.get("company") or "Unknown company"
        title    = e.get("title") or "Unknown role"
        start    = e.get("start_date") or ""
        end      = e.get("end_date") or "Present"
        duration = f"{e.get('duration_months', '')} months" if e.get("duration_months") else ""
        resp     = e.get("responsibilities") or []
        achieve  = e.get("achievements") or []
        lines.append(f"• {title} at {company} ({start}–{end} {duration})".strip())
        if resp:
            lines.append(f"  Responsibilities: {'; '.join(resp[:4])}")
        if achieve:
            lines.append(f"  Achievements: {'; '.join(achieve[:3])}")
    return "\n".join(lines)


def _format_projects(project_entries: list) -> str:
    if not project_entries:
        return "No projects extracted."
    lines = []
    for p in project_entries:
        name  = p.get("name") or "Unnamed project"
        desc  = p.get("description") or ""
        techs = p.get("tech_used") or []
        lines.append(f"• {name}")
        if desc:
            lines.append(f"  Description: {desc[:200]}")
        if techs:
            lines.append(f"  Technologies: {', '.join(techs)}")
    return "\n".join(lines)


def generate_interview_questions(candidate: dict, jd_criteria: dict) -> dict:
    name             = candidate.get("name") or "the candidate"
    skill_matches    = candidate.get("skill_matches") or []
    skill_gaps       = candidate.get("skill_gaps") or []
    experience_score = candidate.get("experience_score", 0)
    reasoning        = candidate.get("reasoning") or ""
    role_type        = jd_criteria.get("role_type") or "the role"
    seniority        = jd_criteria.get("seniority") or ""
    required_skills  = (jd_criteria.get("skills") or {}).get("required") or []
    experience_years = jd_criteria.get("experience_years")

    experience_entries = candidate.get("experience") or []
    project_entries    = candidate.get("projects") or []

    experience_level = (
        "junior"    if experience_score < 40 else
        "mid-level" if experience_score < 70 else
        "senior"
    )

    experience_text = _format_experience(experience_entries)
    projects_text   = _format_projects(project_entries)

    prompt = f"""You are interviewing {name} for a {seniority} {role_type} role requiring {experience_years}+ years.

CANDIDATE PROFILE:
- Confirmed skills: {', '.join(skill_matches) if skill_matches else 'None verified'}
- Skill gaps: {', '.join(skill_gaps) if skill_gaps else 'None'}
- Experience level: {experience_level} (score {experience_score}/100)

WORK EXPERIENCE:
{experience_text}

PROJECTS:
{projects_text}

AI ASSESSMENT: {reasoning}

JOB REQUIREMENTS:
- Role: {seniority} {role_type}
- Required: {', '.join(required_skills) if required_skills else 'Not specified'}
- Experience: {experience_years} years

RULES:
- Every question must reference something SPECIFIC from the candidate's actual experience, projects, or claimed skills above.
- NEVER ask generic definitional questions like "What is a REST API?" or "Explain microservices."
- Anchor every question to real details: company names, project names, specific technologies, durations, responsibilities.
- Questions must be hard enough that only someone who genuinely did the work can answer well.
- For skill gaps, probe transferable knowledge — not "do you know X?" but what adjacent depth they have.

Return ONLY this JSON with exactly 5+5+3+2 questions:

{{
  "technical_depth": [
    "question 1",
    "question 2",
    "question 3",
    "question 4",
    "question 5"
  ],
  "skill_gap_probing": [
    "question 1",
    "question 2",
    "question 3",
    "question 4",
    "question 5"
  ],
  "experience_leadership": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "culture_problem_solving": [
    "question 1",
    "question 2"
  ]
}}"""

    logger.info(f"Generating interview questions for {name} ({len(experience_entries)} jobs, {len(project_entries)} projects)...")

    try:
        result, confidence = call_ollama_for_schema(
            prompt        = prompt,
            system_prompt = SYSTEM_PROMPT,
            schema        = QuestionBank,
            context_label = "interview questions",
        )

        sections = {
            "technical_depth":         result.technical_depth,
            "skill_gap_probing":       result.skill_gap_probing,
            "experience_leadership":   result.experience_leadership,
            "culture_problem_solving": result.culture_problem_solving,
        }

        total = sum(len(v) for v in sections.values())
        logger.info(f"Generated {total} questions for {name} (confidence {confidence}).")

        return {
            "candidate_name":  name,
            "role":            f"{seniority} {role_type}".strip(),
            "sections":        sections,
            "total_questions": total,
        }

    except Exception as e:
        logger.error(f"Interview question generation failed for {name}: {e}")
        raise
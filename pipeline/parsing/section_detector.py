import re
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)
class SectionType(str, Enum):
    SUMMARY        = "summary"
    SKILLS         = "skills"
    EXPERIENCE     = "experience"
    EDUCATION      = "education"
    PROJECTS       = "projects"
    CERTIFICATIONS = "certifications"
    LANGUAGES      = "languages"
    AWARDS         = "awards"
    OTHER          = "other"
    HEADER         = "header"   

_SECTION_KEYWORDS: dict[SectionType, list[str]] = {
    SectionType.SUMMARY: [
        "summary", "profile", "objective", "about me", "about",
        "professional summary", "career objective", "overview",
        "personal statement", "introduction",
    ],
    SectionType.SKILLS: [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tech stack", "tools", "expertise",
        "areas of expertise", "key skills", "proficiencies",
        "technical expertise", "technical proficiencies",
    ],
    SectionType.EXPERIENCE: [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history",
        "career history", "internship", "internships",
        "industry experience", "relevant experience",
    ],
    SectionType.EDUCATION: [
        "education", "academic background", "academic qualifications",
        "qualifications", "academic history", "degrees",
        "educational background", "academics",
    ],
    SectionType.PROJECTS: [
        "projects", "personal projects", "academic projects",
        "side projects", "open source", "portfolio",
        "key projects", "notable projects",
    ],
    SectionType.CERTIFICATIONS: [
        "certifications", "certificates", "certification",
        "licenses", "accreditations", "professional certifications",
        "courses", "training", "professional development",
    ],
    SectionType.LANGUAGES: [
        "languages", "language skills", "spoken languages",
    ],
    SectionType.AWARDS: [
        "awards", "honors", "achievements", "accomplishments",
        "recognition", "publications", "research",
    ],
}

@dataclass
class Section:
    section_type: SectionType
    header_text:  str         
    content:      str         
    line_start:   int          
    line_end:     int          

@dataclass
class SectionResult:
    sections:       list[Section]
    sections_dict:  dict[str, str]  
    detected_types: list[str]        
    unmatched_lines: int           

def detect_sections(clean_text: str) -> SectionResult:
    lines    = clean_text.split("\n")
    sections = []

    current_type    = SectionType.HEADER
    current_header  = "HEADER"
    current_lines   = []
    current_start   = 0

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        detected_type = _detect_header(stripped)

        if detected_type is not None:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(Section(
                        section_type = current_type,
                        header_text  = current_header,
                        content      = content,
                        line_start   = current_start,
                        line_end     = line_num - 1,
                    ))

            current_type   = detected_type
            current_header = stripped
            current_lines  = []
            current_start  = line_num

        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(Section(
                section_type = current_type,
                header_text  = current_header,
                content      = content,
                line_start   = current_start,
                line_end     = len(lines) - 1,
            ))

    sections_dict: dict[str, str] = {}
    for section in sections:
        key = section.section_type.value
        if key in sections_dict:
            sections_dict[key] += "\n" + section.content
        else:
            sections_dict[key] = section.content

    detected_types  = list({s.section_type.value for s in sections
                            if s.section_type != SectionType.OTHER})
    unmatched_lines = sum(
        len(s.content.split("\n")) for s in sections
        if s.section_type == SectionType.OTHER
    )

    logger.info(
        f"Section detection: found {len(sections)} sections — "
        f"{detected_types}. Unmatched lines: {unmatched_lines}"
    )

    return SectionResult(
        sections        = sections,
        sections_dict   = sections_dict,
        detected_types  = detected_types,
        unmatched_lines = unmatched_lines,
    )

def _detect_header(line: str) -> SectionType | None:
    if not line:
        return None

    if len(line) > 60:
        return None

    normalized = re.sub(r"[:\-_•\|]+$", "", line.lower()).strip()

    for section_type, keywords in _SECTION_KEYWORDS.items():
        for keyword in keywords:
            if normalized == keyword or normalized == keyword + ":":
                return section_type
            if normalized.startswith(keyword) and len(normalized) <= len(keyword) + 3:
                return section_type
    if line.isupper() and 2 <= len(line.split()) <= 4 and len(line) <= 40:
        for section_type, keywords in _SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword.upper() in line:
                    return section_type
        return SectionType.OTHER   

    return None

def get_section(result: SectionResult, section_type: SectionType) -> str:
    return result.sections_dict.get(section_type.value, "")

def get_sections_for_llm(result: SectionResult) -> str:
    parts = []
    priority_order = [
        SectionType.SUMMARY,
        SectionType.SKILLS,
        SectionType.EXPERIENCE,
        SectionType.EDUCATION,
        SectionType.PROJECTS,
        SectionType.CERTIFICATIONS,
        SectionType.AWARDS,
        SectionType.LANGUAGES,
    ]

    for section_type in priority_order:
        content = result.sections_dict.get(section_type.value, "")
        if content.strip():
            label = section_type.value.upper()
            parts.append(f"=== {label} ===\n{content.strip()}")

    other_content = result.sections_dict.get(SectionType.OTHER.value, "")
    if other_content.strip():
        parts.append(f"=== OTHER ===\n{other_content.strip()}")

    return "\n\n".join(parts)
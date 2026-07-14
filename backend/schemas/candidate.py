from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator

def _coerce_to_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for sep in ["\n", ";", "|", ",", " - ", " • "]:
            if sep in s:
                return [part.strip() for part in s.split(sep) if part.strip()]
        return [s]
    if isinstance(v, list):
        return [str(item).strip() for item in v if str(item).strip()]
    return [str(v)]


class ContactInfo(BaseModel):
    name:     Optional[str]   = None
    email:    Optional[str]   = None
    phone:    Optional[str]   = None
    location: Optional[str]   = None
    linkedin: Optional[str]   = None
    github:   Optional[str]   = None


class Skills(BaseModel):
    technical: list[str] = []
    tools:     list[str] = []
    soft:      list[str] = []

    @field_validator("technical", "tools", "soft", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []


class ExperienceEntry(BaseModel):
    company:          Optional[str]       = None
    title:            Optional[str]       = None
    location:         Optional[str]       = None
    start_date:       Optional[str]       = None
    end_date:         Optional[str]       = None
    duration_months:  Optional[int]       = None
    responsibilities: list[str]           = []
    achievements:     list[str]           = []
    is_current:       bool                = False

    @field_validator("responsibilities", "achievements", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return _coerce_to_list(v)

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if v is None or isinstance(v, int):
            return v
        if isinstance(v, str):
            import re
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        return v


class EducationEntry(BaseModel):
    degree:          Optional[str] = None
    field:           Optional[str] = None
    institution:     Optional[str] = None
    graduation_year: Optional[int] = None
    gpa:             Optional[str] = None

    @field_validator("graduation_year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        if v is None or isinstance(v, int):
            return v
        if isinstance(v, str):
            import re
            m = re.search(r"(19|20)\d{2}", v)
            return int(m.group()) if m else None
        return v

    @field_validator("gpa", mode="before")
    @classmethod
    def coerce_gpa(cls, v):
        if v is None:
            return None
        return str(v)


class ProjectEntry(BaseModel):
    name:        Optional[str] = None
    description: Optional[str] = None
    tech_used:   list[str]     = []
    url:         Optional[str] = None

    @field_validator("tech_used", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return _coerce_to_list(v)


class ParsedProfile(BaseModel):
    contact: ContactInfo = ContactInfo()
    summary:          Optional[str]         = None
    skills:           Skills                = Skills()
    experience:       list[ExperienceEntry] = []
    education:        list[EducationEntry]  = []
    projects:         list[ProjectEntry]    = []
    certifications:   list[str]             = []
    total_experience_years: Optional[float] = None
    raw_sections: dict[str, str] = {}
    parser_version: str  = "v1"
    extraction_confidence: Optional[float] = None

    @field_validator("certifications", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return _coerce_to_list(v)

    @field_validator("total_experience_years", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None or isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            import re
            m = re.search(r"\d+(\.\d+)?", v)
            return float(m.group()) if m else None
        return v

    class Config:
        extra = "allow"


class CandidateResponse(BaseModel):
    id:                str
    original_filename: str
    status:            str
    parser_version:    str
    parsed_profile:    Optional[ParsedProfile] = None

    class Config:
        from_attributes = True
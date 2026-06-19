from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator

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

class EducationEntry(BaseModel):
    degree:          Optional[str] = None   
    field:           Optional[str] = None  
    institution:     Optional[str] = None
    graduation_year: Optional[int] = None
    gpa:             Optional[str] = None  

class ProjectEntry(BaseModel):
    name:        Optional[str] = None
    description: Optional[str] = None
    tech_used:   list[str]     = []
    url:         Optional[str] = None  

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
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, field_validator

from config.pipeline_config import (
    DEFAULT_WEIGHT_SKILLS, DEFAULT_WEIGHT_EXPERIENCE,
    DEFAULT_WEIGHT_LLM, DEFAULT_WEIGHT_EDUCATION,
)

class SkillRequirement(BaseModel):
    required:  list[str] = []  
    preferred: list[str] = []  

    @field_validator("required", "preferred", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []


class EducationRequirement(BaseModel):
    minimum_degree: Optional[str]       = None  
    preferred_fields: list[str]         = []    
    is_mandatory:     bool              = False 

class StructuredCriteria(BaseModel):

    skills:            SkillRequirement     = SkillRequirement()
    experience_years:  Optional[float]      = None  
    education:         EducationRequirement = EducationRequirement()

    role_type:         Optional[str]        = None   
    seniority:         Optional[str]        = None   
    responsibilities:  list[str]            = []     
    location:          Optional[str]        = None
    work_mode:         Optional[str]        = None  
    scoring_weights: ScoringWeights = None  
    extraction_confidence: Optional[float] = None  
    jd_word_count:         Optional[int]   = None   

    class Config:
        extra = "allow"


class ScoringWeights(BaseModel):
    skills:     float = DEFAULT_WEIGHT_SKILLS  
    experience: float = DEFAULT_WEIGHT_EXPERIENCE   
    llm:        float = DEFAULT_WEIGHT_LLM 
    education:  float = DEFAULT_WEIGHT_EDUCATION  

    @field_validator("skills", "experience", "llm", "education", mode="after")
    @classmethod
    def weights_valid(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v

    def validate_sum(self) -> bool:
        total = self.skills + self.experience + self.llm + self.education
        return abs(total - 1.0) < 0.001


StructuredCriteria.model_fields["scoring_weights"].default_factory = ScoringWeights 

class JobResponse(BaseModel):
    id:                 str
    title:              str
    department:         Optional[str]
    location:           Optional[str]
    status:             str
    structured_criteria: Optional[StructuredCriteria] = None

    class Config:
        from_attributes = True
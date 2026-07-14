from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, field_validator
from config.pipeline_config import (
    DEFAULT_WEIGHT_SKILLS, DEFAULT_WEIGHT_EXPERIENCE,
    DEFAULT_WEIGHT_LLM,
)

# Valid education requirement levels
EducationLevel = Literal[
    "none",
    "diploma_preferred",
    "diploma_required",
    "bachelors_preferred",
    "bachelors_required",
    "masters_preferred",
    "masters_required",
    "phd_preferred",
    "phd_required",
]


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
    minimum_degree:         Optional[str]   = None
    preferred_fields:       list[str]       = []
    is_mandatory:           bool            = False
    minimum_gpa:            Optional[float] = None
    preferred_institutions: list[str]       = []

    @field_validator("minimum_gpa", mode="before")
    @classmethod
    def coerce_gpa(cls, v):
        if v is None or isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            import re
            m = re.search(r"\d+(\.\d+)?", v)
            return float(m.group()) if m else None
        return v

    @field_validator("preferred_institutions", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []


class ScoringWeights(BaseModel):
    # Only 3 sliders now — education is a separate gate
    skills:                float          = DEFAULT_WEIGHT_SKILLS
    experience:            float          = DEFAULT_WEIGHT_EXPERIENCE
    llm:                   float          = DEFAULT_WEIGHT_LLM
    education_requirement: EducationLevel = "none"

    @field_validator("skills", "experience", "llm", mode="after")
    @classmethod
    def weights_valid(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v

    def validate_sum(self) -> bool:
        total = self.skills + self.experience + self.llm
        return abs(total - 1.0) < 0.001


class StructuredCriteria(BaseModel):
    skills:                SkillRequirement     = SkillRequirement()
    experience_years:      Optional[float]      = None
    education:             EducationRequirement = EducationRequirement()
    role_type:             Optional[str]        = None
    seniority:             Optional[str]        = None
    responsibilities:      list[str]            = []
    location:              Optional[str]        = None
    work_mode:             Optional[str]        = None
    scoring_weights:       ScoringWeights       = None
    extraction_confidence: Optional[float]      = None
    jd_word_count:         Optional[int]        = None

    class Config:
        extra = "allow"


StructuredCriteria.model_fields["scoring_weights"].default_factory = ScoringWeights


class JobResponse(BaseModel):
    id:                  str
    title:               str
    department:          Optional[str]
    location:            Optional[str]
    status:              str
    structured_criteria: Optional[StructuredCriteria] = None

    class Config:
        from_attributes = True
from typing import Optional
from pydantic import BaseModel


class MatchResult(BaseModel):
    candidate_id:         str
    job_id:               str

    vector_similarity:    float          
    semantic_score:       int            
    skill_matches:        list[str] = []
    skill_gaps:           list[str] = []
    experience_relevance: int            
    reasoning:            str
    is_transferable:      bool = False
    confidence:           Optional[float] = None

    class Config:
        extra = "allow"
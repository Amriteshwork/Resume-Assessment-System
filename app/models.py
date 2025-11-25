from typing import Dict, Any
from pydantic import BaseModel
from typing_extensions import TypedDict


class AssessmentRequest(BaseModel):
    jd_text: str

class AssessmentResponse(BaseModel):
    overall_score: float
    skills_score: float
    experience_score: float
    seniority_score: float
    assessment_text: str

# LangGraph state
class AgentState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    resume_structured: Dict[str, Any]
    jd_structured: Dict[str, Any]
    scores: Dict[str, float]
    guidelines: str
    assessment_text: str
    cleaned_assessment_text: str
    errors: str
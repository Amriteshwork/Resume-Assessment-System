from typing import Dict, Any

from .tools import (
    extract_resume_structured,
    extract_jd_structured,
    compute_scores,
    generate_assessment,
    mask_pii,
    save_assessment_to_db,
)


class BaseAgent:
    """Base class just for clarity and docs - no heavy framework."""
    name: str
    description: str

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class ResumeParserAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ResumeParserAgent",
            description="Extracts structured info from raw resume text.",
        )

    def run(self, resume_text: str) -> Dict[str, Any]:
        return extract_resume_structured(resume_text)


class JDParserAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="JDParserAgent",
            description="Extracts structured requirements from job descriptions.",
        )

    def run(self, jd_text: str) -> Dict[str, Any]:
        return extract_jd_structured(jd_text)


class ScoringAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ScoringAgent",
            description="Computes objective scores based on resume/JD alignment.",
        )

    def run(self, resume_struct: Dict[str, Any], jd_struct: Dict[str, Any]) -> Dict[str, float]:
        return compute_scores(resume_struct, jd_struct)


class ReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ReviewerAgent",
            description="Writes human-readable assessment grounded in scores and RAG guidelines.",
        )

    def run(self, resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], scores: Dict[str, float]) -> str:
        return generate_assessment(resume_struct, jd_struct, scores)


class SafetyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="SafetyAgent",
            description="Guardrails (PII masking) and persists the assessment.",
        )

    def run(
        self,
        resume_struct: Dict[str, Any],
        jd_struct: Dict[str, Any],
        scores: Dict[str, float],
        assessment_text: str,
    ) -> str:
        cleaned = mask_pii(assessment_text)
        save_assessment_to_db(resume_struct, jd_struct, scores, cleaned)
        return cleaned


resume_parser_agent = ResumeParserAgent()
jd_parser_agent = JDParserAgent()
scoring_agent = ScoringAgent()
reviewer_agent = ReviewerAgent()
safety_agent = SafetyAgent()

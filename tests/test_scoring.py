import pytest

from app.tools import compute_scores


def test_compute_scores_overlap_and_formula():
    def fake_llm(prompt: str):
        
        return {"relevant_years": 5.0, "seniority_fit": 1.0}

    resume = {"skills": ["Python", "SQL", "FastAPI"]}
    jd = {"required_skills": ["Python", "FastAPI", "Docker"]}

    scores = compute_scores(resume, jd, llm_json_fn=fake_llm)

    assert scores["skills_score"] == pytest.approx(2 / 3, rel=1e-3) # skills: 2 / 3 overlap

    assert scores["experience_score"] == pytest.approx(1.0) # relevant_years=5 => experience_score capped at 1.0

    assert scores["seniority_score"] == pytest.approx(1.0) # seniority_fit from fake_llm

    expected_overall = 0.5 * (2 / 3) + 0.3 * 1.0 + 0.2 * 1.0
    assert scores["overall_score"] == pytest.approx(expected_overall, rel=1e-3)

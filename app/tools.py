import re
import json
from io import BytesIO
from typing import Dict, Any

from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

from openai import OpenAI
from .config import settings
from .rag import rag_retriever
from .db import SessionLocal, Assessment

# Initialize client
client = OpenAI(api_key=settings.openai_api_key)

# Parsing
def parse_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception as e:
        return f"Error parsing PDF: {e}"

def parse_docx(file_bytes: bytes) -> str:
    try:
        mem_file = BytesIO(file_bytes)
        doc = DocxDocument(mem_file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"Error parsing DOCX: {e}"

def parse_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "[OCR found no text]"
    except ImportError:
        return "[OCR skipped: pytesseract library not installed]"
    except Exception as e:
        if "tesseract is not installed" in str(e).lower() or "not found" in str(e).lower():
             return "[OCR skipped: Tesseract binary not found on system. Please install Tesseract-OCR.]"
        return f"[OCR Error: {e}]"

def parse_resume_text(file_bytes: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif fname.endswith(".docx"):
        return parse_docx(file_bytes)
    elif fname.endswith((".png", ".jpg", ".jpeg")):
        return parse_image(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="ignore") # treat as plain text

# LLM helpers
def llm_json_system_prompt() -> str:
    return "You are a helpful assistant. Always respond with valid JSON only, no extra text."

def call_llm_json(prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
    if not settings.openai_api_key:
        return {"error": "OpenAI API Key missing"}

    sys = system_prompt or llm_json_system_prompt()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {}

# Extraction
def extract_resume_structured(resume_text: str) -> Dict[str, Any]:
    prompt = f"""
    Extract structured information from this resume text.
    Return JSON with keys:
    - name: string
    - email: string
    - skills: list of strings
    - experience: list of objects with fields (title, company, years, description)
    - education: list of objects with fields (degree, institution, year)

    Resume:
    {resume_text[:4000]} 
    """
    return call_llm_json(
    prompt,
    system_prompt=(
        "You are a Resume Parsing Agent. "
        "Extract only the requested fields from resumes and respond with strict JSON."
    ),
)

def extract_jd_structured(jd_text: str) -> Dict[str, Any]:
    prompt = f"""
    Extract structured information from this job description.
    Return JSON with keys:
    - title: string
    - required_skills: list of strings
    - preferred_skills: list of strings
    - seniority_level: one of ["junior", "mid", "senior"]
    - summary: string

    Job Description:
    {jd_text[:40000]}
    """
    return call_llm_json(
    prompt,
    system_prompt=(
        "You are a Job Description Parsing Agent. "
        "Focus on role title, required_skills, preferred_skills, seniority_level, and summary. "
        "Respond with strict JSON."
    ),
)

# Scoring
def compute_scores(
    resume: Dict[str, Any],
    jd: Dict[str, Any],
    llm_json_fn=call_llm_json,
) -> Dict[str, float]:
    resume_skills = {str(s).lower().strip() for s in resume.get("skills", [])}
    jd_skills = {str(s).lower().strip() for s in jd.get("required_skills", []) if s}

    intersection = resume_skills & jd_skills
    skills_score = (len(intersection) / len(jd_skills)) if jd_skills else 0.0

    prompt = f"""
    Given this resume experience and job description, estimate:
    - relevant_years: number (float)
    - seniority_fit: 0.0 to 1.0 (float)

    Return JSON with keys: relevant_years, seniority_fit.

    Resume experience:
    {resume.get("experience", [])}

    Job description:
    {jd}
    """
    extra = llm_json_fn(prompt)
    relevant_years = float(extra.get("relevant_years", 0.0) or 0.0)
    seniority_fit = float(extra.get("seniority_fit", 0.5) or 0.5)

    experience_score = min(relevant_years / 5.0, 1.0)  # Assume 5 years is max score
    overall = 0.5 * skills_score + 0.3 * experience_score + 0.2 * seniority_fit

    return {
        "skills_score": round(skills_score, 3),
        "experience_score": round(experience_score, 3),
        "seniority_score": round(seniority_fit, 3),
        "overall_score": round(overall, 3),
    }

# Assessment with RAG
def generate_assessment(resume: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float]) -> str:
    guidelines = rag_retriever.retrieve("resume evaluation best practices")

    user_prompt = f"""
    You are an expert technical recruiter collaborating with other agents:
    - A Resume Parser Agent that produced the structured resume.
    - A JD Parser Agent that produced the structured JD.
    - A Scoring Agent that computed the numeric scores below.

    Your job is to explain and contextualize those numeric scores and give actionable feedback.

    Data:
    - Structured resume: {resume}
    - Structured JD: {jd}
    - Objective scores (0.0-1.0 unless otherwise noted): {scores}
    - Evaluation guidelines: {guidelines}

    Instructions:
    1. Structure your answer with the following **exact headings**:
       - "Overall fit"
       - "Skills analysis"
       - "Experience and seniority analysis"
       - "Suggestions for improvement"

    2. In the "Overall fit" section:
       - Explicitly mention the overall_score value (e.g., "overall_score = 0.72").
       - Explain in 2-3 sentences what mainly drives this score (skills, experience, seniority).

    3. In the "Skills analysis" section:
       - Explicitly mention skills_score.
       - Describe which key JD skills are present, partially present, or missing, using bullet points.
       - Keep the language neutral and evidence-based (avoid subjective adjectives like "amazing" or "terrible").

    4. In the "Experience and seniority analysis" section:
       - Explicitly mention experience_score and seniority_score.
       - Refer to the candidate's roles and durations when explaining these scores (e.g., "3 years as X, 2 years as Y").
       - If there is a mismatch with the JD's seniority_level, state it clearly but neutrally.

    5. In the "Suggestions for improvement" section:
       - Provide 3-5 concrete, actionable suggestions (e.g., skills to acquire, ways to highlight achievements).
       - Do not restate the scores; focus on actions.

    6. Do **not** invent skills or experience that are not reasonably implied by the structured data.

    Return plain Markdown suitable for display to a recruiter.

    """
    if not settings.openai_api_key:
        return "Assessment could not be generated (No API Key)."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fair, objective resume reviewer."},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# PII masking 
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}")

def mask_pii(text: str) -> str:
    if not text: return ""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text

# DB helpers
def save_assessment_to_db(resume: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float], assessment_text: str):
    session = SessionLocal()
    try:
        candidate_name = resume.get("name") or "Unknown"
        jd_title = jd.get("title") or "Unknown"

        record = Assessment(
            candidate_name=str(candidate_name),
            jd_title=str(jd_title),
            overall_score=scores.get("overall_score", 0.0),
            skills_score=scores.get("skills_score", 0.0),
            experience_score=scores.get("experience_score", 0.0),
            seniority_score=scores.get("seniority_score", 0.0),
            raw_assessment=assessment_text,
        )
        session.add(record)
        session.commit()
    except Exception as e:
        print(f"DB Error: {e}")
        session.rollback()
    finally:
        session.close()
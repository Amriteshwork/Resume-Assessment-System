from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import AssessmentResponse
from .graph import build_graph
from .tools import parse_resume_text
from .db import init_db

app = FastAPI(title="RESUME ASSESSMENT AGENT")

# CORS optional
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()
init_db()

@app.post("/assess_resume", response_model=AssessmentResponse)
async def assess_resume(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty resume file")

    resume_text = parse_resume_text(file_bytes, resume_file.filename)

    state = {
        "resume_text": resume_text,
        "jd_text": jd_text,
    }

    final_state = graph_app.invoke(state)

    scores = final_state.get("scores", {})
    cleaned_assessment = final_state.get("cleaned_assessment_text", "")

    disclaimer = (
        "\n\nDisclaimer: This assessment is AI-generated based only on the provided "
        "resume and job description. Use human judgment for final hiring decisions."
    )

    return AssessmentResponse(
        overall_score=scores.get("overall_score", 0.0),
        skills_score=scores.get("skills_score", 0.0),
        experience_score=scores.get("experience_score", 0.0),
        seniority_score=scores.get("seniority_score", 0.0),
        assessment_text=cleaned_assessment + disclaimer,
    )

@app.get("/")
def root():
    return {"message": "Resume assessment agent is running."}
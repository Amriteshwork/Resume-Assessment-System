# Multi-Agent Resume Assessment System (ML-Focused)

This project is a multi-agent, RAG-powered resume assessment system built for hiring workflows.

It ingests resumes in multiple formats, compares them to a given **Job Description** (JD), computes objective scores (skills, experience, seniority, overall), and generates a structured assessment with clear rationale for each score. The system is especially tuned for **machine learning roles** via role-specific guidelines.

---
## Project Structure

```
Resume-Assessment-System/
├── .env 
├── app
│   ├── agents.py
│   ├── config.py
│   ├── db.py
│   ├── graph.py
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── rag.py
│   ├── tools.py
│   └── ui.py
├── data
│   ├── guidelines_best_practices.md
│   └── guidelines_software_role.md   
├── tests
│   ├── conftest.py
│   ├── test_guardrails.py
│   ├── test_parsing_routing.py
│   └── test_scoring.py
├── README.md
├── requirements.txt
└── assessments.db  # SQLite DB (created at runtime)
```


## 1. Features

This project was built to demonstrate:

- **Multi-agent orchestration**  
  - Five specialized agents (parsing, JD parsing, scoring, reviewing, safety) coordinated with **LangGraph**.

- **Tooling & RAG**
  - Tools for parsing PDFs/DOCX/images, extracting structured fields, scoring alignment, generating assessments, masking PII, and saving to DB.
  - **RAG** over markdown guidelines using **FAISS + OpenAI embeddings** to ground assessments in internal best practices and ML-role guidelines.

- **Real-world integrations**
  - **FastAPI** endpoint for programmatic use.
  - **Gradio** UI for interactive demo.
  - **SQLite + SQLAlchemy** for persistent storage of assessments.
  - **Pytest** suite for core logic (parsing routing, guardrails, scoring).

- **Guardrails & safety**
  - PII masking for emails and phone numbers.
  - Neutral, objective evaluation style + disclaimers in responses.
  - Structured prompts to reduce hallucination and enforce JSON outputs where needed.

---

## 2. High-Level Architecture

The system is organized around:

1. **Agents (in `app/agents.py`)**
2. **A LangGraph workflow (in `app/graph.py`)**
3. **Tools/helpers (in `app/tools.py`, `app/rag.py`, `app/db.py`)**
4. **API/UI frontends (in `app/main.py`, `app/ui.py`)**

### 2.1 Multi-Agent Workflow

The assessment pipeline is modeled as five collaborating agents:

1. **ResumeParserAgent**
   - Input: raw resume text.
   - Output: structured resume JSON (`name`, `contact`, `skills`, `experience`, `education`, etc.).
   - Uses: `extract_resume_structured` tool.

2. **JDParserAgent**
   - Input: raw JD text.
   - Output: structured JD JSON (`title`, `required_skills`, `preferred_skills`, `seniority_level`, `summary`).
   - Uses: `extract_jd_structured` tool.

3. **ScoringAgent**
   - Input: structured resume + structured JD.
   - Output: `scores` dict:
     - `skills_score`
     - `experience_score`
     - `seniority_score`
     - `overall_score`
   - Uses: `compute_scores` tool, which:
     - Computes **skills overlap**.
     - Asks the LLM for `relevant_years` & `seniority_fit` (via JSON mode).
     - Uses a fixed formula:

        > $$
            \text{overall\_score} = 0.5 \cdot \text{skills\_score} + 0.3 \cdot \text{experience\_score} + 0.2 \cdot \text{seniority\_score}
          $$


4. **ReviewerAgent**
   - Input: structured resume + JD + scores.
   - Output: human-readable Markdown assessment.
   - Uses: `generate_assessment` tool, which:
     - Calls the LLM with:
       - structured data
       - scores
       - RAG-retrieved **general guidelines** and **ML-role guidelines**
     - Produces a report with these headings:
       - `Overall fit`
       - `Skills analysis`
       - `Experience and seniority analysis`
       - `Suggestions for improvement`
     - Explicitly ties each section back to the numeric scores.

5. **SafetyAgent**
   - Input: structured resume + JD + scores + raw assessment text.
   - Output: cleaned assessment text.
   - Uses:
     - `mask_pii` to redact emails and phone numbers.
     - `save_assessment_to_db` to persist the assessment in SQLite.

### 2.2 LangGraph StateGraph

The overall orchestration is defined in `app/graph.py` using LangGraph’s `StateGraph`:

```text
parse  →  score  →  assess  →  guardrail_and_save  →  END
```

The shared AgentState includes:
- `resume_text`, `jd_text`
- `resume_structured`, `jd_structured`
- `scores`
- `assessment_text`
- `cleaned_assessment_text`   

Each node runs one or more agents, updates the state, and passes it to the next node.

## 3. RAG & Guidelines
RAG is used to ground the assessment in internal guidelines:
- **Indexing** (`app/rag.py`)
  - Loads markdown files from `data/`:
    - `guidelines_best_practices.md` - general resume evaluation guidance.
    - `guidelines_ml_role.md` - focused guidelines for **machine learning roles**.
  - Chunked and embedded using OpenAI `text-embedding-3-small` (configurable).
  - Stored in an in-memory **FAISS** index.
- **Retrieval**
  - generate_assessment calls `rag_retriever.retrieve("resume evaluation best practices")` (and similar queries) to fetch relevant guideline snippets.
  - These snippets are included in the LLM prompt so that feedback follows your own policies instead of ad-hoc model behavior.

The ML guideline file is deliberately short, focusing on:
- Core ML skills (classical ML, DL frameworks).
- Data skills (Python + data stack, SQL).
- MLOps/production experience (serving, monitoring, retraining).
- Typical seniority patterns for ML roles (junior/mid/senior).
- Concrete suggestions for improvement (clarify models/metrics, show impact, highlight deployments).

## 4. Modules Overview

`app/config.py`   
Centralized configuration using Pydantic settings:
- `OPENAI_API_KEY`
- database URL (SQLite)
- embedding model name (`text-embedding-3-small` by default)
---

`app/db.py`   
Database setup using SQLAlchemy:
  - Engine + `SessionLocal`
  - Base models
  - `Assessment` table (stores candidate name, JD title, scores JSON, assessment text, timestamp)
  - `init_db()` to create tables
---
`app/models.py`   
Core tools used by the agents:
- **Parsing**
  - `parse_pdf`, `parse_docx`, `parse_image` (OCR) for resumes.
  - `parse_resume_text(file_bytes, filename)` routes based on file extension.
- **LLM JSON helper**   
  - `call_llm_json(prompt, system_prompt=None)`:
    - Uses `gpt-4o-mini` in JSON mode (`response_format={"type":"json_object"}`).
    - Used by parsing and scoring tools to get structured outputs.
- **Structured extraction**   
  - `extract_resume_structured(resume_text)`
  - `extract_jd_structured(jd_text)`
- **Scoring**
  - `compute_scores(resume_structured, jd_structured, llm_json_fn=call_llm_json):`
    - Computes skill overlap.
    - Calls the LLM for `relevant_years` and `seniority_fit`.
    - Outputs `skills_score`, `experience_score`, `seniority_score`, `overall_score`.
- **Assessment generation**   
  - `generate_assessment(resume_structured, jd_structured, scores)`:
    - Uses `gpt-4o` + RAG guidelines.
    - Produces structured Markdown explaining each score.
- **Guardrails & persistence**
  - `mask_pii(text)` – redact emails and phone numbers.
  - `save_assessment_to_db(...)` – persist assessment in SQLite.
---
`app/agents.py`   
Defines the five agents mentioned earlier, each wrapping the relevant tools and exposing a `.run(...)` method.

---
`app/main.py`
FastAPI service:
- Initializes DB and workflow on startup.
- Exposes a `POST /assess_resume` endpoint:
  - Accepts:
    - `resume_file` (file upload: PDF/DOCX/image/txt)
    - `jd_text` (job description, as text)

  - Pipeline:
    - Extract resume text with `parse_resume_text`.
    - Invoke the multi-agent workflow.
    - Return:
      - `scores`
      - `assessment` (cleaned, PII-masked)
      - `disclaimer`

---
`app/ui.py`   
Gradio interface:
- Upload a resume and paste a JD.
- On submit:
  - Runs the same workflow as the API.
  - Displays:
    - Assessment (Markdown)
    - Objective scores (JSON)

---
`data/`
Markdown guidelines used by the RAG retriever.

---
`tests/`
A small pytest suite to validate core logic:
- test_parsing_routing.py
  - Ensures `parse_resume_text` correctly routes based on file extension via monkeypatching.
- `test_guardrails.py`
  - Verifies `mask_pii` redacts emails and phone numbers correctly.
- `test_scoring.py`
  - Uses a fake LLM function to test:
    - skill overlap logic
    - overall scoring formula
    - independence from real API calls

---

## 6. Setup & Running

**NOTE**   
> Create a `.env` file in the project root with:   
> - OPENAI_API_KEY=your_key_here
> - EMBEDDING_MODEL=text-embedding-3-small


### 6.1 Requirements
- Python 3.10+ (tested with 3.11)
- An OpenAI API key
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set your OpenAI API key:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- (Optional) override embedding model:
  ```bash
  export EMBEDDING_MODEL="text-embedding-3-small"
  ```
### 6.2 Running the API (FastAPI)
From the project root:
```bash
uvicorn app.main:app --reload
```
Then open:
- Swagger docs: http://127.0.0.1:8000/docs   

Example request using curl:
```bash
curl -X POST "http://127.0.0.1:8000/assess_resume" \
  -F "resume_file=@path/to/resume.pdf" \
  -F "jd_text=We are hiring a Senior ML Engineer with strong Python, PyTorch, and MLOps experience."
```
You’ll get a JSON response:
```json
{
  "scores": {
    "skills_score": 0.67,
    "experience_score": 0.8,
    "seniority_score": 0.7,
    "overall_score": 0.75
  },
  "assessment": "## Overall fit\n...\n",
  "disclaimer": "This assessment is AI-generated and should not be used as the sole basis for hiring decisions."
}
```
### 6.3 Running the Gradio UI
From the project root:
```bash
python -m app.ui
```
Then open the printed URL (typically http://127.0.0.1:7860
) and interactively:
- Upload a resume (PDF/DOCX/image/txt).
- Paste a JD.
- See scores + assessment in the browser.

### 6.4 Running Tests
```bash
pytest
```
All tests should pass (core parsing, guardrails, scoring).

---
## 7. Limitations & Possible Extensions
Some known limitations and future ideas:

- **Bias/fairness auditing**:   
    Currently mitigated mainly via prompts and guidelines. A dedicated “BiasCheckAgent” could review assessments for potentially biased language.
- **More role types**:   
    The ML guidelines are role-specific; similar files could be added for backend, frontend, data engineering, etc.
- **Richer analytics**:   
    The database can be used to analyze average scores per role, track how many “strong fit” candidates appear, etc.
- **Conversation memory**:   
    For a chat-based UI, conversational memory could be introduced so reviewers can ask follow-up questions about each assessment.
---

## 8. Mapping Back to the Assignment Requirements
- **Design, reason, and orchestrate AI agents**
  - 5 explicit agents in `app/agents.py`, orchestrated by a LangGraph `StateGraph`.
- **Use of tools like RAG, memory, tool calling, multi-agent collaboration**
  - RAG over markdown guidelines via FAISS.
  - DB as persistent memory.
  - Tools in `app/tools.py` plus JSON-mode LLM calls.
  - Multi-agent pipeline coordinated in `app/graph.py`.
- **Integration with APIs, DB, file systems, test suites**
  - FastAPI + Gradio frontends.
  - SQLite + SQLAlchemy.
  - Filesystem for resumes and guidelines.
  - Pytest suite in `tests/`.
- **Guardrails and safety controls**
  - PII masking, disclaimers, guidelines emphasizing neutrality and non-fabrication.
- **Use of recent and advanced agent tech**
  - LangGraph multi-agent orchestration.
  - OpenAI `gpt-4o / gpt-4o-mini` with JSON-mode and RAG using `text-embedding-3-small`.
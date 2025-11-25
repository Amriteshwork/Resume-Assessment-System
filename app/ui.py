import os
import gradio as gr

from .tools import parse_resume_text
from .graph import build_graph
from .db import init_db

# Build graph and init DB once
graph_app = build_graph()
init_db()

def assess_with_ui(resume_file, jd_text: str):
    if resume_file is None:
        return "Please upload a resume file.", {}
    if not jd_text or not jd_text.strip():
        return "Please paste a Job Description.", {}

    try:
        filename = os.path.basename(resume_file)
        with open(resume_file, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}", {}

    if not file_bytes:
        return "Empty resume file.", {}

    resume_text = parse_resume_text(file_bytes, filename)

    state = {
        "resume_text": resume_text,
        "jd_text": jd_text,
    }

    # Run LangGraph pipeline
    final_state = graph_app.invoke(state)
    scores = final_state.get("scores", {})
    assessment = final_state.get("cleaned_assessment_text", "")

    disclaimer = (
        "\n\n---\n"
        "_Disclaimer: This assessment is AI-generated based only on the provided "
        "resume and job description. Use human judgment for final hiring decisions._"
    )

    return assessment + disclaimer, scores

def create_demo():
    with gr.Blocks(title="Resume Assessment Agent") as demo:
        gr.Markdown(
            """
            # RESUME ASSESSMENT AGENT

            Upload a resume and paste a Job Description.  
            The system will:
            - Parse the resume
            - Compare it with the JD
            - Compute objective scores
            - Generate a detailed, **guardrailed** assessment
            """
        )

        with gr.Row():
            resume_input = gr.File(
                label="Upload Resume",
                file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"],
                type="filepath"  # Explicitly request a filepath
            )
            jd_input = gr.Textbox(
                label="Job Description",
                lines=15,
                placeholder="Paste the JD here...",
            )

        assess_btn = gr.Button("Assess Resume")

        with gr.Row():
            assessment_output = gr.Markdown(label="Assessment")
            scores_output = gr.JSON(label="Objective Scores")

        assess_btn.click(
            fn=assess_with_ui,
            inputs=[resume_input, jd_input],
            outputs=[assessment_output, scores_output],
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
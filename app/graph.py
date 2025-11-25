from langgraph.graph import StateGraph, END

from .models import AgentState
from .agents import (
    resume_parser_agent,
    jd_parser_agent,
    scoring_agent,
    reviewer_agent,
    safety_agent,
)


def node_parse(state: AgentState) -> AgentState:
    """Orchestrates collaboration between ResumeParserAgent and JDParserAgent."""
    resume_structured = resume_parser_agent.run(state["resume_text"])
    jd_structured = jd_parser_agent.run(state["jd_text"])
    state["resume_structured"] = resume_structured
    state["jd_structured"] = jd_structured
    return state


def node_score(state: AgentState) -> AgentState:
    """Delegates scoring to ScoringAgent."""
    scores = scoring_agent.run(state["resume_structured"], state["jd_structured"])
    state["scores"] = scores
    return state


def node_assess(state: AgentState) -> AgentState:
    """ReviewerAgent produces the narrative assessment."""
    assessment = reviewer_agent.run(
        state["resume_structured"],
        state["jd_structured"],
        state["scores"],
    )
    state["assessment_text"] = assessment
    return state


def node_guardrail_and_save(state: AgentState) -> AgentState:
    """SafetyAgent applies guardrails and persists the record."""
    cleaned = safety_agent.run(
        state["resume_structured"],
        state["jd_structured"],
        state["scores"],
        state.get("assessment_text", ""),
    )
    state["cleaned_assessment_text"] = cleaned
    return state


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse", node_parse)
    workflow.add_node("score", node_score)
    workflow.add_node("assess", node_assess)
    workflow.add_node("guardrail_and_save", node_guardrail_and_save)

    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "score")
    workflow.add_edge("score", "assess")
    workflow.add_edge("assess", "guardrail_and_save")
    workflow.add_edge("guardrail_and_save", END)

    return workflow.compile()

import json
import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


# -----------------------------------------------------------------------------
# Groq configuration through OpenAI-compatible API
# -----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

WRITER_MODEL = os.getenv("GROQ_WRITER_MODEL", DEFAULT_MODEL)
REVIEWER_MODEL = os.getenv("GROQ_REVIEWER_MODEL", DEFAULT_MODEL)
REVISER_MODEL = os.getenv("GROQ_REVISER_MODEL", DEFAULT_MODEL)

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))


def _build_model(model_name: str) -> ChatOpenAI:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file "
            "or Render environment variables."
        )

    return ChatOpenAI(
        model=model_name,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=2,
        timeout=90,
    )


def _writer_model() -> ChatOpenAI:
    return _build_model(WRITER_MODEL)


def _reviewer_model() -> ChatOpenAI:
    return _build_model(REVIEWER_MODEL)


def _reviser_model() -> ChatOpenAI:
    return _build_model(REVISER_MODEL)


class Review(BaseModel):
    decision: Literal["PASS", "REVISE"] = Field(
        description="PASS only if the answer satisfies every review rule; otherwise REVISE."
    )
    feedback: str = Field(
        description="Short, specific feedback. Empty string when decision is PASS."
    )




class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    decision: str
    revision_count: int




def _content_to_text(content) -> str:
    """Normalize LangChain response content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)





def _parse_review(raw_text: str) -> Review:
    """Parse strict reviewer JSON without depending on provider-specific tools."""
    text = raw_text.strip()
    text = re.sub(
        r"^```(?:json)?\s*", 
        "", 
        text, 
        flags=re.IGNORECASE,
        )
    text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Small recovery path if the model surrounds the JSON with prose.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Reviewer returned invalid JSON: {raw_text}")
        payload = json.loads(match.group(0))

    try:
        review = Review.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Reviewer returned an invalid review object: {payload}") from exc

    # Keep PASS output tidy even if the model adds unnecessary feedback.
    if review.decision == "PASS":
        review.feedback = ""
    return review




# -----------------------------------------------------------------------------
# Agent 1: Writer
# -----------------------------------------------------------------------------
def writer(state: State):
    response = _writer_model().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the WRITER agent in a self-correcting multi-agent system. "
                    "You are a beginner-friendly teacher. Explain the topic in 120-160 words. "
                    "Use simple language, one everyday analogy, and one tiny example."
                ),
            },
            {"role": "user", "content": f"Explain: {state['topic']}"},
        ]
    )
    return {
        "draft": _content_to_text(response.content),
        "feedback": "",
        "decision": "",
        "revision_count": 0,
    }




# -----------------------------------------------------------------------------
# Agent 2: Reviewer / verifier in the loop
# -----------------------------------------------------------------------------
def reviewer(state: State):
    response = _reviewer_model().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the REVIEWER agent in a self-correcting multi-agent system. "
                    "Check the answer using ONLY these rules:\n"
                    "1. It is easy for a beginner.\n"
                    "2. It contains an everyday analogy.\n"
                    "3. It contains a tiny concrete example.\n"
                    "4. It stays focused on the requested topic.\n"
                    "If any rule fails, choose REVISE and give one or two precise improvements.\n\n"
                    "Return ONLY valid JSON in exactly this shape:\n"
                    '{"decision":"PASS|REVISE","feedback":"..."}\n'
                    "When the decision is PASS, feedback must be an empty string."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\nAnswer:\n{state['draft']}",
            },
        ]
    )
    review = _parse_review(_content_to_text(response.content))
    return {"decision": review.decision, "feedback": review.feedback}






# -----------------------------------------------------------------------------
# Agent 3: Reviser
# -----------------------------------------------------------------------------
def reviser(state: State):
    response = _reviser_model().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the REVISER agent in a self-correcting multi-agent system. "
                    "Improve the answer using the reviewer feedback. Keep it beginner-friendly "
                    "and concise. Return only the improved answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {state['topic']}\n\n"
                    f"Current answer:\n{state['draft']}\n\n"
                    f"Reviewer feedback:\n{state['feedback']}"
                ),
            },
        ]
    )
    return {
        "draft": _content_to_text(response.content),
        "revision_count": state["revision_count"] + 1,
    }




def route_after_review(state: State):
    """The original self-correction loop controller."""
    if state["decision"] == "PASS":
        return "done"
    if state["revision_count"] >= MAX_REVISIONS:
        return "done"
    return "revise"




builder = StateGraph(State)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("reviser", reviser)
builder.add_edge(START, "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {"revise": "reviser", "done": END},
)
builder.add_edge("reviser", "reviewer")
graph = builder.compile()



def get_runtime_info() -> dict:
        return {
        "provider": "groq",
        "model": DEFAULT_MODEL,
        "writer_model": WRITER_MODEL,
        "reviewer_model": REVIEWER_MODEL,
        "reviser_model": REVISER_MODEL
    }




def run_workflow(topic: str):
    """Run the graph and return data that both the CLI and FastAPI UI can use."""
    initial_state: State = {
        "topic": topic,
        "draft": "",
        "feedback": "",
        "decision": "",
        "revision_count": 0,
    }

    final_state = initial_state.copy()
    events = []

    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, values in update.items():
            final_state.update(values)
            model_for_agent = {
                "writer": WRITER_MODEL,
                "reviewer": REVIEWER_MODEL,
                "reviser": REVISER_MODEL,
            }.get(node_name, DEFAULT_MODEL)

            events.append(
                {
                    "agent": node_name,
                    "draft": values.get("draft", ""),
                    "decision": values.get("decision", ""),
                    "feedback": values.get("feedback", ""),
                    "revision_count": final_state["revision_count"],
                    "provider": "Groq",
                    "model": model_for_agent,
                }
            )

    info = get_runtime_info()
    return {
        "topic": topic,
        "events": events,
        "final_answer": final_state["draft"],
        "final_decision": final_state["decision"],
        "revision_count": final_state["revision_count"],
        **info,
    }





def run_demo(topic: str):
    """Small CLI version, useful if you want to demo without the browser."""
    result = run_workflow(topic)

    print("\n=== SELF-CORRECTING MULTI-AGENT DEMO ===")
    print(f"Provider: {result['provider']}")
    print(f"Writer model: {result['writer_model']}")
    print(f"Reviewer model: {result['reviewer_model']}")
    print(f"Reviser model: {result['reviser_model']}")
    print(f"Topic: {topic}\n")

    for event in result["events"]:
        print(f"\n--- {event['agent'].upper()} ({event['model']}) ---")

        if event["agent"] in {"writer", "reviser"}:
            print(event["draft"])

        elif event["agent"] == "reviewer":
            print("Decision:", event["decision"])
            print("Feedback:", event["feedback"] or "No changes needed")

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    print(f"\nFinal decision: {result['final_decision']}")
    print(f"Revisions used: {result['revision_count']}")


if __name__ == "__main__":
    topic = input("Enter a topic (example: What is an AI agent?): ").strip()
    if not topic:
        topic = "What is an AI agent?"
    run_demo(topic)
"""
agent.py — ReAct-style agent built with LangGraph.

Loop: think → act → [think again if needed] → synthesize → END

The agent picks the right tool based on the question:
  - search_company_policy: single-topic lookup via RAG
  - compare_policies:      year-over-year comparison via LangGraph flow
  - finish:                return the answer directly

A safety check overrides misrouted comparison queries (e.g. LLM picks
search_company_policy when the question clearly mentions multiple years).
"""

import re
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from query import ask_rag
from langgraph_flow import build_graph as build_compare_graph
from config import llm_generate

MAX_ITERATIONS = 3  # Stop looping after this many think/act cycles


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:     str
    scratchpad:   List[dict]  # Running log of Thought and Observation entries
    answer:       str
    tools_used:   List[str]
    iteration:    int
    _next_action: str         # Action decided in think(), consumed by act()
    _next_input:  str         # Input for the next action


VALID_ACTIONS = {"search_company_policy", "compare_policies", "finish"}

# ── Prompt ────────────────────────────────────────────────────────────────────

REACT_PROMPT = """You are a company policy assistant. You can call these tools:

  search_company_policy -- find a specific policy, rule, or benefit (single lookup)
  compare_policies      -- compare policies across years (use when the question mentions
                           comparing, differences, changes, or two or more years)
  finish                -- return your final answer to the user

ROUTING RULES:
- Words like "compare", "difference", "changed", "vs", or TWO OR MORE years → use compare_policies
- Single policy or benefit question → use search_company_policy
- Once you have an Observation with an answer → call finish immediately

Your reasoning so far:
{scratchpad}

Question: {question}

Respond with a JSON object containing exactly:
- "thought":       your reasoning about what to do next
- "action":        one of "search_company_policy", "compare_policies", "finish"
- "action_input":  the query to pass to the tool, or your final answer if action is "finish"

Examples:
{{"thought": "I need to look up the remote work policy.", "action": "search_company_policy", "action_input": "remote work policy"}}
{{"thought": "User wants to compare parental leave between 2024 and 2026.", "action": "compare_policies", "action_input": "compare parental leave policy between 2024 and 2026"}}
{{"thought": "I have the answer.", "action": "finish", "action_input": "The remote work policy allows..."}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_scratchpad(scratchpad: List[dict]) -> str:
    """Render the scratchpad as readable text for the prompt."""
    if not scratchpad:
        return "None yet."
    parts = []
    for entry in scratchpad:
        if entry["role"] == "Observation":
            parts.append(f"\n>>> OBSERVATION <<<\n{entry['content']}\n>>> END <<<\n")
        else:
            parts.append(f"{entry['role']}: {entry['content']}")
    return "\n".join(parts)


def parse_json_response(text: str) -> tuple:
    """
    Extract (thought, action, action_input) from the LLM's JSON output.
    Falls back to regex if the JSON is malformed.
    """
    # Try JSON parse first
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data         = json.loads(match.group())
            thought      = str(data.get("thought", "")).strip()
            action       = str(data.get("action", "finish")).strip().lower().replace(" ", "_")
            action_input = str(data.get("action_input", "")).strip()
            # Normalise any partial action name
            if action not in VALID_ACTIONS:
                action = next((v for v in VALID_ACTIONS if v in action), "finish")
            return thought, action, action_input
        except json.JSONDecodeError:
            pass

    # Regex fallback
    thought      = (re.search(r'"thought"\s*:\s*"(.*?)"',      text, re.DOTALL) or type("", (), {"group": lambda s, i: ""})()).group(1).strip()
    action_raw   = (re.search(r'"action"\s*:\s*"(.*?)"',       text, re.DOTALL) or type("", (), {"group": lambda s, i: "finish"})()).group(1).strip()
    action_input = (re.search(r'"action_input"\s*:\s*"(.*?)"', text, re.DOTALL) or type("", (), {"group": lambda s, i: ""})()).group(1).strip()
    action       = action_raw.lower().replace(" ", "_")
    if action not in VALID_ACTIONS:
        action = "finish"
    return thought, action, action_input


def detect_comparison_query(question: str) -> bool:
    """
    Heuristic check: does the question look like a year-over-year comparison?
    Used as a safety net to catch LLM misrouting.
    """
    q = question.lower()
    compare_words = ["compare", "comparison", "difference", "changed", "changes", "vs", "versus", "between"]
    has_compare_word  = any(w in q for w in compare_words)
    has_multiple_years = len(set(re.findall(r"20\d{2}", q))) >= 2
    return has_compare_word and has_multiple_years


# ── Graph nodes ───────────────────────────────────────────────────────────────

def think(state: AgentState) -> dict:
    """Ask the LLM what to do next and parse its action decision."""
    prompt = REACT_PROMPT.format(
        scratchpad=format_scratchpad(state["scratchpad"]),
        question=state["question"],
    )
    raw = llm_generate(prompt, json_mode=True)
    thought, action, action_input = parse_json_response(raw)

    # Safety net: override misrouted comparison queries
    if action == "search_company_policy" and detect_comparison_query(state["question"]):
        action       = "compare_policies"
        action_input = state["question"]
        thought     += " [Routing corrected: comparison query detected]"

    entry = {"role": "Thought", "content": f"{thought}\n-> Action: {action} | Input: {action_input}"}
    return {
        "scratchpad":   state["scratchpad"] + [entry],
        "_next_action": action,
        "_next_input":  action_input,
        "iteration":    state.get("iteration", 0) + 1,
    }


# Pre-built comparison graph (shared across all agent invocations)
_compare_graph = build_compare_graph()


def act(state: AgentState) -> dict:
    """Execute the action chosen by think() and record the observation."""
    action       = state.get("_next_action", "finish")
    action_input = state.get("_next_input", "") or state["question"]

    if action == "search_company_policy":
        result    = ask_rag(action_input)
        tool_name = "search_company_policy"

    elif action == "compare_policies":
        graph_result = _compare_graph.invoke({
            "query": action_input, "years": [], "results": [], "final_answer": "",
        })
        result    = graph_result["final_answer"]
        tool_name = "compare_policies"

    elif action == "finish":
        return {"answer": action_input}  # action_input is the final answer

    else:
        return {"answer": f"Unrecognised action '{action}'."}

    observation = {"role": "Observation", "content": result}
    return {
        "scratchpad": state["scratchpad"] + [observation],
        "tools_used": state.get("tools_used", []) + [tool_name],
    }


def synthesize(state: AgentState) -> dict:
    """
    If the agent hit the iteration limit, synthesize a final answer from all
    observations collected so far instead of looping again.
    """
    if state.get("answer"):
        return {"answer": state["answer"]}

    observations = [e["content"] for e in state.get("scratchpad", []) if e["role"] == "Observation"]
    if not observations:
        return {"answer": "I wasn't able to find relevant information to answer your question."}

    context = "\n\n---\n\n".join(observations)
    prompt = (
        f"You are a company policy assistant. Answer using ONLY the information below.\n\n"
        f"Instructions:\n"
        f"1. Present ALL relevant details clearly.\n"
        f"2. If the context covers multiple years, include all of them.\n"
        f"3. Organise by year or topic if helpful.\n"
        f"4. Only say information is unavailable if the context truly lacks it.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\nAnswer:"
    )
    return {"answer": llm_generate(prompt)}


def should_continue(state: AgentState) -> str:
    """
    Routing function called after act().
    Returns the name of the next node to visit.
    """
    if state.get("answer"):
        return "finish"
    if state.get("_next_action") == "finish":
        return "finish"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "synthesize"
    # If we already have at least one observation, synthesize instead of looping
    if any(e["role"] == "Observation" for e in state.get("scratchpad", [])):
        return "synthesize"
    return "think"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_agent_graph():
    """Compile and return the ReAct agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("think",      think)
    graph.add_node("act",        act)
    graph.add_node("synthesize", synthesize)
    graph.set_entry_point("think")
    graph.add_edge("think",      "act")
    graph.add_edge("synthesize", END)
    graph.add_conditional_edges(
        "act", should_continue,
        {"think": "think", "finish": END, "synthesize": "synthesize"},
    )
    return graph.compile()


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_agent_graph()
    for q in [
        "What is our remote work policy?",
        "Compare PTO policy between 2024 and 2026",
        "Compare parental leave policy between 2024 and 2026",
    ]:
        print(f"\n{'='*60}\nQ: {q}\n{'-'*60}")
        result = agent.invoke({
            "question": q, "scratchpad": [], "answer": "",
            "tools_used": [], "iteration": 0, "_next_action": "", "_next_input": "",
        })
        print("Answer:", result["answer"])

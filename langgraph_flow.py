"""
langgraph_flow.py — LangGraph workflow for comparing policies across multiple years.

Flow:
  extract_years → fetch_data → compare_and_summarize → END

Used by agent.py and mcp_server.py when a question involves year-to-year comparison.
"""

import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from query import fetch_context
from config import llm_generate


class GraphState(TypedDict):
    query:        str         # The original user question
    years:        List[int]   # Years extracted from the query (e.g. [2024, 2026])
    results:      List[str]   # Context text retrieved per year
    final_answer: str         # The LLM's comparison summary


# Graph nodes

def extract_years(state: GraphState) -> dict:
    """Parse all four-digit years (20xx) from the query string."""
    years = sorted(set(int(y) for y in re.findall(r"20\d{2}", state["query"])))
    return {"years": years}


def fetch_data(state: GraphState) -> dict:
    """
    Retrieve context from Qdrant for each year in the query.
    If no years were found, falls back to a single unfiltered search.
    """
    if not state["years"]:
        ctx = fetch_context(state["query"])
        return {"results": [ctx] if ctx else ["No relevant data found."]}

    results = []
    for year in state["years"]:
        ctx = fetch_context(state["query"], year=year)
        label = f"=== {year} ==="
        results.append(f"{label}\n{ctx}" if ctx else f"{label}\nNo data found for {year}.")

    return {"results": results}


def compare_and_summarize(state: GraphState) -> dict:
    """Ask the LLM to compare the per-year context and list every difference."""
    context   = "\n\n".join(state["results"])
    years_str = " and ".join(str(y) for y in state["years"]) if state["years"] else "the available years"

    prompt = (
        f"You are a policy analyst. You have been given company policy data "
        f"from different years, separated by year headers.\n\n"
        f"{context}\n\n"
        f"Original question: {state['query']}\n\n"
        f"Instructions:\n"
        f"1. For each year ({years_str}), state exactly what the policy says.\n"
        f"2. Then list every specific difference (numbers, durations, eligibility, coverage, etc.).\n"
        f"3. If a policy exists in one year but not the other, say so explicitly.\n"
        f"4. Do NOT infer or assume anything not stated in the data above.\n"
        f"5. If the data lacks information, say what is missing.\n\n"
        f"Comparison:"
    )
    return {"final_answer": llm_generate(prompt)}


# Graph assembly 

def build_graph():
    """Compile and return the comparison workflow graph."""
    workflow = StateGraph(GraphState)
    workflow.add_node("extract", extract_years)
    workflow.add_node("fetch",   fetch_data)
    workflow.add_node("compare", compare_and_summarize)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "fetch")
    workflow.add_edge("fetch",   "compare")
    workflow.add_edge("compare", END)
    return workflow.compile()

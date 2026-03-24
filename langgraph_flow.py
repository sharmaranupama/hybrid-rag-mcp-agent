import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from query import fetch_context
from config import llm_generate


class GraphState(TypedDict):
    query:        str
    years:        List[int]
    results:      List[str]
    final_answer: str


def extract_years(state: GraphState) -> dict:
    years = sorted(set(int(y) for y in re.findall(r"20\d{2}", state["query"])))
    return {"years": years}


def fetch_data(state: GraphState) -> dict:
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


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("extract", extract_years)
    workflow.add_node("fetch",   fetch_data)
    workflow.add_node("compare", compare_and_summarize)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "fetch")
    workflow.add_edge("fetch",   "compare")
    workflow.add_edge("compare", END)
    return workflow.compile()

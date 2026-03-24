import logging
import sys

from dotenv import load_dotenv
load_dotenv()

from opentelemetry import trace
from phoenix.otel import register
from openinference.instrumentation.mcp import MCPInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME

tracer_provider = register(
    auto_instrument=True,
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
)
MCPInstrumentor().instrument(tracer_provider=tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer(PHOENIX_PROJECT_NAME)

from query import ask_rag
from langgraph_flow import build_graph
from agent import build_agent_graph
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp-server")

mcp       = FastMCP("Company-Knowledge")
graph_app = build_graph()
agent_app = build_agent_graph()


@mcp.tool()
def search_company_policy(query: str) -> str:
    """Search for a specific company policy using RAG."""
    with tracer.start_as_current_span("search_company_policy") as span:
        span.set_attribute("input.value", query)
        span.set_attribute("openinference.span.kind", "TOOL")
        try:
            result = ask_rag(query)
            span.set_attribute("output.value", result)
            return result
        except Exception as e:
            span.record_exception(e)
            logger.error(f"RAG error: {e}")
            return f"Error: {e}"


@mcp.tool()
def compare_policies(query: str) -> str:
    """Compare policies across different years using the LangGraph workflow."""
    with tracer.start_as_current_span("compare_policies") as span:
        span.set_attribute("input.value", query)
        span.set_attribute("openinference.span.kind", "TOOL")
        try:
            result = graph_app.invoke({"query": query, "years": [], "results": [], "final_answer": ""})
            answer = result["final_answer"]
            span.set_attribute("output.value", answer)
            return answer
        except Exception as e:
            span.record_exception(e)
            logger.error(f"Graph error: {e}")
            return f"Error: {e}"


@mcp.tool()
def ask_agent(query: str) -> str:
    """ReAct agent: reasons step by step and calls tools until it has a complete answer."""
    with tracer.start_as_current_span("ask_agent") as span:
        span.set_attribute("input.value", query)
        span.set_attribute("openinference.span.kind", "TOOL")
        try:
            result = agent_app.invoke({
                "question": query, "scratchpad": [], "answer": "",
                "tools_used": [], "iteration": 0, "_next_action": "", "_next_input": "",
            })
            answer     = result["answer"]
            tools_used = result.get("tools_used", [])
            span.set_attribute("output.value",     answer)
            span.set_attribute("agent.tools_used", ", ".join(tools_used))
            span.set_attribute("agent.iterations", result.get("iteration", 0))
            span.set_attribute("agent.scratchpad", str(result.get("scratchpad", [])))
            return answer
        except Exception as e:
            span.record_exception(e)
            logger.error(f"Agent error: {e}")
            return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()

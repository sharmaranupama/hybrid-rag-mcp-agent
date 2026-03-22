# hybrid-rag-mcp-agent
 RAG system with hybrid search, a ReAct agent, MCP tool exposure, and LLM evaluation — built as a weekend learning project.
Stack: Qdrant · Arize Phoenix · LangGraph · FastMCP · Ollama · Gemini · Streamlit · Docker

# What this is
A company knowledge assistant that answers questions about internal policies and compares them across years. The interesting part is the architecture beneath it:

Hybrid search — dense (nomic-embed-text) + sparse (BM25) vectors fused with RRF, served by Qdrant
Three retrieval layers — direct RAG, a LangGraph multi-year comparison workflow, and a ReAct agent that routes between them
MCP server — all three tools exposed via FastMCP, callable from Claude Desktop, Cursor, or any MCP client
Full observability — every span traced in Arize Phoenix via OpenTelemetry auto-instrumentation
LLM-as-judge evaluation — Hallucination, QA Correctness, and Context Relevance scored with Phoenix evaluators (Gemini or Ollama as judge)


Project structure
├── app.py                # Streamlit UI (ingest, chat, experiments tabs)
├── ingest.py             # CSV → Qdrant (dense + BM25 hybrid vectors)
├── query.py              # Hybrid search + multi-year balanced retrieval
├── agent.py              # ReAct agent (LangGraph)
├── langgraph_flow.py     # Multi-year comparison workflow
├── mcp_server.py         # FastMCP server exposing 3 tools
├── rag_eval.py           # Pull Phoenix spans → evaluate with LLM-as-judge
├── phoenix_datasets.py   # Create eval datasets + run experiments in Phoenix
├── experiment.py         # Grid runner (embedding × prompt × model combos)
├── config.py             # All settings, model config, LLM provider abstraction
├── data.csv              # Sample company policy data (synthetic, 20 rows)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

# Quickstart
## Prerequisites

Docker + Docker Compose
Ollama running locally (for local embeddings + LLM)
A Gemini API key (optional — needed for Gemini provider and fast eval)

1. Pull required Ollama models
bashollama pull nomic-embed-text   # embedding model (required)
ollama pull llama3.2           # reasoning model (required for Ollama provider)
2. Configure environment
bashcp .env.example .env
# Edit .env — add GEMINI_API_KEY if you want Gemini provider or fast eval
3. Start the stack
bashdocker-compose up --build
This starts four services:
ServicePortPurposeQdrant6333Vector databasePhoenix6006Observability + eval UIMCP server6274/6277FastMCP tool serverStreamlit UI8501Main interface
4. Ingest data and start chatting
Open http://localhost:8501

Data Ingestion tab → upload data.csv → click Start Hybrid Ingestion
Agent Chat tab → ask questions or use the sample queries


# Architecture
Retrieval
The system uses Qdrant's hybrid search — each document is indexed with both a dense semantic embedding and a sparse BM25 vector. At query time, both are searched in parallel and merged server-side with Reciprocal Rank Fusion (RRF).
For multi-year queries, a balanced fetch runs an additional per-year search to prevent one year's data from dominating the RRF rankings.
Query
  │
  ├─► dense prefetch (nomic-embed-text)  ─┐
  │                                        ├─► RRF fusion ─► top-5 chunks
  └─► sparse prefetch (BM25)             ─┘
Agent layers
User query
    │
    ▼
ReAct Agent (agent.py)
    ├─► search_company_policy  →  ask_rag()         [single hybrid search]
    └─► compare_policies       →  LangGraph flow    [per-year fetch + compare]
                                    extract_years → fetch_data → compare_and_summarize
MCP server
Three tools exposed via FastMCP:
python@mcp.tool()
def search_company_policy(query: str) -> str: ...

@mcp.tool()
def compare_policies(query: str) -> str: ...

@mcp.tool()
def ask_agent(query: str) -> str: ...
Every tool call is traced in Phoenix automatically.

Switching between Ollama and Gemini
Set LLM_PROVIDER in .env:
bashLLM_PROVIDER=ollama   # default — uses llama3.2 locally
LLM_PROVIDER=gemini   # uses Gemini API (requires GEMINI_API_KEY)
Or switch live in the Streamlit sidebar without restarting.
Available Gemini models (set via GEMINI_MODEL):
Model stringDescriptiongemini-2.5-flash-liteDefault — fast, cost-efficientgemini-2.5-flashBalancedgemini-3.1-flash-lite-previewPreview, fastestgemini-3-flash-previewPreview, most capable

# Evaluation
Run LLM-as-judge evaluation (rag_eval.py)
Pulls traces from Phoenix, re-fetches Qdrant context, scores each answer on three dimensions:
bash# With Gemini as judge (fast — ~1.3s/query)
GEMINI_API_KEY=your_key python rag_eval.py

# Ollama only (slow — ~25s/query, no API key needed)
python rag_eval.py
Scores produced per answer:
MetricWhat it measuresGood valueHallucinationClaims not supported by retrieved contextLowQA CorrectnessAnswer correctly addresses the questionHighContext RelevanceRetrieved chunks are relevant to the queryHigh
Results are uploaded to Phoenix as span annotations — visible in the Traces view.
Run structured experiments (phoenix_datasets.py)
bash# Create an eval dataset in Phoenix
python phoenix_datasets.py create-dataset

# Run an experiment against it and upload results
python phoenix_datasets.py run-experiment

# List all datasets
python phoenix_datasets.py show-datasets
Results from the Datasets & Experiments tab in the UI do the same thing interactively.

Real results (Gemini 3.1 Flash-Lite vs Ollama llama3.2)
10 queries across in-scope lookups, cross-year comparisons, and out-of-scope hallucination tests:
CategoryGemini scoreGemini latencyOllama scoreOllama latencyIn-scope (n=6)1.0001.3s1.00025.4sComparison (n=2)1.0001.5s1.00026.1sOut-of-scope (n=2)0.9171.2s0.91721.7sTopic hit rate100%—100%—HallucinationNone—None—
Retrieval quality was identical across providers — the vector DB does the heavy lifting. The only meaningful difference was latency.

Observability
Phoenix traces every span automatically. Open http://localhost:6006 to see:

Full span trees for every agent query (think → act → synthesize)
Per-node latencies and token counts
Tool call inputs and outputs
LLM-as-judge eval scores as annotations on each span


Connecting to Claude Desktop or Cursor via MCP
Add to your MCP config:
json{
  "mcpServers": {
    "company-knowledge": {
      "url": "http://localhost:6274/sse"
    }
  }
}
Three tools will appear: search_company_policy, compare_policies, ask_agent.





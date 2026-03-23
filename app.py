"""
app.py — Streamlit UI for the Company Knowledge Agent.

Tabs:
  1. Data Ingestion          — Upload a CSV and push it to Qdrant
  2. Agent Chat              — Ask questions via the ReAct agent
  3. Qdrant & Phoenix        — Collection stats + observability dashboard
  4. Datasets & Experiments  — Run and view Phoenix eval experiments
  5. RAG Eval                — LLM-as-judge scoring + logs results back to Phoenix
"""

import os
import glob
import io
import sys
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Observability (must be set up before importing tool modules) ───────────────
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import config
from config import (
    EMBED_MODEL, COLLECTION_NAME, INGEST_BATCH_SIZE,
    QDRANT_URL, OLLAMA_HOST,
    PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME,
    REASONING_MODEL, GEMINI_MODELS, DEFAULT_GEMINI_MODEL, PHOENIX_BASE_URL,
)

tracer_provider = register(
    auto_instrument=True,
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Company Knowledge Agent", layout="wide")
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def check_qdrant():
    """Return (ok: bool, num_collections: int)."""
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL, timeout=3)
        n  = len(qc.get_collections().collections)
        qc.close()
        return True, n
    except Exception:
        return False, 0


def check_ollama():
    """Return (ok: bool, model_names: list)."""
    try:
        from ollama import Client as OllamaClient
        oc       = OllamaClient(host=OLLAMA_HOST)
        response = oc.list()
        if isinstance(response, dict):
            models = [m.get("name", "") for m in response.get("models", [])]
        else:
            models = [getattr(m, "model", getattr(m, "name", "")) for m in (response.models or [])]
        return True, models
    except Exception:
        return False, []


def check_phoenix():
    """Return True if Phoenix is reachable."""
    try:
        import urllib.request
        url = PHOENIX_COLLECTOR_ENDPOINT.replace("/v1/traces", "")
        return urllib.request.urlopen(url, timeout=3).status == 200
    except Exception:
        return False


def validate_gemini_key(api_key: str, model_name: str) -> tuple[bool, str]:
    """Validate a Gemini API key by making a tiny test call."""
    if not api_key or not api_key.strip():
        return False, "No API key provided."
    try:
        from google import genai
        client = genai.Client(api_key=api_key.strip())
        client.models.generate_content(model=model_name, contents="Say OK")
        return True, "Key is valid."
    except Exception as e:
        return False, str(e)


def _capture_stdout(fn):
    """
    Run fn(), stream its stdout line-by-line into a Streamlit code block,
    and return (output_str, error_or_None).
    """
    lines      = []
    output_box = st.empty()

    class _Capture(io.StringIO):
        def write(self, s):
            super().write(s)
            lines.append(s)
            output_box.code("".join(lines), language="bash")

    old = sys.stdout
    sys.stdout = _Capture()
    try:
        fn()
        return "".join(lines), None
    except Exception as e:
        return "".join(lines), e
    finally:
        sys.stdout = old


def log_evaluations_to_phoenix(eval_df, hallucination_df, qa_df, relevance_df):
    """
    Send LLM-as-judge eval scores back to Phoenix as span annotations.

    Uses the official Phoenix client API (per docs.arize.com/phoenix):
      phoenix_client.spans.log_span_annotations(span_annotations=[...])

    Each SpanAnnotationData is linked to the original trace span via span_id,
    so annotations appear inline when you open a trace in the Phoenix UI.

    Scores:
      hallucination:     1.0 = hallucinated (bad),  0.0 = grounded (good)
      qa_correctness:    1.0 = correct (good),       0.0 = incorrect (bad)
      context_relevance: 1.0 = relevant (good),      0.0 = irrelevant (bad)

    Args:
        eval_df:           DataFrame indexed by context.span_id
        hallucination_df:  Output from HallucinationEvaluator — indexed by span_id
        qa_df:             Output from QAEvaluator             — indexed by span_id
        relevance_df:      Output from RelevanceEvaluator      — indexed by span_id

    Returns:
        (True, success_message) or (False, error_message)
    """
    try:
        from phoenix.client import Client
        from phoenix.client.resources.spans import SpanAnnotationData

        phoenix_client = Client(base_url=PHOENIX_BASE_URL)
        annotations    = []

        for span_id in eval_df.index:

            # ── Hallucination ──────────────────────────────────────────────
            if span_id in hallucination_df.index:
                row = hallucination_df.loc[span_id]
                annotations.append(SpanAnnotationData(
                    name="hallucination",
                    span_id=str(span_id),
                    annotator_kind="LLM",
                    result={
                        "label": row["label"],
                        "score": 1.0 if row["label"] == "hallucinated" else 0.0,
                    },
                    metadata={"explanation": str(row.get("explanation", ""))},
                ))

            # ── QA Correctness ─────────────────────────────────────────────
            if span_id in qa_df.index:
                row = qa_df.loc[span_id]
                annotations.append(SpanAnnotationData(
                    name="qa_correctness",
                    span_id=str(span_id),
                    annotator_kind="LLM",
                    result={
                        "label": row["label"],
                        "score": 1.0 if row["label"] == "correct" else 0.0,
                    },
                    metadata={"explanation": str(row.get("explanation", ""))},
                ))

            # ── Context Relevance ──────────────────────────────────────────
            if span_id in relevance_df.index:
                row = relevance_df.loc[span_id]
                annotations.append(SpanAnnotationData(
                    name="context_relevance",
                    span_id=str(span_id),
                    annotator_kind="LLM",
                    result={
                        "label": row["label"],
                        "score": 1.0 if row["label"] == "relevant" else 0.0,
                    },
                    metadata={"explanation": str(row.get("explanation", ""))},
                ))

        # Send all annotations to Phoenix in a single async call
        phoenix_client.spans.log_span_annotations(
            span_annotations=annotations,
            sync=False,
        )

        return True, f"Logged {len(annotations)} annotations to Phoenix ({PHOENIX_BASE_URL})."

    except Exception as e:
        return False, f"Phoenix annotation logging failed: {e}"


# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "llm_provider":     config.get_provider(),
    "gemini_api_key":   config.get_gemini_api_key(),
    "gemini_model":     config.get_gemini_model(),
    "gemini_key_valid": None,
    "gemini_key_error": "",
    "messages":         [],
    "ds_log":           "",
    "exp_log":          "",
    "list_log":         "",
    "eval_scores":      None,
    "eval_detail_df":   None,
    "eval_log":         "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Check services once per render (results reused across all tabs)
qdrant_ok, n_collections = check_qdrant()
ollama_ok, model_list    = check_ollama()
phoenix_ok               = check_phoenix()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Company Knowledge")
    st.markdown("---")
    st.markdown("### Service Health")
    st.write("Qdrant:",  f"✅ {n_collections} collection(s)" if qdrant_ok  else "❌ unreachable")
    st.write("Ollama:",  f"✅ {len(model_list)} model(s)"    if ollama_ok  else "❌ unreachable")
    st.write("Phoenix:", "✅ tracing active"                  if phoenix_ok else "❌ unreachable")
    st.markdown("---")
    st.markdown("### Agent Graph")
    st.code("think --> act --[need more info]--> think\n              \\--[done / limit]---> END")
    st.caption("ReAct loop with tool routing.")
    st.caption("Built with LangGraph + MCP + Qdrant + Ollama/Gemini")


# ── Model selector (top bar, always visible) ───────────────────────────────────
with st.container():
    st.markdown("### 🧠 Reasoning Model")
    col_provider, col_model, col_key, col_btn, col_status = st.columns([1.2, 2, 2.5, 1.2, 2])

    with col_provider:
        provider = st.radio(
            "Provider",
            options=["ollama", "gemini"],
            index=0 if st.session_state.llm_provider == "ollama" else 1,
            format_func=lambda x: "Ollama" if x == "ollama" else "Gemini",
            horizontal=False,
            key="provider_radio",
        )

    with col_model:
        if provider == "gemini":
            model_options = list(GEMINI_MODELS.keys())
            default_idx   = model_options.index(st.session_state.gemini_model) \
                            if st.session_state.gemini_model in model_options else 0
            selected_model = st.selectbox(
                "Gemini Model",
                options=model_options,
                index=default_idx,
                format_func=lambda m: GEMINI_MODELS[m],
                key="gemini_model_select",
            )
            if selected_model != st.session_state.gemini_model:
                st.session_state.gemini_model     = selected_model
                st.session_state.gemini_key_valid = None
        else:
            st.text_input("Model", value=REASONING_MODEL, disabled=True, key="ollama_model_display")
            selected_model = REASONING_MODEL

    with col_key:
        if provider == "gemini":
            api_key = st.text_input(
                "API Key",
                value=st.session_state.gemini_api_key,
                type="password",
                placeholder="Paste Gemini API key...",
                key="gemini_key_input",
            )
            if api_key != st.session_state.gemini_api_key:
                st.session_state.gemini_key_valid = None

    with col_btn:
        if provider == "gemini":
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Activate", type="primary", use_container_width=True):
                with st.spinner("Validating..."):
                    is_valid, msg = validate_gemini_key(api_key, selected_model)
                st.session_state.gemini_key_valid = is_valid
                st.session_state.gemini_key_error = msg
                if is_valid:
                    st.session_state.gemini_api_key = api_key.strip()

    with col_status:
        if provider == "gemini":
            if st.session_state.gemini_key_valid is True:
                st.success(f"Active: `{selected_model}`", icon="✅")
                config.set_provider("gemini", st.session_state.gemini_api_key, selected_model)
            elif st.session_state.gemini_key_valid is False:
                st.error("Invalid key — using Ollama", icon="❌")
                config.set_provider("ollama")
            else:
                st.info("Enter key and click Activate", icon="ℹ️")
                config.set_provider("ollama")
            st.caption("[Get free API key →](https://aistudio.google.com/apikey)")
        else:
            config.set_provider("ollama")
            st.success(f"Active: `{REASONING_MODEL}`", icon="✅")

    st.session_state.llm_provider = config.get_provider()
    st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ingest, tab_agent, tab_stats, tab_datasets, tab_eval = st.tabs(
    ["Data Ingestion", "Agent Chat", "Qdrant & Phoenix", "Datasets & Experiments", "RAG Eval"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data Ingestion
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest Data into Qdrant")
    st.write("Upload a CSV. Each row gets **dense** (semantic) + **BM25** (keyword) vectors.")

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        st.markdown("#### Settings")
        collection_name = st.text_input("Collection Name", value=COLLECTION_NAME)
        embed_model     = st.text_input("Embedding Model", value=EMBED_MODEL)
        batch_size      = st.slider("Batch Size", 5, 100, INGEST_BATCH_SIZE)
        st.markdown("#### Features enabled")
        st.caption("Dense vectors (nomic-embed-text)")
        st.caption("Sparse vectors (BM25 + server-side IDF)")
        st.caption("Int8 scalar quantization")
        st.caption("Payload indexes: year, topic, category")

    with col_upload:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Preview — {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df.head(20), use_container_width=True, height=300)

            if st.button("Start Hybrid Ingestion", type="primary", use_container_width=True):
                if not qdrant_ok or not ollama_ok:
                    st.error("Qdrant and Ollama must both be running.")
                else:
                    # Write to a temp file so ingest_csv can read it from disk
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        df.to_csv(tmp.name, index=False)
                        tmp_path = tmp.name
                    progress = st.progress(0, text="Setting up hybrid collection...")
                    try:
                        from ingest import ingest_csv
                        def _update(frac, msg):
                            progress.progress(min(frac, 0.99), text=msg)
                        ingest_csv(tmp_path, progress_callback=_update)
                        progress.progress(1.0, text="Done!")
                        st.success(f"Ingested {len(df)} rows with hybrid vectors.")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                    finally:
                        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Agent Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.header("Ask the Agent")
    st.write("The ReAct agent reasons step by step, calling tools as needed.")

    active_provider = config.get_provider()
    active_model    = config.get_gemini_model() if active_provider == "gemini" else REASONING_MODEL
    st.info(f"Reasoning with **{active_provider}** → `{active_model}`", icon="🧠")

    # Example question shortcuts
    example_questions = [
        "What is the company's remote work policy?",
        "Compare vacation policy between 2024 and 2026",
        "What health insurance options are available?",
        "Compare parental leave policy between 2024 and 2026",
    ]
    st.write("Try these:")
    for idx, q in enumerate(st.columns(len(example_questions))):
        with q:
            if st.button(example_questions[idx], key=f"ex_{idx}", use_container_width=True):
                st.session_state["prefill_query"] = example_questions[idx]

    st.markdown("---")

    def render_message(msg: dict):
        """Display a single chat message, with reasoning steps expandable."""
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                scratchpad = msg.get("scratchpad", [])
                answer     = msg.get("answer", msg.get("content", ""))
                tools_used = msg.get("tools_used", [])
                iterations = msg.get("iterations", 0)
                model_used = msg.get("model_used", "")

                # Show the think/act steps in a collapsible section
                if scratchpad:
                    with st.expander("Reasoning steps", expanded=False):
                        for entry in scratchpad:
                            color = "#1a73e8" if entry["role"] == "Thought" else "#2e7d32"
                            st.markdown(
                                f"<span style='font-weight:600;color:{color}'>{entry['role']}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(entry["content"])
                            st.divider()

                meta = f"**Tools:** {', '.join(tools_used) or 'none'} &nbsp;|&nbsp; **Steps:** {iterations}"
                if model_used:
                    meta += f" &nbsp;|&nbsp; **Model:** {model_used}"
                st.markdown(meta)
                st.markdown(answer)
            else:
                st.markdown(msg["content"])

    # Render existing conversation history
    for msg in st.session_state.messages:
        render_message(msg)

    # Handle new user input (either typed or from an example button)
    prefill    = st.session_state.pop("prefill_query", None)
    user_input = st.chat_input("Ask about company policies...") or prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not qdrant_ok or not ollama_ok:
                msg = "Services not available. Make sure Qdrant and Ollama are running."
                st.warning(msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": msg,
                    "answer": msg, "scratchpad": [], "tools_used": [], "iterations": 0,
                })
            else:
                try:
                    from agent import build_agent_graph
                    import agent as agent_module

                    # Wrap think/act to show live progress in the UI
                    original_think = agent_module.think
                    original_act   = agent_module.act
                    step_counter   = {"n": 0}
                    progress_box   = st.expander("Reasoning steps", expanded=True)

                    def think_with_progress(state):
                        step_counter["n"] += 1
                        try:
                            with progress_box:
                                st.markdown(
                                    f"<span style='font-weight:600;color:#1a73e8'>Thought (step {step_counter['n']})</span>",
                                    unsafe_allow_html=True,
                                )
                                st.caption("Deciding what to do next...")
                        except Exception:
                            pass  # Streamlit context may be gone mid-stream
                        return original_think(state)

                    def act_with_progress(state):
                        action = state.get("_next_action", "")
                        inp    = (state.get("_next_input", "") or "")[:80]
                        label  = "Finishing up..." if action == "finish" else f"Calling `{action}` — _{inp}_"
                        try:
                            with progress_box:
                                st.markdown("<span style='font-weight:600;color:#2e7d32'>Action</span>",
                                            unsafe_allow_html=True)
                                st.caption(label)
                                st.divider()
                        except Exception:
                            pass
                        return original_act(state)

                    # Temporarily replace think/act with progress-aware versions
                    agent_module.think = think_with_progress
                    agent_module.act   = act_with_progress

                    current_provider = config.get_provider()
                    current_model    = config.get_gemini_model() if current_provider == "gemini" else REASONING_MODEL

                    result = build_agent_graph().invoke({
                        "question": user_input, "scratchpad": [], "answer": "",
                        "tools_used": [], "iteration": 0, "_next_action": "", "_next_input": "",
                    })

                    # Restore original functions
                    agent_module.think = original_think
                    agent_module.act   = original_act

                    answer     = result.get("answer", "No answer returned.")
                    tools_used = result.get("tools_used", [])
                    scratchpad = result.get("scratchpad", [])
                    iterations = result.get("iteration", 0)

                    st.markdown(
                        f"**Tools:** {', '.join(tools_used) or 'none'} &nbsp;|&nbsp; "
                        f"**Steps:** {iterations} &nbsp;|&nbsp; **Model:** {current_model}"
                    )
                    st.markdown(answer)

                    # Show which chunks were retrieved and their scores
                    try:
                        from query import fetch_context_with_scores, _collection_has_named_vectors
                        chunks = fetch_context_with_scores(user_input)
                        if chunks:
                            with st.expander("Retrieval details (hybrid search)", expanded=False):
                                for ci, c in enumerate(chunks):
                                    score_str = f"{c['score']:.4f}" if c['score'] else "n/a"
                                    st.markdown(f"**#{ci+1}** score: `{score_str}` | topic: `{c['topic']}` | year: `{c['year']}`")
                                    st.caption(c['text'][:200])
                                is_hybrid = _collection_has_named_vectors()
                                st.caption(f"Search mode: {'hybrid (dense + BM25 RRF)' if is_hybrid else 'dense only'}")
                    except Exception:
                        pass

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "answer": answer, "scratchpad": scratchpad,
                        "tools_used": tools_used, "iterations": iterations,
                        "model_used": current_model,
                    })

                except Exception as e:
                    err = f"Agent error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err,
                        "answer": err, "scratchpad": [], "tools_used": [], "iterations": 0,
                    })


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Qdrant & Phoenix stats
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.header("Qdrant & Phoenix Dashboard")
    col_qdrant, col_phoenix = st.columns(2)

    with col_qdrant:
        st.subheader("Qdrant Collection")
        if qdrant_ok:
            try:
                from qdrant_client import QdrantClient as _QC
                _qc  = _QC(url=QDRANT_URL, timeout=5)
                info = _qc.get_collection(COLLECTION_NAME)

                st.metric("Points",   info.points_count)
                st.metric("Segments", info.segments_count)
                st.metric("Status",   str(info.status))

                vecs = info.config.params.vectors
                st.markdown("**Vector configs:**")
                if isinstance(vecs, dict):
                    for name, vconf in vecs.items():
                        st.caption(f"  `{name}`: dim={vconf.size}, distance={vconf.distance}")
                elif vecs:
                    st.caption(f"  default: dim={vecs.size}")

                sp = info.config.params.sparse_vectors
                if sp:
                    st.markdown("**Sparse vectors:**")
                    for name, sconf in sp.items():
                        st.caption(f"  `{name}`: modifier={getattr(sconf, 'modifier', 'none')}")

                st.markdown(f"**Quantization:** {'int8 scalar' if info.config.quantization_config else 'disabled'}")

                st.markdown("**Payload indexes:**")
                if hasattr(info, "payload_schema") and info.payload_schema:
                    for field, schema in info.payload_schema.items():
                        st.caption(f"  `{field}`: {schema.data_type}")
                else:
                    st.caption("  none detected")

                st.markdown("---")
                st.markdown("**Test hybrid search**")
                test_query = st.text_input("Query", value="remote work policy", key="qdrant_test_q")
                if st.button("Search", key="qdrant_test_btn"):
                    try:
                        from query import fetch_context_with_scores
                        results = fetch_context_with_scores(test_query)
                        if results:
                            for idx, r in enumerate(results):
                                st.markdown(f"**#{idx+1}** score: `{r['score']:.4f}` | topic: `{r['topic']}` | year: `{r['year']}`")
                                st.caption(r['text'][:200])
                        else:
                            st.warning("No results. Re-ingest with hybrid vectors first.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                _qc.close()

            except Exception as e:
                st.error(f"Could not read collection: {e}")
        else:
            st.warning("Qdrant is not reachable.")

    with col_phoenix:
        st.subheader("Phoenix Observability")
        if phoenix_ok:
            st.markdown(f"[Open Phoenix UI]({PHOENIX_BASE_URL})")
            st.markdown(f"**Project:** `{PHOENIX_PROJECT_NAME}`")
            try:
                from phoenix.client import Client as _PxC
                _px   = _PxC(base_url=PHOENIX_BASE_URL)
                spans = _px.spans.get_spans_dataframe(project_name=PHOENIX_PROJECT_NAME)
                if spans is not None and not spans.empty:
                    st.metric("Total spans", len(spans))
                    if "span_kind" in spans.columns:
                        st.markdown("**Span kinds:**")
                        for kind, count in spans["span_kind"].value_counts().items():
                            st.caption(f"  {kind}: {count}")
                else:
                    st.caption("No spans yet. Run some queries first.")
            except Exception:
                st.caption("Span query requires phoenix-client >= 13.x")
            st.markdown("---")
            st.markdown(f"- [Traces]({PHOENIX_BASE_URL}/projects)")
            st.markdown(f"- [Datasets]({PHOENIX_BASE_URL}/datasets)")
        else:
            st.warning("Phoenix is not reachable.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Datasets & Experiments
# ══════════════════════════════════════════════════════════════════════════════
with tab_datasets:
    st.header("Phoenix Datasets & Experiments")
    st.write("Create eval datasets, run experiments, and view results — all from the UI.")

    if not phoenix_ok:
        st.warning("Phoenix is not reachable. Start it with `docker-compose up -d phoenix`.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Create Eval Dataset")
        st.write("Uploads 10 built-in test queries to Phoenix as a versioned eval dataset.")
        with st.expander("Test queries", expanded=False):
            from phoenix_datasets import TEST_EXAMPLES
            badges = {"in-scope": "🟢", "comparison": "🔵", "out-of-scope": "🔴"}
            for ex in TEST_EXAMPLES:
                st.caption(f"{badges.get(ex['category'], '⚪')} **{ex['category']}** — {ex['input']}")

        if st.button("Create Dataset in Phoenix", type="primary", use_container_width=True, key="btn_create_ds"):
            if not phoenix_ok:
                st.error("Phoenix is not reachable.")
            else:
                from phoenix_datasets import create_dataset
                log, err = _capture_stdout(create_dataset)
                st.session_state.ds_log = log
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Done! Check Phoenix → Datasets.")

        if st.session_state.ds_log:
            with st.expander("Last output", expanded=False):
                st.code(st.session_state.ds_log, language="bash")

        st.markdown("---")
        st.subheader("3. List Datasets")
        if st.button("List Datasets", use_container_width=True, key="btn_list_ds"):
            if not phoenix_ok:
                st.error("Phoenix is not reachable.")
            else:
                from phoenix_datasets import show_datasets
                log, err = _capture_stdout(show_datasets)
                st.session_state.list_log = log
                if err:
                    st.error(f"Error: {err}")

        if st.session_state.list_log:
            with st.expander("Last output", expanded=False):
                st.code(st.session_state.list_log, language="bash")

    with col_right:
        st.subheader("2. Run Experiment")
        st.write("Runs all 10 queries through the RAG pipeline and uploads results to Phoenix.")
        _ap = config.get_provider()
        _am = config.get_gemini_model() if _ap == "gemini" else REASONING_MODEL
        st.info(f"Will run with **{_ap}** → `{_am}`", icon="🧠")

        if st.button("Run Experiment", type="primary", use_container_width=True, key="btn_run_exp"):
            if not phoenix_ok:
                st.error("Phoenix is not reachable.")
            elif not qdrant_ok:
                st.error("Qdrant is not reachable.")
            else:
                from phoenix_datasets import run_experiment
                with st.spinner("Running experiment..."):
                    log, err = _capture_stdout(run_experiment)
                st.session_state.exp_log = log
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Done! Results saved to `experiments/` and uploaded to Phoenix.")
                    st.markdown(f"[View in Phoenix → Datasets]({PHOENIX_BASE_URL}/datasets)")

        if st.session_state.exp_log:
            with st.expander("Last experiment output", expanded=True):
                st.code(st.session_state.exp_log, language="bash")

        st.markdown("---")
        st.subheader("Local Experiment Results")
        csv_files = sorted(glob.glob("experiments/*.csv"), reverse=True)
        if csv_files:
            selected = st.selectbox(
                "Select a result file",
                options=csv_files,
                format_func=os.path.basename,
                key="exp_csv_select",
            )
            if selected:
                try:
                    result_df = pd.read_csv(selected)
                    if "category" in result_df.columns and "top_score" in result_df.columns:
                        st.markdown("**Score by category:**")
                        summary = result_df.groupby("category").agg(
                            avg_score=("top_score", "mean"),
                            avg_latency=("latency_s", "mean"),
                            count=("question", "count"),
                        ).round(4)
                        st.dataframe(summary, use_container_width=True)
                    display_cols = [c for c in ["question", "category", "topic_found", "top_score", "latency_s", "answer"]
                                    if c in result_df.columns]
                    st.dataframe(result_df[display_cols], use_container_width=True, height=300)
                except Exception as e:
                    st.error(f"Could not load CSV: {e}")
        else:
            st.caption("No experiment CSVs found yet. Run an experiment first.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — RAG Eval (LLM-as-judge + Phoenix annotation logging)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("RAG Evaluation")
    st.write(
        "Pull live traces from Phoenix, score each answer with **LLM-as-judge** "
        "(Hallucination · QA Correctness · Context Relevance), "
        "then log the scores back to Phoenix as span annotations."
    )

    if not phoenix_ok:
        st.warning("Phoenix is not reachable. Start it with `docker-compose up -d phoenix`.")
    if not qdrant_ok:
        st.warning("Qdrant is not reachable.")

    # ── Judge model selector ──────────────────────────────────────────────────
    st.markdown("#### Judge model")
    _ep = config.get_provider()
    _ek = config.get_gemini_api_key()
    _em = config.get_gemini_model() if _ep == "gemini" else REASONING_MODEL

    col_judge, col_judge_info = st.columns([1, 2])
    with col_judge:
        judge_choice = st.radio(
            "Eval judge",
            options=["Gemini (fast ~1s)", "Ollama (slow ~25s)"],
            index=0 if (_ep == "gemini" and _ek) else 1,
            key="eval_judge_radio",
        )
    with col_judge_info:
        if "Gemini" in judge_choice:
            if _ep == "gemini" and _ek:
                st.success(f"Using Gemini judge: `{_em}`", icon="✅")
            else:
                st.warning("Gemini not active. Activate it above first, or switch to Ollama.", icon="⚠️")
        else:
            st.info(f"Using Ollama judge: `{REASONING_MODEL}` — expect ~25s per query", icon="ℹ️")

    use_gemini_judge = "Gemini" in judge_choice and _ep == "gemini" and bool(_ek)
    st.divider()

    col_run, _ = st.columns([1, 2])
    with col_run:
        run_eval = st.button("▶ Run RAG Eval", type="primary", use_container_width=True, key="btn_run_eval")

    if run_eval:
        if not phoenix_ok or not qdrant_ok:
            st.error("Phoenix and Qdrant must both be running.")
        else:
            from qdrant_client import QdrantClient as _QC
            from ollama import Client as _OC
            from phoenix.evals import (
                HallucinationEvaluator, QAEvaluator, RelevanceEvaluator,
                LiteLLMModel, run_evals,
            )

            log_lines = []
            log_area  = st.empty()

            class _StreamCapture(io.StringIO):
                def write(self, s):
                    super().write(s)
                    log_lines.append(s)
                    log_area.code("".join(log_lines), language="bash")

            old_stdout = sys.stdout
            sys.stdout = _StreamCapture()

            try:
                # ── 1. Fetch spans from Phoenix ───────────────────────────
                with st.spinner("Fetching spans from Phoenix…"):
                    spans_df = None
                    try:
                        from phoenix.client import Client as _PhxClient
                        _phx     = _PhxClient(base_url=PHOENIX_BASE_URL)
                        spans_df = _phx.spans.get_spans_dataframe(project_name=PHOENIX_PROJECT_NAME)
                        print(f"Pulled {len(spans_df)} spans via new Phoenix client.")
                    except Exception as e1:
                        print(f"New client failed: {e1}, trying legacy…")
                        try:
                            import phoenix as px
                            from phoenix.trace.dsl import SpanQuery
                            spans_df = px.Client(endpoint=PHOENIX_BASE_URL).query_spans(
                                SpanQuery().select(
                                    input="input.value", output="output.value",
                                    name="name", span_kind="span_kind",
                                ),
                                project_name=PHOENIX_PROJECT_NAME,
                            )
                            print(f"Pulled {len(spans_df)} spans via legacy client.")
                        except Exception as e2:
                            print(f"Legacy client also failed: {e2}")

                if spans_df is None or spans_df.empty:
                    st.error("No spans found. Run some queries in the Agent Chat tab first.")
                else:
                    # ── 2. Normalise column names across Phoenix versions ──
                    def _find_col(df, *candidates):
                        return next((c for c in candidates if c in df.columns), None)

                    input_col  = _find_col(spans_df, "input",  "attributes.input.value",  "input.value")
                    output_col = _find_col(spans_df, "output", "attributes.output.value", "output.value")
                    name_col   = _find_col(spans_df, "name",   "span_name")
                    print(f"Column mapping: input←'{input_col}', output←'{output_col}', name←'{name_col}'")

                    if not input_col or not output_col:
                        st.error(
                            f"Cannot find input/output columns.\n"
                            f"Available: {list(spans_df.columns)}\n"
                            f"Run some queries in Agent Chat first."
                        )
                        st.stop()

                    rename = {k: v for k, v in [
                        (input_col,  "input")  if input_col  != "input"  else (None, None),
                        (output_col, "output") if output_col != "output" else (None, None),
                        (name_col,   "name")   if name_col and name_col != "name" else (None, None),
                    ] if k}
                    if rename:
                        spans_df = spans_df.rename(columns=rename)

                    # ── 3. Keep root spans only (filter out LangGraph child nodes) ──
                    parent_col = _find_col(spans_df, "parent_id", "context.parent_id", "parent_span_id")
                    if parent_col:
                        root_df  = spans_df[spans_df[parent_col].isna()]
                        spans_df = root_df if not root_df.empty else spans_df
                        print(f"Root spans after filtering: {len(spans_df)}")

                    # Filter by span kind (prefer TOOL > CHAIN > LLM)
                    kind_col  = _find_col(spans_df, "span_kind", "context.span_kind", "kind",
                                          "attributes.openinference.span.kind")
                    target_df = spans_df
                    if kind_col:
                        for kind in ["TOOL", "CHAIN", "LLM"]:
                            f = spans_df[spans_df[kind_col] == kind]
                            if not f.empty:
                                target_df = f
                                print(f"Using {len(f)} {kind} spans.")
                                break

                    # Drop rows with empty or error input/output
                    target_df = (
                        target_df
                        .dropna(subset=["input", "output"])
                        .pipe(lambda d: d[d["input"].astype(str).str.strip() != ""])
                        .pipe(lambda d: d[d["output"].astype(str).str.strip() != ""])
                        .pipe(lambda d: d[~d["output"].astype(str).str.startswith("Error:")])
                    )

                    if target_df.empty:
                        st.error("No usable spans after filtering. Run some queries first.")
                    else:
                        # ── 4. Re-fetch context from Qdrant for each span ──
                        print(f"Evaluating {len(target_df)} spans…")
                        _qc = _QC(url=QDRANT_URL)
                        _oc = _OC(host=OLLAMA_HOST)

                        def _embed(text: str):
                            return _oc.embed(model=EMBED_MODEL, input=text).embeddings[0]

                        def _extract_question(raw: str) -> str:
                            """
                            CHAIN spans store the full LangGraph state as JSON.
                            Extract just the plain question string from it.
                            """
                            import json as _json
                            raw = raw.strip()
                            try:
                                obj = _json.loads(raw)
                                if isinstance(obj, dict):
                                    for key in ("question", "query", "input", "text", "message"):
                                        if key in obj and isinstance(obj[key], str):
                                            return obj[key].strip()
                                    for v in obj.values():
                                        if isinstance(v, str) and len(v) > 5:
                                            return v.strip()
                            except (_json.JSONDecodeError, TypeError):
                                pass
                            return raw

                        def _get_context(query_text: str) -> str:
                            """Re-retrieve context from Qdrant for a given question."""
                            vec = _embed(query_text)
                            try:
                                pts = _qc.query_points(
                                    collection_name=COLLECTION_NAME,
                                    query=vec, using="dense", limit=5,
                                )
                            except Exception:
                                pts = _qc.query_points(
                                    collection_name=COLLECTION_NAME, query=vec, limit=5,
                                )
                            return "\n".join(p.payload.get("original_text", "") for p in pts.points)

                        rows = []
                        for span_id, row in target_df.iterrows():
                            raw_input = str(row.get("input",  "")).strip()
                            answer    = str(row.get("output", "")).strip()
                            if not raw_input or not answer:
                                continue
                            q_text = _extract_question(raw_input)
                            print(f"  Context for: {q_text[:60]}…")
                            ctx = _get_context(q_text)
                            rows.append({
                                "context.span_id": span_id,
                                "input":           q_text,
                                "output":          answer,
                                "context":         ctx,
                                "reference":       ctx,
                                "tool_name":       str(row.get("name", "")),
                            })

                        eval_df_raw = pd.DataFrame(rows)

                        # Deduplicate: same question can appear across multiple LangGraph nodes.
                        # Keep the version with the longest (most complete) output.
                        before = len(eval_df_raw)
                        eval_df_raw["_len"] = eval_df_raw["output"].str.len()
                        eval_df_raw = (
                            eval_df_raw
                            .sort_values("_len", ascending=False)
                            .drop_duplicates(subset=["input"], keep="first")
                            .drop(columns=["_len"])
                            .reset_index(drop=True)
                        )
                        print(f"Deduplicated {before} spans → {len(eval_df_raw)} unique questions.")

                        # Index by span_id so evaluator results stay aligned
                        eval_df = eval_df_raw.set_index("context.span_id")
                        print(f"Prepared {len(eval_df)} rows for eval.")

                        # ── 5. Run LLM-as-judge evals ─────────────────────
                        if use_gemini_judge:
                            os.environ["GEMINI_API_KEY"] = _ek
                            judge_model = LiteLLMModel(model=f"gemini/{_em}", max_tokens=512)
                            judge_label = f"gemini/{_em}"
                        else:
                            os.environ["OLLAMA_API_BASE"] = OLLAMA_HOST
                            judge_model = LiteLLMModel(model=f"ollama/{REASONING_MODEL}", max_tokens=512)
                            judge_label = f"ollama/{REASONING_MODEL}"

                        print(f"Running evals with judge: {judge_label}")

                        with st.spinner(f"Scoring with {judge_label}…"):
                            hallucination_df, qa_df, relevance_df = run_evals(
                                dataframe=eval_df,
                                evaluators=[
                                    HallucinationEvaluator(judge_model),
                                    QAEvaluator(judge_model),
                                    RelevanceEvaluator(judge_model),
                                ],
                                provide_explanation=True,
                                concurrency=1,
                            )

                        # Compute aggregate percentages for the metrics cards
                        h = hallucination_df["label"].value_counts(normalize=True).get("hallucinated", 0)
                        q = qa_df["label"].value_counts(normalize=True).get("correct", 0)
                        r = relevance_df["label"].value_counts(normalize=True).get("relevant", 0)

                        st.session_state.eval_scores = {
                            "label":              judge_label,
                            "hallucination_df":   hallucination_df,
                            "qa_df":              qa_df,
                            "relevance_df":       relevance_df,
                            "hallucination_rate": round(h * 100, 1),
                            "qa_correctness":     round(q * 100, 1),
                            "context_relevance":  round(r * 100, 1),
                        }

                        # ── 6. Log eval results back to Phoenix as annotations ──
                        # After this step each trace in Phoenix will show:
                        #   hallucination / qa_correctness / context_relevance
                        # in the Annotations panel on the right side of the trace view.
                        print("Uploading annotations to Phoenix…")
                        with st.spinner("Logging annotations to Phoenix…"):
                            ok, log_msg = log_evaluations_to_phoenix(
                                eval_df,
                                hallucination_df,
                                qa_df,
                                relevance_df,
                            )
                        if ok:
                            st.success(log_msg)
                            print(log_msg)
                        else:
                            st.warning(log_msg)
                            print(log_msg)

                        # Build per-row detail table for the breakdown section below
                        detail = eval_df_raw[["input", "output"]].copy()
                        detail["hallucination"]     = hallucination_df["label"].values
                        detail["hallucination_exp"] = hallucination_df.get("explanation", pd.Series()).values
                        detail["qa_label"]          = qa_df["label"].values
                        detail["qa_exp"]            = qa_df.get("explanation", pd.Series()).values
                        detail["relevance_label"]   = relevance_df["label"].values
                        detail["relevance_exp"]     = relevance_df.get("explanation", pd.Series()).values
                        st.session_state.eval_detail_df = detail.reset_index(drop=True)

            except Exception as e:
                st.error(f"Eval failed: {e}")
            finally:
                sys.stdout = old_stdout
                st.session_state.eval_log = "".join(log_lines)

    # ── Display eval results ──────────────────────────────────────────────────
    if st.session_state.eval_scores:
        scores = st.session_state.eval_scores
        st.divider()
        st.markdown(f"#### Results — judge: `{scores['label']}`")

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Hallucination Rate", f"{scores['hallucination_rate']}%")
        m1.caption("Claims not supported by context — **lower is better**")
        m2.metric("🟢 QA Correctness",     f"{scores['qa_correctness']}%")
        m2.caption("Answer correctly addresses the question — **higher is better**")
        m3.metric("🔵 Context Relevance",  f"{scores['context_relevance']}%")
        m3.caption("Retrieved chunks relevant to the query — **higher is better**")

        # Per-query breakdown table with colour-coded hallucination column
        if st.session_state.eval_detail_df is not None:
            st.markdown("#### Per-query breakdown")
            detail = st.session_state.eval_detail_df

            def _hl_style(val):
                if str(val).lower() == "hallucinated":
                    return "background-color: #FFCCCC"
                if str(val).lower() in ("not_hallucinated", "not hallucinated", "grounded"):
                    return "background-color: #CCFFCC"
                return ""

            show_cols = [c for c in ["input", "hallucination", "qa_label", "relevance_label", "output"]
                         if c in detail.columns]
            styled = detail[show_cols].style.applymap(
                _hl_style, subset=["hallucination"] if "hallucination" in show_cols else []
            )
            st.dataframe(styled, use_container_width=True, height=350)

            with st.expander("Full explanations", expanded=False):
                exp_cols = [c for c in [
                    "input", "hallucination", "hallucination_exp",
                    "qa_label", "qa_exp", "relevance_label", "relevance_exp",
                ] if c in detail.columns]
                st.dataframe(detail[exp_cols], use_container_width=True, height=400)

        st.markdown(f"[View annotations in Phoenix → Traces]({PHOENIX_BASE_URL}/projects)")

    if st.session_state.eval_log:
        with st.expander("Run log", expanded=False):
            st.code(st.session_state.eval_log, language="bash")
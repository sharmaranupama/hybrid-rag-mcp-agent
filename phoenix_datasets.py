"""
phoenix_datasets.py — Manage evaluation datasets and experiments in Phoenix (Arize).

Commands:
  python phoenix_datasets.py create-dataset   # Upload TEST_EXAMPLES to Phoenix
  python phoenix_datasets.py run-experiment   # Query the RAG and score every example
  python phoenix_datasets.py show-datasets    # List all datasets in Phoenix
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from config import (
    REASONING_MODEL, OLLAMA_HOST,
    PHOENIX_BASE_URL, PHOENIX_PROJECT_NAME,
    PHOENIX_DATASET_NAME, PHOENIX_EXPERIMENT_PREFIX,
    llm_generate, get_provider, get_gemini_model, DEFAULT_GEMINI_MODEL,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

# ── Test dataset ──────────────────────────────────────────────────────────────
# These are the sample questions used for evaluation.
# "in-scope"   — questions the system should answer confidently
# "comparison" — questions requiring year-over-year comparison
# "out-of-scope" — questions where no relevant data exists

TEST_EXAMPLES = [
    {"input": "What is the remote work policy?",                     "expected_topic": "Remote Policy",         "category": "in-scope"},
    {"input": "How many days of vacation do employees get?",         "expected_topic": "Vacation Policy (PTO)", "category": "in-scope"},
    {"input": "What does the health insurance cover?",               "expected_topic": "Health Benefits",       "category": "in-scope"},
    {"input": "What is the travel reimbursement rate per mile?",     "expected_topic": "Travel Reimbursement",  "category": "in-scope"},
    {"input": "Is multi-factor authentication required?",            "expected_topic": "Security Protocols",    "category": "in-scope"},
    {"input": "What is the parental leave policy?",                  "expected_topic": "Parental Leave",        "category": "in-scope"},
    {"input": "Compare vacation policy between 2024 and 2026",       "expected_topic": "Vacation Policy (PTO)", "category": "comparison"},
    {"input": "What changed in health insurance from 2024 to 2026?", "expected_topic": "Health Benefits",       "category": "comparison"},
    {"input": "What is the dental insurance deductible amount?",     "expected_topic": None,                    "category": "out-of-scope"},
    {"input": "What is the company policy on stock options?",        "expected_topic": None,                    "category": "out-of-scope"},
]


# ── Phoenix client helpers ────────────────────────────────────────────────────

def _new_client():
    """Return a new-style Phoenix client (arize-phoenix-client >= 1.22)."""
    from phoenix.client import Client
    return Client(base_url=PHOENIX_BASE_URL)

def _legacy_client():
    """Return a legacy Phoenix client (arize-phoenix)."""
    import phoenix as px
    return px.Client(endpoint=PHOENIX_BASE_URL)


# ── Commands ──────────────────────────────────────────────────────────────────

def create_dataset():
    """
    Upload TEST_EXAMPLES to Phoenix as a named evaluation dataset.
    Tries the new client API first, falls back to the legacy API,
    and finally saves a CSV if neither works.
    """
    inputs   = [{"question": ex["input"]} for ex in TEST_EXAMPLES]
    metadata = [{"expected_topic": ex["expected_topic"] or "none", "category": ex["category"]}
                for ex in TEST_EXAMPLES]

    print(f"Creating dataset '{PHOENIX_DATASET_NAME}' in Phoenix...")

    # --- New client API ---
    try:
        client  = _new_client()
        dataset = client.datasets.create_dataset(
            name=PHOENIX_DATASET_NAME,
            dataset_description="Company policy RAG evaluation",
            inputs=inputs,
            metadata=metadata,
        )
        print(f"  Created '{dataset.name}' with {len(dataset)} examples.")
        print(f"  View at {PHOENIX_BASE_URL}/datasets")
        return
    except Exception as e:
        # If the dataset already exists, append to it instead
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            try:
                _new_client().append_to_dataset(
                    dataset_name=PHOENIX_DATASET_NAME, inputs=inputs, metadata=metadata
                )
                print(f"  Appended {len(inputs)} examples to existing dataset.")
                print(f"  View at {PHOENIX_BASE_URL}/datasets")
                return
            except Exception as e2:
                print(f"  Append failed: {e2}")
        else:
            print(f"  New client error: {e}")

    # --- Legacy client API ---
    try:
        _legacy_client().upload_dataset(
            dataset_name=PHOENIX_DATASET_NAME, inputs=inputs, metadata=metadata
        )
        print(f"  Created via legacy API.")
        print(f"  View at {PHOENIX_BASE_URL}/datasets")
        return
    except Exception as e:
        print(f"  Legacy upload error: {e}")

    # --- CSV fallback ---
    print("\n  Could not create dataset via API. Saving as CSV for manual upload...")
    os.makedirs("experiments", exist_ok=True)
    csv_path = "experiments/eval_dataset.csv"
    pd.DataFrame([
        {"question": ex["input"], "expected_topic": ex["expected_topic"] or "none", "category": ex["category"]}
        for ex in TEST_EXAMPLES
    ]).to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}. Upload manually at {PHOENIX_BASE_URL}/datasets")


def run_experiment():
    """
    Run a RAG evaluation experiment:
      1. For each test question, retrieve context and generate an answer.
      2. Score answers with LLM-as-judge (hallucination, QA correctness, context relevance).
      3. Save results locally as CSV and upload to Phoenix.
    """
    from query import fetch_context_with_scores

    provider  = get_provider()
    model     = get_gemini_model() if provider == "gemini" else REASONING_MODEL
    exp_name  = f"{PHOENIX_EXPERIMENT_PREFIX}-{provider}-{model}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    print(f"\nRunning experiment: {exp_name}")
    print(f"Provider: {provider}, Model: {model}\n{'='*60}")

    results = []

    for i, ex in enumerate(TEST_EXAMPLES):
        question = ex["input"]
        print(f"\n[{i+1}/{len(TEST_EXAMPLES)}] {question}")
        t0 = time.time()

        # Retrieve context
        try:
            chunks = fetch_context_with_scores(question)
        except Exception as e:
            print(f"  Retrieval error: {e}")
            chunks = []

        context = "\n".join(c["text"] for c in chunks)

        # Generate answer
        if context:
            prompt = (
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                f"Answer using ONLY the context. Include all years if multiple exist.\nAnswer:"
            )
            try:
                answer = llm_generate(prompt)
            except Exception as e:
                answer = f"LLM error: {e}"
        else:
            answer = "I couldn't find any relevant company information."

        latency     = round(time.time() - t0, 2)
        top_score   = max((c["score"] for c in chunks), default=0.0)
        topics_hit  = [c["topic"] for c in chunks]
        topic_found = ex["expected_topic"] is not None and ex["expected_topic"] in topics_hit

        results.append({
            "question":    question,
            "answer":      answer,
            "context":     context,
            "category":    ex["category"],
            "topic_found": topic_found,
            "top_score":   round(top_score, 4),
            "latency_s":   latency,
            "num_chunks":  len(chunks),
            "provider":    provider,
            "model":       model,
        })

        print(f"  Chunks: {len(chunks)}, Top score: {top_score:.4f}, Topic found: {topic_found}")
        print(f"  Answer: {answer[:120]}...")
        print(f"  Latency: {latency}s")

    # Save results locally
    df = pd.DataFrame(results)
    os.makedirs("experiments", exist_ok=True)
    csv_path = f"experiments/{exp_name}.csv"
    df.to_csv(csv_path, index=False)

    # Print per-category summary
    print(f"\n\n{'='*60}\nEXPERIMENT RESULTS: {exp_name}\n{'='*60}")
    for cat in ["in-scope", "comparison", "out-of-scope"]:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        print(f"\n  {cat} ({len(cat_df)} queries):")
        print(f"    Avg top score  : {cat_df['top_score'].mean():.4f}")
        print(f"    Avg latency    : {cat_df['latency_s'].mean():.2f}s")
        if cat != "out-of-scope":
            print(f"    Topic hit rate : {cat_df['topic_found'].mean()*100:.0f}%")

    # LLM-as-judge evaluation
    eval_results = {}
    try:
        from phoenix.evals import (
            HallucinationEvaluator, QAEvaluator, RelevanceEvaluator,
            LiteLLMModel, run_evals,
        )

        # Rename columns to what Phoenix evaluators expect
        eval_df = df.rename(columns={"question": "input", "answer": "output", "context": "reference"})
        if "context" not in eval_df.columns:
            eval_df["context"] = eval_df["reference"]
        eval_df = eval_df.reset_index(drop=True)

        # Prepare judges: always Ollama, add Gemini if key is available
        judges = {}
        os.environ["OLLAMA_API_BASE"] = OLLAMA_HOST
        judges["ollama"] = {"label": f"ollama/{REASONING_MODEL}",
                            "model": LiteLLMModel(model=f"ollama/{REASONING_MODEL}", max_tokens=512)}
        if GEMINI_API_KEY:
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
            judges["gemini"] = {"label": f"gemini/{GEMINI_MODEL}",
                                "model": LiteLLMModel(model=f"gemini/{GEMINI_MODEL}", max_tokens=512)}

        for key, judge in judges.items():
            print(f"\n  Scoring with judge: {judge['label']}")
            h_df, q_df, r_df = run_evals(
                dataframe=eval_df,
                evaluators=[
                    HallucinationEvaluator(judge["model"]),
                    QAEvaluator(judge["model"]),
                    RelevanceEvaluator(judge["model"]),
                ],
                provide_explanation=True,
                concurrency=1,
            )
            h = h_df["label"].value_counts(normalize=True).get("hallucinated", 0)
            q = q_df["label"].value_counts(normalize=True).get("correct", 0)
            r = r_df["label"].value_counts(normalize=True).get("relevant", 0)
            eval_results[key] = {
                "label":              judge["label"],
                "hallucination_rate": round(h * 100, 1),
                "qa_correctness":     round(q * 100, 1),
                "context_relevance":  round(r * 100, 1),
            }
            print(f"    Hallucination: {eval_results[key]['hallucination_rate']}%")
            print(f"    QA Correct:    {eval_results[key]['qa_correctness']}%")
            print(f"    Ctx Relevance: {eval_results[key]['context_relevance']}%")

    except Exception as e:
        print(f"  Eval scoring failed: {e}")

    # Upload experiment results to Phoenix
    try:
        _new_client().datasets.create_dataset(
            name=exp_name,
            dataset_description=f"Experiment results: {provider}/{model}",
            inputs=[{"question": r["question"]} for r in results],
            outputs=[{"answer": r["answer"]}    for r in results],
            metadata=[{
                "category":    r["category"],
                "topic_found": str(r["topic_found"]),
                "top_score":   str(r["top_score"]),
                "latency_s":   str(r["latency_s"]),
                "provider":    r["provider"],
                "model":       r["model"],
            } for r in results],
        )
        print(f"  Uploaded results to Phoenix as dataset '{exp_name}'.")
    except Exception as e:
        print(f"  Phoenix upload failed: {e}. Results are in {csv_path}.")

    print(f"\nView at {PHOENIX_BASE_URL}/datasets")
    return {"exp_name": exp_name, "csv_path": csv_path, "results_df": df, "eval_results": eval_results}


def show_datasets():
    """List all datasets currently stored in Phoenix."""
    # Try new client first, fall back to legacy
    for get_client, list_fn in [
        (_new_client, lambda c: c.datasets.list()),
        (_legacy_client, lambda c: c.list_datasets()),
    ]:
        try:
            client   = get_client()
            datasets = list_fn(client)
            if not datasets:
                print("No datasets found.")
                return
            print(f"\nDatasets in Phoenix ({PHOENIX_BASE_URL}):")
            for d in datasets:
                name  = d.get("name", "?") if isinstance(d, dict) else getattr(d, "name", "?")
                count = d.get("example_count", "?") if isinstance(d, dict) else getattr(d, "example_count", "?")
                print(f"  - {name} ({count} examples)")
            return
        except Exception:
            continue

    print(f"Could not list datasets. Is Phoenix running at {PHOENIX_BASE_URL}?")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    commands = {"create-dataset": create_dataset, "run-experiment": run_experiment, "show-datasets": show_datasets}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python phoenix_datasets.py <command>")
        print("Commands:", ", ".join(commands))
        sys.exit(1)
    commands[sys.argv[1]]()

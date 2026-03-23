"""
config.py — Central settings: model names, service URLs, and the LLM abstraction.

This is the single file you edit to switch between Ollama (local) and Gemini (cloud).
Everything else imports from here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────

REASONING_MODEL = "llama3.2"          # Ollama model used for text generation
EMBED_MODEL     = "nomic-embed-text"  # Ollama model used for dense embeddings
VECTOR_SIZE     = 768                  # Must match nomic-embed-text output dimension

# Available Gemini models (displayed in the UI dropdown)
GEMINI_MODELS = {
    "gemini-2.5-flash-lite": "2.5 Flash-Lite (stable, fast)",
    "gemini-2.5-flash":      "2.5 Flash (stable, balanced)",
    "gemini-3.1-flash-lite-preview": "3.1 Flash-Lite (preview, fastest)",
    "gemini-3-flash-preview":        "3 Flash (preview, powerful)",
}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

# ── Qdrant collection ─────────────────────────────────────────────────────────

COLLECTION_NAME   = "policy_data_collection"
RETRIEVAL_LIMIT   = 5    # Number of chunks returned per query
INGEST_BATCH_SIZE = 10   # Rows processed per Qdrant upsert batch

# Named vector spaces inside the collection
DENSE_VECTOR_NAME  = "dense"  # Semantic (nomic-embed-text) vectors
SPARSE_VECTOR_NAME = "bm25"   # Keyword (BM25) vectors

ENABLE_QUANTIZATION = True   # int8 scalar quantization — ~4x memory saving
HYBRID_SEARCH_LIMIT = 20     # How many candidates each sub-query fetches before RRF fusion

# ── Service URLs ──────────────────────────────────────────────────────────────

QDRANT_URL  = os.getenv("QDRANT_URL",  "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
)
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "mcp-company-knowledge-server")


def _detect_phoenix_url() -> str:
    """
    Find a reachable Phoenix URL.
    Tries the env override first, then localhost, then host.docker.internal.
    Falls back to localhost if nothing responds.
    """
    import urllib.request
    candidates = []
    env_url = os.getenv("PHOENIX_BASE_URL", "")
    if env_url:
        candidates.append(env_url)
    candidates += ["http://localhost:6006", "http://host.docker.internal:6006"]
    for url in candidates:
        try:
            urllib.request.urlopen(url, timeout=2)
            return url
        except Exception:
            continue
    return env_url or "http://localhost:6006"

PHOENIX_BASE_URL = _detect_phoenix_url()

# Phoenix dataset/experiment names used when saving eval results
PHOENIX_DATASET_NAME      = "policy-eval-dataset"
PHOENIX_EXPERIMENT_PREFIX = "rag-experiment"

# ── LLM provider state ────────────────────────────────────────────────────────
# These module-level variables are mutated at runtime by set_provider().

_active_provider   = os.getenv("LLM_PROVIDER", "ollama")
_gemini_api_key    = os.getenv("GEMINI_API_KEY", "")
_gemini_model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def set_provider(provider: str, gemini_api_key: str = "", gemini_model: str = ""):
    """Switch the active LLM provider at runtime (called from the Streamlit UI)."""
    global _active_provider, _gemini_api_key, _gemini_model_name
    _active_provider = provider
    if gemini_api_key:
        _gemini_api_key = gemini_api_key
    if gemini_model:
        _gemini_model_name = gemini_model

def get_provider() -> str:
    return _active_provider

def get_gemini_api_key() -> str:
    return _gemini_api_key

def get_gemini_model() -> str:
    return _gemini_model_name


# ── LLM generation ────────────────────────────────────────────────────────────

def llm_generate(prompt: str, json_mode: bool = False) -> str:
    """
    Generate text with the currently active provider.
    Routes to Gemini if configured, otherwise falls back to Ollama.
    """
    if _active_provider == "gemini" and _gemini_api_key:
        return _gemini_generate(prompt, json_mode)
    return _ollama_generate(prompt, json_mode)


def _ollama_generate(prompt: str, json_mode: bool = False) -> str:
    """Call the local Ollama server to generate text."""
    from ollama import Client as OllamaClient
    client = OllamaClient(host=OLLAMA_HOST)
    kwargs = {"model": REASONING_MODEL, "prompt": prompt}
    if json_mode:
        kwargs["format"] = "json"  # Force JSON-only output
    return client.generate(**kwargs)["response"]


def _gemini_generate(prompt: str, json_mode: bool = False) -> str:
    """Call the Google Gemini API to generate text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_api_key)
    config = None
    if json_mode:
        config = types.GenerateContentConfig(response_mime_type="application/json")
    response = client.models.generate_content(
        model=_gemini_model_name,
        contents=prompt,
        config=config,
    )
    return response.text

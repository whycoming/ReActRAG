import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Data paths
MARKDOWN_DIR = BASE_DIR / "financebench_output" / "markdown"
MINI_MARKDOWN_DIR = BASE_DIR / "financebench_output" / "eval" / "mini_markdown"
EVAL_DIR = BASE_DIR / "financebench_output" / "eval"
INDEX_DIR = BASE_DIR / "indexes"
RESULTS_DIR = BASE_DIR / "results"
SCHEMAS_DIR = BASE_DIR / "schemas"

# Default schema files
DEFAULT_INDEX_SCHEMA    = SCHEMAS_DIR / "default_index_schema.yaml"
DEFAULT_QUERY_SCHEMA    = SCHEMAS_DIR / "default_query_schema.yaml"
DEFAULT_SEARCH_SCHEMA   = SCHEMAS_DIR / "default_search_schema.yaml"
DEFAULT_ANSWER_SCHEMA   = SCHEMAS_DIR / "default_answer_schema.yaml"
DEFAULT_JUDGE_SCHEMA    = SCHEMAS_DIR / "default_judge_schema.yaml"
DEFAULT_PIPELINE_SCHEMA = SCHEMAS_DIR / "default_pipeline.yaml"

# LLM provider: "anthropic" | "qwen" | "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # Qwen
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Model names per provider and role
MODELS = {
    "anthropic": {
        "indexer": "claude-haiku-4-5-20251001",
        "searcher": "claude-sonnet-4-6",
        "answerer": "claude-sonnet-4-6",
        "judge": "claude-haiku-4-5-20251001",
    },
    "qwen": {
        "indexer": "qwen-turbo",
        "searcher": "qwen-plus",
        "answerer": "qwen-plus",
        "judge": "qwen-turbo",
    },
    "deepseek": {
        "indexer": "deepseek-chat",
        "searcher": "deepseek-chat",
        "answerer": "deepseek-chat",
        "judge": "deepseek-chat",
    },
}

def get_model(role: str) -> str:
    return MODELS[LLM_PROVIDER][role]

# Token chunking
CHUNK_TOKENS = 800
OVERLAP_TOKENS = 150
APPROX_CHARS_PER_TOKEN = 3.5  # fast estimation without tokenizer

# ReAct agent
MAX_REACT_STEPS = 12
TOP_K_CHUNKS = 5

# Evaluation
LLM_JUDGE_TEMPERATURE = 0.0
INDEXER_TEMPERATURE = 0.0
SEARCHER_TEMPERATURE = 0.0
ANSWERER_TEMPERATURE = 0.0

"""
Experiment configuration for InstrumentalEval × OLMo-3.

Edit this file to change experiment parameters. Secrets (API keys, tokens) are
loaded from environment variables only — never hardcode them here.

Environment variables expected:
    OPENAI_API_KEY   – required, used by GPT-4o judge
    GEMINI_API_KEY   – required if running the Gemini second judge
    HF_TOKEN         – optional, speeds up HuggingFace downloads / private models

You can put them in a .env file at the repo root (see .env.example) and they
will be loaded automatically.
"""

import os
from pathlib import Path

# ── Optional: load .env file if python-dotenv is installed ────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Model size: "7B" (fits A100 40GB, ~14 GB VRAM in bfloat16)
# "32B" requires A100 80GB or 2xGPU
MODEL_SIZE = "7B"

# Named post-training checkpoints to evaluate
CHECKPOINTS = ["SFT", "DPO", "RLVR"]

# Step-wise RLVR checkpoints
# RLVR_STEPS is auto-populated by discover_rlvr_steps() in functions.py.
# You can override it manually here.
# Example: RLVR_STEPS = ["step_50", "step_150", "step_300"]
EVAL_RLVR_STEPS = True
RLVR_STEPS = []
MAX_RLVR_STEPS = 15
STEP_SAMPLING_STRATEGY = "front_loaded"  # "even" or "front_loaded"

# Prompt conditions
RUN_GOAL_NUDGE = False
RUN_ANTI_IC = False

# Judge model
JUDGE_MODEL = "gpt-4o"

# Output directory — override via env var OUTPUT_DIR if you want to use a
# different location. Default points to a local results folder.
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    str(Path(__file__).parent / "results"),
)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Speed settings
JUDGE_CONCURRENCY = 5

# Generation settings (paper defaults — do not change for reproducibility)
GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS = 400
JUDGE_TEMPERATURE = 0.0

# A100 40GB memory settings
USE_FLASH_ATTN = False
AGGRESSIVE_VRAM_CLEANUP = True

# Benchmark repo location (cloned at runtime by load_benchmark_tasks)
BENCHMARK_REPO_URL = "https://github.com/yf-he/InstrumentalEval"
BENCHMARK_DIR = os.environ.get(
    "BENCHMARK_DIR",
    "/content/InstrumentalEval/benchmark",
)

# Tagged task spreadsheet (optional — used by Section 6b)
# Set to a local path or env var. Leave None to skip tag merging.
TASK_TAGGING_XLSX = os.environ.get("TASK_TAGGING_XLSX", None)


# ============================================================
# SECRETS — loaded from environment only
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def assert_required_secrets(require_gemini: bool = False) -> None:
    """Fail loudly with a clear message if required secrets are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if require_gemini and not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            f"Set them in your shell or in a .env file at the repo root.\n"
            f"See .env.example for the expected format."
        )

"""Central configuration for data generation, services, and filtering."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load local configuration without overriding explicit environment variables.
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with _env_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

TOTAL_SAMPLES = 100_000
BATCH_SIZE = 1
ROUND_SIZE = 100

# Large seed pools reduce repeated entities and templated generation.
SEED_TARGET_COUNT = 3_000
SEED_PER_ROUND = 200

# OpenRouter takes precedence when both values are configured.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "google/gemma-4-31b-it")
VLLM_START_SCRIPT = os.environ.get(
    "VLLM_START_SCRIPT", str(PROJECT_ROOT / "pipeline" / "scripts" / "start_vllm.sh")
)
VLLM_PID_FILE = os.environ.get(
    "VLLM_PID_FILE", str(PROJECT_ROOT / "pipeline" / "runtime" / "vllm.pid")
)
VLLM_LOG_FILE = os.environ.get(
    "VLLM_LOG_FILE", str(PROJECT_ROOT / "pipeline" / "runtime" / "vllm.log")
)
VLLM_HEALTH_TIMEOUT = float(os.environ.get("VLLM_HEALTH_TIMEOUT", "300"))
VLLM_HEALTH_INTERVAL = float(os.environ.get("VLLM_HEALTH_INTERVAL", "2"))
VLLM_STOP_TIMEOUT = float(os.environ.get("VLLM_STOP_TIMEOUT", "30"))

USE_OPENROUTER = bool(OPENROUTER_API_KEY and OPENROUTER_MODEL)
if USE_OPENROUTER:
    LLM_BASE_URL = OPENROUTER_BASE_URL
    LLM_MODEL_NAME = OPENROUTER_MODEL
    LLM_API_KEY = OPENROUTER_API_KEY
else:
    LLM_BASE_URL = VLLM_BASE_URL
    LLM_MODEL_NAME = VLLM_MODEL_NAME
    LLM_API_KEY = "EMPTY"

# Internal bilingual scene keys sum to the paper's ten scene-level ratios.
SCENE_DISTRIBUTION = {
    "daily_chat": 0.07,
    "english_daily": 0.08,
    "vibe_coding": 0.12,
    "explanation": 0.12,
    "meeting": 0.06,
    "english_meeting": 0.05,
    "customer_service": 0.06,
    "english_customer_service": 0.05,
    "academic": 0.04,
    "english_academic": 0.05,
    "navigation": 0.03,
    "dictation_memo": 0.05,
    "english_dictation": 0.05,
    "voice_search": 0.03,
    "english_voice_search": 0.04,
    "english_tech": 0.10,
}

NOISE_DISTRIBUTION = {"none": 0.20, "light": 0.25, "medium": 0.25, "heavy": 0.30}
ASR_SIMULATION_RATIO = float(os.environ.get("ASR_SIMULATION_RATIO", "0.20"))

# llm_sim reproduces the paper. text_sim and tts_asr are diagnostic extensions.
ASR_MODE = os.environ.get("ASR_MODE", "llm_sim")
ASR_STRENGTH = os.environ.get("ASR_STRENGTH", "medium")
ASR_TTS_PROVIDER = os.environ.get("ASR_TTS_PROVIDER", "moss_api")
ASR_STT_PROVIDER = os.environ.get("ASR_STT_PROVIDER", "qwen3_api")
ASR_TTS_BASE_URL = os.environ.get("ASR_TTS_BASE_URL", "http://127.0.0.1:8011")
ASR_STT_BASE_URL = os.environ.get("ASR_STT_BASE_URL", "http://127.0.0.1:8012")
ASR_HTTP_TIMEOUT = float(os.environ.get("ASR_HTTP_TIMEOUT", "600"))
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")
ASR_CACHE_DIR = os.environ.get(
    "ASR_CACHE_DIR", str(PROJECT_ROOT / "data" / "cache" / "tts_asr")
)
ASR_ENABLE_CACHE = os.environ.get("ASR_ENABLE_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
}
ASR_FALLBACK_TEXT_SIM = os.environ.get("ASR_FALLBACK_TEXT_SIM", "1").lower() not in {
    "0",
    "false",
    "no",
}

PASSTHROUGH_RATIO = float(os.environ.get("PASSTHROUGH_RATIO", "0.08"))
SELF_CORRECTION_RATIO = float(os.environ.get("SELF_CORRECTION_RATIO", "0.18"))
MULTI_CORRECTION_RATIO = float(os.environ.get("MULTI_CORRECTION_RATIO", "0.08"))
EXPLANATION_CORRECTION_RATIO = float(
    os.environ.get("EXPLANATION_CORRECTION_RATIO", "0.60")
)

KEYWORD_MODE = os.environ.get("KEYWORD_MODE", "review_targets")
KEYWORD_INCLUDE_EXPLAINED_ENTITIES = os.environ.get(
    "KEYWORD_INCLUDE_EXPLAINED_ENTITIES", "1"
).lower() not in {"0", "false", "no"}
KEYWORD_QC_MODE = os.environ.get("KEYWORD_QC_MODE", "rule")

TEMPERATURE_CLEAN = 0.8
TEMPERATURE_ORAL = 0.9
TEMPERATURE_SEED = 0.85
LLM_MAX_CONCURRENT = 20
LLM_MAX_TOKENS_SEED = 6_164
LLM_MAX_TOKENS_ORAL = 4_096
LLM_MAX_TOKENS_CLEAN = 4_096
LLM_MAX_TOKENS_QC = 1_024
LLM_MAX_TOKENS_DEFAULT = 4_096
DEDUP_THRESHOLD = 0.75

DATA_DIR = str(PROJECT_ROOT / "data")
SEEDS_DIR = str(Path(DATA_DIR) / "seeds")
RAW_DIR = str(Path(DATA_DIR) / "raw")
PROCESSED_DIR = str(Path(DATA_DIR) / "processed")
FINAL_DIR = str(Path(DATA_DIR) / "final")

# Kept for compatibility with the reset script.
PROTECTED_DATA_DIRS = ["datav0", "datav1"]

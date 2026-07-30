# Data Pipeline

This module creates Refiner training pairs from scene prompts and LLM-generated transcripts.

## Workflow

`run_pipeline.py` runs five stages:

1. Seed and Oral generation.
2. Oral-to-Clean generation.
3. Aligned ASR simulation.
4. Record assembly.
5. Semantic QC, deduplication, and final export.

Run it from the repository root with an OpenAI-compatible LLM service configured through `VLLM_BASE_URL` and `VLLM_MODEL_NAME`.

## Files

- `run_pipeline.py`: orchestrates the five stages.
- `configs/`: scene registry, prompts, and runtime configuration.
- `scripts/`: stage entry points, SFT export, vLLM launcher, and MLX conversion.
- `src/generators/`: seed, Oral, Clean, and ASR-simulation generators.
- `src/pipeline/`: quality-control and deduplication logic.
- `src/services/`: LLM and vLLM lifecycle clients.
- `src/speech/`: optional TTS/ASR service adapters.
- `src/utils/`: JSONL, prompt, keyword, and logging utilities.

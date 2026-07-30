# Pipeline Scripts

- `01_seed_and_oral.py`: generate scene seed pools and spoken transcripts.
- `02_clean_text.py`: generate Clean targets from Oral transcripts.
- `03_simulate_asr.py`: create aligned ASR-style inputs.
- `04_assemble.py`: convert per-scene outputs to the common record schema.
- `05_finalize.py`: run semantic QC, deduplication, distribution checks, and write `data/final/train.jsonl`.
- `export_sft.py`: export finalized records to LLaMA Factory ShareGPT JSON.
- `start_vllm.sh`: launch the local OpenAI-compatible generation service.
- `convert_to_mlx.py`: convert a Hugging Face Refiner checkpoint for MLX-LM on macOS.
- `reset_data.sh`: clear generated working data before a new run.

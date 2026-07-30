# Inference Scripts

- `postprocess_asr.py`: batch Transformer Refiner inference.
- `postprocess_contract.py`: input/output validation for Refiner inference.
- `main.py`: benchmark judge entry point.
- `benchmark_io.py`: benchmark JSON/JSONL schema parsing and record matching.
- `client.py`: OpenAI-compatible judge client.
- `result_store.py`: resumable judge results and summaries.
- `metrics.py`: benchmark aggregate metrics.
- `text_metrics.py`: WER, CER, and MER utilities.

# AgenticASR

**AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach**.

AgenticASR defines the Agentic Speech Recognition task, provides a data-generation pipeline for training a Refiner, and implements a streaming system that incrementally rewrites ASR hypotheses as context arrives.

<p align="center">
  <img src="assets/teaser6.png" alt="AgenticASR teaser" width="100%">
</p>

<p align="center">
  <img src="assets/AgenticASR-method1.png" alt="AgenticASR method overview" width="100%">
</p>

## Links

- Paper: **link to be added**
- Product and desktop downloads (Windows/macOS): [VibeXASR](https://vibexasr.speech.wiki/)
- Benchmark: [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench) | [Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench)
- Refiner checkpoint: **ModelScope link to be added** | **Hugging Face link to be added**

## Repository Structure

```text
pipeline/       Generate and filter Refiner training pairs.
experiments/    Offline Refiner inference and benchmark evaluation.
system/         Streaming AgenticASR implementation used by the desktop App.
assets/         Paper overview figures.
refiner.yaml    LLaMA Factory training configuration.
requirements.txt
run_pipeline.py Top-level data pipeline entry point.
```

See the README in each module for its file map and implementation details.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r system/requirements.txt
```

## 1. Run the Data Pipeline

Start an OpenAI-compatible vLLM service, or configure an OpenRouter-compatible service, then run:

```bash
export VLLM_MODEL_NAME=/path/to/gemma-4-31b-it
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
python run_pipeline.py
```

The pipeline generates Oral/Clean pairs, simulates ASR noise, assembles records, performs quality control and deduplication, and writes the final data under `data/final/`.

## 2. Run Refiner Inference

For batch inference with a trained Transformer checkpoint:

```bash
python experiments/scripts/postprocess_asr.py \
  path/to/asr_output.jsonl \
  path/to/refined_output.jsonl \
  --model /path/to/refiner-checkpoint
```

The input records must contain `source_record_id` and `output.raw_text`. See `experiments/README.md` for the benchmark judge workflow.

## 3. Train the Refiner

Export finalized pipeline records to the ShareGPT format:

```bash
python pipeline/scripts/export_sft.py \
  --inputs data/final/train.jsonl \
  --train-output data/final/train_sft.json \
  --val-output data/final/val_sft.json
```

Register the generated files in the LLaMA Factory dataset catalog, update the model and output paths in `refiner.yaml`, then run:

```bash
llamafactory-cli train refiner.yaml
```

## Streaming System

`system/` contains the streaming implementation: audio input, VAD, online sherpa-onnx ASR, bounded chunking, and a K=3 sliding-window Refiner. The full local App path uses the MLX Refiner backend on macOS; `--identity-refiner` is only for diagnostics. The packaged Windows/macOS application is available from the [VibeXASR product page](https://vibexasr.speech.wiki/).

```bash
bash system/download_vad.sh models
python -m system.live_asr \
  --wav path/to/example.wav \
  --asr-dir models/asr \
  --refiner models/refiner-mlx
```

See `system/README.md` for model and VAD preparation.

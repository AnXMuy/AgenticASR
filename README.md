# AgenticASR

<p align="center">
  <strong>AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach</strong>
</p>

<p align="center">
  A benchmark, a data simulation pipeline, and a streaming speech recognition system that refines ASR hypotheses as context arrives.
</p>

<p align="center">
  <img src="assets/teaser6.png" alt="AgenticASR teaser" width="100%">
</p>

<p align="center">
  <img src="assets/AgenticASR-method1.png" alt="AgenticASR method overview" width="100%">
</p>

<p align="center">
  <a href="https://vibexasr.speech.wiki/"><img src="https://img.shields.io/badge/Desktop_App-Windows%20%7C%20macOS-111827?style=flat-square" alt="Desktop App"></a>
  <a href="https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench"><img src="https://img.shields.io/badge/Benchmark-AASR--Bench-2563eb?style=flat-square" alt="Benchmark"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-16a34a?style=flat-square" alt="License"></a>
</p>

## Resources

<table>
  <tr>
    <td align="center" width="25%"><b>Paper</b><br><sub>Link coming soon</sub></td>
    <td align="center" width="25%"><a href="https://vibexasr.speech.wiki/"><b>Desktop App</b></a><br><sub>Windows and macOS downloads</sub></td>
    <td align="center" width="25%"><a href="https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench"><b>Benchmark</b></a><br><sub>ModelScope</sub> · <a href="https://huggingface.co/datasets/Andrew0425/AASR-Bench"><sub>Hugging Face</sub></a></td>
    <td align="center" width="25%"><a href="https://www.modelscope.cn/models/MuyuanJ/AgenticASR-Refiner"><b>Refiner Checkpoint</b></a><br><sub>ModelScope</sub> · <a href="https://huggingface.co/Andrew0425/AgenticASR-Refiner/tree/main"><sub>Hugging Face</sub></a></td>
  </tr>
</table>

## What Is Included

<table>
  <tr>
    <td width="25%"><b>Benchmark</b><br><sub>AASR-Bench evaluates content, formatting, filtering, and self-correction.</sub></td>
    <td width="25%"><b>Data Pipeline</b><br><sub>Generate Oral/Clean pairs, simulate ASR noise, run QC, and export training data.</sub></td>
    <td width="25%"><b>Refiner</b><br><sub>Train or run the ASR text correction model with the paper prompt.</sub></td>
    <td width="25%"><b>Streaming System</b><br><sub>VAD, incremental ASR, bounded chunks, and K=3 refinement for the desktop App.</sub></td>
  </tr>
</table>

The packaged Windows/macOS application is available from the [VibeXASR product page](https://vibexasr.speech.wiki/). This repository contains the research code and reproducible core implementation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r system/requirements.txt
```

## Quick Start

<table>
  <tr>
    <td width="33%"><b>01 · Generate</b><br><sub>Create Refiner training pairs.</sub></td>
    <td width="33%"><b>02 · Refine</b><br><sub>Run a trained Refiner on ASR JSONL.</sub></td>
    <td width="33%"><b>03 · Train</b><br><sub>Fine-tune a Refiner with LLaMA Factory.</sub></td>
  </tr>
</table>

### 01 · Generate Training Data

Start an OpenAI-compatible vLLM service, or configure an OpenRouter-compatible service:

```bash
export VLLM_MODEL_NAME=/path/to/gemma-4-31b-it
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
python run_pipeline.py
```

The final records are written to `data/final/`. See [pipeline/README.md](pipeline/README.md) for the stage layout.

### 02 · Run Refiner Inference

The batch inference entry point is `experiments/scripts/postprocess_asr.py`:

```bash
python experiments/scripts/postprocess_asr.py \
  /path/to/asr_output.jsonl \
  /path/to/refined_output.jsonl \
  --model /path/to/refiner-checkpoint
```

Each input record must contain `source_record_id` and `output.raw_text`. The checkpoint uses the Refiner system prompt defined in the inference and training code.

### 03 · Train the Refiner

First export the finalized pipeline records. Use paths that exist on your machine:

```bash
python pipeline/scripts/export_sft.py \
  --inputs /path/to/AgenticASR/data/final/train.jsonl \
  --train-output /path/to/llamafactory-data/train_sft.json \
  --val-output /path/to/llamafactory-data/val_sft.json
```

Register those files in the LLaMA Factory dataset directory at `/path/to/llamafactory-data/dataset_info.json`:

```json
{
  "refiner_train": {"file_name": "train_sft.json"},
  "refiner_val": {"file_name": "val_sft.json"}
}
```

Then edit a copy of `refiner.yaml` and replace every machine-specific path:

```yaml
model_name_or_path: /path/to/base-model
dataset_dir: /path/to/llamafactory-data
dataset: refiner_train
eval_dataset: refiner_val
output_dir: /path/to/refiner-output
```

Launch training with the edited configuration:

```bash
llamafactory-cli train /path/to/refiner.yaml
```

## Streaming AgenticASR

The `system/` implementation is the streaming core of the AgenticASR desktop App. It combines VAD, online sherpa-onnx ASR, stable text chunking, and a K=3 sliding-window Refiner. The current local backend uses MLX-LM on macOS; the full App path requires `--refiner`.

```bash
bash system/download_vad.sh /path/to/models
python -m system.live_asr \
  --wav /path/to/example.wav \
  --asr-dir /path/to/models/asr \
  --refiner /path/to/models/refiner-mlx
```

Use `--identity-refiner` only for ASR/chunking diagnostics. See [system/README.md](system/README.md) for VAD and model preparation.

## Benchmark Evaluation

Download `rubric.json` from [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench) or [Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench), then run the judge in `experiments/scripts/main.py` with `--rubric /path/to/rubric.json`. See [experiments/README.md](experiments/README.md).

## Module Documentation

- [Pipeline](pipeline/README.md)
- [Experiments and inference](experiments/README.md)
- [Streaming system](system/README.md)
- [Assets](assets/README.md)

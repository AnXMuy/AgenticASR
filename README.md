# AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach

<p align="center">
  <a href="https://arxiv.org/html/2607.28175v1"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square&logo=arxiv" alt="Paper"></a>
  <a href="https://anxmuy.github.io/blog/agenticasr/"><img src="https://img.shields.io/badge/Project-Page-176b87?style=flat-square&logo=githubpages" alt="Project Page"></a>
  <a href="https://vibexasr.speech.wiki/"><img src="https://img.shields.io/badge/Desktop_App-VibeXASR-8250df?style=flat-square" alt="VibeXASR Desktop App"></a>
  <a href="https://huggingface.co/datasets/Andrew0425/AASR-Bench"><img src="https://img.shields.io/badge/Benchmark-Hugging_Face-f2c94c?style=flat-square" alt="AASR-Bench on Hugging Face"></a>
  <a href="https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench"><img src="https://img.shields.io/badge/Benchmark-ModelScope-624aff?style=flat-square" alt="AASR-Bench on ModelScope"></a>
  <a href="https://huggingface.co/Andrew0425/AgenticASR-Refiner/tree/main"><img src="https://img.shields.io/badge/Refiner-Hugging_Face-f2c94c?style=flat-square" alt="Refiner on Hugging Face"></a>
  <a href="https://www.modelscope.cn/models/MuyuanJ/AgenticASR-Refiner"><img src="https://img.shields.io/badge/Refiner-ModelScope-624aff?style=flat-square" alt="Refiner on ModelScope"></a>
</p>

## News

- **2026-07-30:** Paper, code, AASR-Bench, and Refiner checkpoint are released.

## Features

**AgenticSR** turns speech into clean written text while preserving the speaker's final intent. It removes disfluencies, resolves self-corrections, normalizes written form, and can revise previously emitted text when later speech adds new evidence.

- **Bilingual:** AgenticASR supports both English and Chinese speech-to-clean-text refinement.
- **ASR-agnostic:** The Refiner is decoupled from the recognizer and can be attached to any ASR frontend that produces text hypotheses.
- **Online + offline:** Refine a complete transcript once, or continually replace a bounded active span as speech arrives.
- **AASR-Bench:** 917 samples and 6,637 atomic rubrics covering Content, Format, Filter, and Rephrase.

## Bilingual Demo

<table>
  <tr>
    <th width="50%">English Demo</th>
    <th width="50%">中文演示</th>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="MediaSup/en_demo.mp4"><img src="MediaSup/en_demo_preview.jpg" alt="Play the English AgenticASR demo" width="100%"></a>
      <br><a href="MediaSup/en_demo.mp4"><b>▶ Play English demo</b></a>
    </td>
    <td width="50%" align="center">
      <a href="MediaSup/CH_demo.mp4"><img src="MediaSup/CH_demo_preview.jpg" alt="播放 AgenticASR 中文演示" width="100%"></a>
      <br><a href="MediaSup/CH_demo.mp4"><b>▶ 播放中文演示</b></a>
    </td>
  </tr>
</table>

We evaluate the Refiner with Qwen3-ASR and Whisper. The packaged Windows/macOS application is available from the [VibeXASR product page](https://vibexasr.speech.wiki/). This repository contains the research code and reproducible core implementation.

## Installation

### Create a separate environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install the dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r system/requirements.txt
```

Refiner training additionally requires [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory). Install it separately before running `llamafactory-cli train`.

## Inference

### 1. Batch Refiner inference

The Refiner accepts JSONL output from any ASR frontend. Each record must contain `source_record_id` and `output.raw_text`.

```bash
python experiments/scripts/postprocess_asr.py \
  /path/to/asr_output.jsonl \
  /path/to/refined_output.jsonl \
  --model /path/to/refiner-checkpoint
```

The checkpoint uses the Refiner system prompt defined in the inference and training code. See [experiments/README.md](experiments/README.md) for the input contract and backend options.

### 2. Streaming AgenticASR

The streaming system combines VAD, an online sherpa-onnx ASR frontend, stable text chunking, and a default `K=3` sliding-window Refiner. The current local backend uses MLX-LM on macOS.

```bash
bash system/download_vad.sh /path/to/models
python -m system.live_asr \
  --wav /path/to/example.wav \
  --asr-dir /path/to/models/asr \
  --refiner /path/to/models/refiner-mlx
```

Use `--identity-refiner` only for ASR and chunking diagnostics. See [system/README.md](system/README.md) for model preparation and runtime options.

## Training

### 1. Generate Refiner training data

Start an OpenAI-compatible vLLM service, or configure an OpenRouter-compatible service:

```bash
export VLLM_MODEL_NAME=/path/to/gemma-4-31b-it
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
python run_pipeline.py
```

The five-stage pipeline generates Oral/Clean pairs, simulates ASR hypotheses, performs semantic quality control, and deduplicates the final records. Outputs are written to `data/final/`. See [pipeline/README.md](pipeline/README.md).

### 2. Export SFT data

Export the finalized records into LLaMA Factory format:

```bash
python pipeline/scripts/export_sft.py \
  --inputs /path/to/AgenticASR/data/final/train.jsonl \
  --train-output /path/to/llamafactory-data/train_sft.json \
  --val-output /path/to/llamafactory-data/val_sft.json
```

### 3. Fine-tune the Refiner

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

## Evaluation

Download `rubric.json` from [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench) or [Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench), then run the judge in `experiments/scripts/main.py` with `--rubric /path/to/rubric.json`. See [experiments/README.md](experiments/README.md).

## Repository Guide

- [`pipeline/`](pipeline/README.md): synthetic training-data generation and SFT export.
- [`experiments/`](experiments/README.md): Refiner inference and AASR-Bench evaluation.
- [`system/`](system/README.md): streaming VAD, ASR, chunk management, and online refinement.

## Citation

```bibtex
@misc{jiang2026agenticasrrefiningspeechrecognition,
      title={AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach},
      author={Zixuan Jiang and Binghao Qiang and Jiaying Chi and Yanqiao Zhu and Kai Yu and Xie Chen},
      year={2026},
      eprint={2607.28175},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.28175},
}
```

## Acknowledgements

We thank the authors and contributors of [LLaMA Factory](https://github.com/hiyouga/LLaMAFactory), [MiniCPM](https://github.com/OpenBMB/MiniCPM), [X-ASR](https://github.com/Gilgamesh-J/X-ASR), [Gemma](https://github.com/google-deepmind/gemma), and [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) for their great work and open-source contributions.

## License

This project is released under the [Apache License 2.0](LICENSE).

# Streaming AgenticASR

`system/` contains the streaming implementation used by the AgenticASR desktop App. The packaged Windows/macOS application is distributed through the [VibeXASR product page](https://vibexasr.speech.wiki/); this directory is the reproducible Python implementation.

## Runtime Path

```text
WAV/microphone -> VAD -> online sherpa-onnx ASR -> ChunkManager -> K=3 Refiner
```

The full App path requires a Refiner. The current backend is MLX-LM on macOS. `--identity-refiner` is only an ASR/chunking diagnostic mode.

## Models

Install dependencies:

```bash
python -m pip install -r system/requirements.txt
python -m pip install mlx mlx-lm
```

Place an online sherpa-onnx checkpoint under `models/asr/` with `tokens.txt` and either transducer files (`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`) or a Wenet CTC model (`model*.onnx`).

Download the default Silero VAD:

```bash
bash system/download_vad.sh models
```

Other supported modes are `--vad energy` (no model) and `--vad firered` (install `fireredvad` and provide `--firered-dir`).

Convert a trained Refiner for the current Mac backend:

```bash
python pipeline/scripts/convert_to_mlx.py \
  --input /path/to/huggingface-refiner \
  --output models/refiner-mlx \
  --quantize q4_0
```

Run:

```bash
python -m system.live_asr \
  --wav path/to/example.wav \
  --asr-dir models/asr \
  --refiner models/refiner-mlx
```

## Files

- `live_asr.py`: terminal entry point and streaming loop.
- `backends.py`: audio input, VAD, and sherpa-onnx adapters.
- `chunking.py`: stable bounded chunks from incremental hypotheses.
- `refiner.py`: Refiner protocol, MLX backend, identity diagnostic backend, and sliding-window session.
- `download_vad.sh`: download the Silero VAD model.
- `requirements.txt`: streaming dependencies.
- `__init__.py`: package exports.

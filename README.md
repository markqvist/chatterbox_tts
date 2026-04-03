# Chatterbox TTS Server

Fast, OpenAI-compatible TTS server for Chatterbox models with voice cloning support for local inference. Built for use with [Ember](https://github.com/markqvist/ember) and [Humanity's Last Command](https://github.com/markqvist/lc).

Has a very useful Hybrid Mode for real-time generation on Strix Halo: T3 layers run on GPU, while the S3Gen audio decoder layers runs on CPU (until ROCm `MIOpen` bugs are fixed, at least). Fully compatible with `llama-swap`.

## Features

- **Three Model Variants**: Turbo (350M, fast), Base (500M, CFG control), Multilingual (500M, 23 languages)
- **Voice Cloning**: Zero-shot voice cloning from reference audio (5+ seconds)
- **Conditionals Caching**: Fast repeated generation with cached speaker embeddings
- **Real-time on Strix Halo**: Runs faster than real-time on Strix Halo
- **Local-First Model Loading**: Uses local weights if available, downloads only when needed
- **Extensible Preprocessing**: Text cleanup pipeline (markdown stripping, etc.) with hooks for future expansion
- **Unified Voice Configs**: Single schema works across all models (unsupported params gracefully ignored)
- **ROCm/CUDA/CPU Support**: Automatic backend detection with graceful fallback
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI TTS endpoints
- **Web UI**: Built-in voice management interface at `/ui`

## Quick Start

### 1. Install PyTorch for Your Platform

**Important:** Install PyTorch for your specific hardware BEFORE installing this package. You'll most likely need Python 3.11 specifically for all of this. Use `pyenv` and potentially a `venv` as well.

**AMD ROCm (including ROCm 7.x):**
```bash
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

**NVIDIA CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Chatterbox TTS Server

```bash
cd chatterbox_tts
pip install -e .
```

### 3. Start Server

```bash
# Default: Turbo on CUDA
python chatterbox_server.py

# Hybrid mode for AMD ROCm
CHATTERBOX_MODEL_PATH=/path/to/chatterbox_tts/models CHATTERBOX_DEVICE=rocm CHATTERBOX_PORT=8082 python chatterbox_server.py

# ROCm (AMD GPU)
CHATTERBOX_DEVICE=rocm CHATTERBOX_PORT=8082 python chatterbox_server.py

# CPU fallback
CHATTERBOX_DEVICE=cpu CHATTERBOX_PORT=8082 python chatterbox_server.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_DEVICE` | `cuda` | torch device: cuda, rocm, mps, cpu |
| `CHATTERBOX_PORT` | `8082` | Server port |
| `CHATTERBOX_HOST` | `127.0.0.1` | Bind address |
| `CHATTERBOX_MODEL_PATH` | - | Local path to model weights |
| `CHATTERBOX_VOICES_DIR` | `./voices` | Voice reference audio storage |
| `CHATTERBOX_MAX_CONCURRENT` | `4` | Max concurrent generations |
| `HF_HOME` | `~/.cache/huggingface` | Model cache directory |
| `CHATTERBOX_ROCM_HYBRID` | `0` | ROCm only: T3 on GPU, S3Gen on CPU (turbo only) |
| `CHATTERBOX_ENABLE_CHUNKING` | `true` | Enable automatic text chunking for long inputs |
| `CHATTERBOX_CHUNK_SIZE` | `375` | Maximum characters per chunk |
| `CHATTERBOX_CHUNK_INTER_DELAY_MS` | `150` | Milliseconds of silence between chunks (0 to disable) |

### ROCm Hybrid Mode (Workaround for MIOpen Bugs)

ROCm 7.x has `MIOpen` convolution bugs that cause `miopenStatusUnknownError` and extreme slowdowns in the S3Gen audio decoder. The hybrid mode works around this:

- **T3 (text encoder)**: Runs on GPU → Fast token generation
- **S3Gen (audio decoder)**: Runs on CPU → Avoids MIOpen conv1d bug

**Trade-off**: ~15-20% slower than full GPU, but 2-3x faster than full CPU. And all in all, much faster due to `MIOpen` allocation slowdowns.

```bash
# Enable hybrid mode
CHATTERBOX_ROCM_HYBRID=1 CHATTERBOX_DEVICE=rocm python chatterbox_server.py
```

# Give Something Back

Use this program? Support it.

---

- Monero:
  ```
  84FpY1QbxHcgdseePYNmhTHcrgMX4nFfBYtz2GKYToqHVVhJp8Eaw1Z1EedRnKD19b3B8NiLCGVxzKV17UMmmeEsCrPyA5w
  ```
- Bitcoin
  ```
  bc1pgqgu8h8xvj4jtafslq396v7ju7hkgymyrzyqft4llfslz5vp99psqfk3a6
  ```
- Ethereum
  ```
  0x91C421DdfB8a30a49A71d63447ddb54cEBe3465E
  ```
- Liberapay: https://liberapay.com/Reticulum/

- Ko-Fi: https://ko-fi.com/markqvist

---

If you're broke, I *may* also accept insightful and paradoxical observations, poetry, postcards or handmade drawings.

You can freely copy, share and distribute `lc` in binary or source form, as long as the `README.md` and `LICENSE.md` files in their entire form, without any modifications is distributed along with it, and displayed prominently to potential users.

## API Endpoints

### OpenAI-Compatible

- `POST /v1/audio/speech` - Generate speech
- `GET /v1/audio/voices` - List voice IDs (simple array)
- `GET /v1/models` - List available models

### Voice Management

- `GET /v1/voices` - List voices with metadata
- `POST /v1/voices` - Create voice from audio
- `GET /v1/voices/{id}` - Get voice config
- `DELETE /v1/voices/{id}` - Delete voice
- `GET /v1/voices/{id}/preview` - Play reference audio
- `POST /v1/voices/{id}/conditionals` - Pre-compute embeddings

### Admin/Debug

- `GET /health` - Health check
- `GET /v1/admin/cache` - Cache statistics
- `GET /v1/admin/model-info` - Model loading info

## llama-swap Integration

Add to your `llama-swap` configuration:

```yaml
audio_tts_chatterbox:
  ttl: 30
  cmd: python3 /path/to/chatterbox_tts/chatterbox_server.py
  proxy: http://localhost:8082
  checkEndpoint: "/health"
  concurrencyLimit: 4
  env:
    - CHATTERBOX_DEVICE=rocm
    - CHATTERBOX_ROCM_HYBRID=1
    - CHATTERBOX_PORT=8082
    - CHATTERBOX_VOICES_DIR=/path/to/voices
    - CHATTERBOX_MODEL_PATH=/path/to/models
    - CHATTERBOX_MAX_CONCURRENT=4
    - CHATTERBOX_ENABLE_CHUNKING=true
    - CHATTERBOX_CHUNK_SIZE=375
    - CHATTERBOX_CHUNK_INTER_DELAY_MS=150
```

## Model Storage

Models are stored in the HuggingFace cache by default. To use a custom location:

```bash
# Download to custom path first
export HF_HOME=/storage/1/models/huggingface
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained('cuda')"

# Then use that path
export CHATTERBOX_MODEL_PATH=/storage/1/models/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/...
```

## Voice Configuration

Each voice has a JSON config file with generation parameters:

```json
{
  "voice_id": "mark_deep",
  "model_type": "turbo",
  "reference_audio": "voices/mark_deep.wav",
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 1000,
  "repetition_penalty": 1.2,
  "norm_loudness": true,
  "cfg_weight": 0.5,
  "exaggeration": 0.5,
  "enable_preprocessing": true
}
```

Parameters are model-aware: `cfg_weight` and `exaggeration` are ignored for Turbo.

## Text Chunking for Long Inputs

For text longer than the configured chunk size (default: 375 characters), the server automatically:

1. **Splits text** on sentence boundaries (periods, exclamation marks, question marks)
2. **Generates audio** for each chunk sequentially using the same voice conditionals
3. **Inserts configurable silence** between chunks (default: 150ms) for natural prosody
4. **Concatenates** all audio into a single seamless output

This prevents quality degradation and garbled speech on long inputs while maintaining consistent voice characteristics across all chunks.

**Configuration:**
```bash
# Increase chunk size for fewer splits (may degrade quality on very long text)
CHATTERBOX_CHUNK_SIZE=500

# Adjust pause between sentences (0 = no pause, 500 = half second)
CHATTERBOX_CHUNK_INTER_DELAY_MS=200

# Disable chunking (not recommended for long text)
CHATTERBOX_ENABLE_CHUNKING=false
```

**Chunking Strategy:**
- Text is split on sentence boundaries to preserve natural prosody
- Sentences exceeding chunk size are split on word boundaries
- All chunks use the same cached voice conditionals for consistency
- Speed modification is applied once to the final concatenated audio

## Paralinguistic Tags

Chatterbox Turbo supports expressive tags in text:

- `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`
- `[groan]`, `[sniff]`, `[gasp]`, `[chuckle]`, `[laugh]`

Example: `"That's hilarious! [laugh] Wait, really?"`

## Preprocessing Pipeline

Current capabilities:
- Strip markdown code blocks → "(see code in content)"
- Strip markdown tables → "(see table in content)"
- Normalize whitespace and punctuation

Future expansion hooks available for sentiment analysis and auto tag insertion.

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌─────────────┐
│   FastAPI   │───▶│  VoiceManager  │───▶│   Voices    │
│   Server    │     │  (conditionals │     │  (audio +   │
│             │     │   caching)     │     │   config)   │
└──────┬──────┘     └───────┬────────┘     └─────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│ TTSGenerator│───▶│  Chatterbox  │
│ (model load,│     │   Models     │
│  generate)  │     │ (Turbo/Base/ │
└─────────────┘     │ Multilingual)│
                    └──────────────┘
```

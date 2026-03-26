#!/bin/bash
cd /path/to/chatterbox_tts
export CHATTERBOX_DEVICE=rocm
export CHATTERBOX_ROCM_HYBRID=1
export CHATTERBOX_HOST=0.0.0.0
export CHATTERBOX_PORT=8082
export CHATTERBOX_MODEL_PATH=/path/to/chatterbox_tts/models
export CHATTERBOX_VOICES_DIR=/path/to/chatterbox_tts/voices
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export CHATTERBOX_MAX_CONCURRENT=4

export CHATTERBOX_ENABLE_CHUNKING=true
export CHATTERBOX_CHUNK_SIZE=500
export CHATTERBOX_CHUNK_INTER_DELAY_MS=200

exec python3 chatterbox_server.py

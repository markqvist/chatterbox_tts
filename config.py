"""
Configuration management for Chatterbox TTS Server.

Environment Variables:
    CHATTERBOX_DEVICE: torch device (cuda, rocm, mps, cpu) - default: cuda
    CHATTERBOX_PORT: Server port - default: 8082
    CHATTERBOX_HOST: Bind address - default: 127.0.0.1
    CHATTERBOX_MODEL_PATH: Local path to model weights
    CHATTERBOX_MODEL_TYPE: Model variant (turbo, base, multilingual) - default: turbo
    CHATTERBOX_VOICES_DIR: Directory for reference audio - default: ./voices
    CHATTERBOX_MAX_CONCURRENT: Generation semaphore limit - default: 4
    CHATTERBOX_MAX_TEXT_CHARS: Input length limit - default: 32768
    CHATTERBOX_DEFAULT_TEMPERATURE: Default temperature - default: 0.8
    CHATTERBOX_DEFAULT_TOP_P: Default top_p - default: 0.95
    CHATTERBOX_DEFAULT_TOP_K: Default top_k - default: 1000
    CHATTERBOX_DEFAULT_REPETITION_PENALTY: Default repetition penalty - default: 1.2
    HF_HOME: HuggingFace cache directory
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ServerConfig:
    """Server runtime configuration."""
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8082
    
    # Model configuration
    device: str = "cuda"  # cuda, rocm (via cuda), mps, cpu
    model_type: Literal["turbo", "base", "multilingual"] = "turbo"
    model_path: Path | None = None  # Local path to model weights
    
    # Paths
    voices_dir: Path = field(default_factory=lambda: Path("voices"))
    
    # Performance
    max_concurrent_generations: int = 4
    max_text_chars: int = 32768
    
    # Default generation parameters
    default_temperature: float = 0.8
    default_top_p: float = 0.95
    default_top_k: int = 1000
    default_repetition_penalty: float = 1.2
    default_norm_loudness: bool = True
    
    # Base/Multilingual model only parameters (ignored for Turbo)
    default_cfg_weight: float = 0.5
    default_exaggeration: float = 0.5
    
    # Audio settings
    supported_formats: tuple[str, ...] = ("wav", "mp3", "flac", "opus")
    default_output_format: str = "wav"
    
    # Text chunking settings for long inputs
    enable_chunking: bool = True  # Enable automatic text chunking for long inputs
    chunk_size: int = 500  # Maximum characters per chunk
    chunk_inter_delay_ms: float = 150.0  # Milliseconds of silence between chunks (0 to disable)
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.voices_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> ServerConfig:
    """Load configuration from environment variables."""
    config = ServerConfig()
    
    # Server settings
    if host := os.getenv("CHATTERBOX_HOST"):
        config.host = host
    if port := os.getenv("CHATTERBOX_PORT"):
        config.port = int(port)
    
    # Model configuration
    if device := os.getenv("CHATTERBOX_DEVICE"):
        config.device = device
    if model_type := os.getenv("CHATTERBOX_MODEL_TYPE"):
        if model_type in ("turbo", "base", "multilingual"):
            config.model_type = model_type  # type: ignore
    if model_path := os.getenv("CHATTERBOX_MODEL_PATH"):
        config.model_path = Path(model_path)
    
    # Paths
    if voices_dir := os.getenv("CHATTERBOX_VOICES_DIR"):
        config.voices_dir = Path(voices_dir)
    
    # Performance
    if max_concurrent := os.getenv("CHATTERBOX_MAX_CONCURRENT"):
        config.max_concurrent_generations = int(max_concurrent)
    if max_chars := os.getenv("CHATTERBOX_MAX_TEXT_CHARS"):
        config.max_text_chars = int(max_chars)
    
    # Default generation parameters
    if temp := os.getenv("CHATTERBOX_DEFAULT_TEMPERATURE"):
        config.default_temperature = float(temp)
    if top_p := os.getenv("CHATTERBOX_DEFAULT_TOP_P"):
        config.default_top_p = float(top_p)
    if top_k := os.getenv("CHATTERBOX_DEFAULT_TOP_K"):
        config.default_top_k = int(top_k)
    if rep_penalty := os.getenv("CHATTERBOX_DEFAULT_REPETITION_PENALTY"):
        config.default_repetition_penalty = float(rep_penalty)
    
    # Text chunking configuration
    if enable_chunking := os.getenv("CHATTERBOX_ENABLE_CHUNKING"):
        config.enable_chunking = enable_chunking.lower() in ("true", "1", "yes")
    if chunk_size := os.getenv("CHATTERBOX_CHUNK_SIZE"):
        config.chunk_size = int(chunk_size)
    if chunk_delay := os.getenv("CHATTERBOX_CHUNK_INTER_DELAY_MS"):
        config.chunk_inter_delay_ms = float(chunk_delay)
    
    # Ensure voices directory exists
    config.voices_dir.mkdir(parents=True, exist_ok=True)
    
    return config


# Global config instance
CONFIG = load_config()

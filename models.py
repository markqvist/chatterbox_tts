"""
Pydantic models for API request/response schemas.
OpenAI-compatible where applicable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request schema."""
    
    model: str = Field(
        default="chatterbox-turbo",
        description="Model identifier (chatterbox-turbo, chatterbox-base, chatterbox-multilingual)"
    )
    input: str = Field(
        ...,
        description="Text to synthesize",
        max_length=65536
    )
    voice: str = Field(
        ...,
        description="Voice ID (reference audio filename without extension)"
    )
    response_format: Literal["wav", "mp3", "flac", "opus"] = Field(
        default="mp3",
        description="Audio output format"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed modifier (applied via resampling)"
    )
    
    # Extended parameters for Chatterbox-specific control
    temperature: float | None = Field(
        default=None,
        ge=0.05,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        le=1000,
        description="Top-k sampling parameter"
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=1.0,
        le=2.0,
        description="Repetition penalty"
    )
    norm_loudness: bool | None = Field(
        default=None,
        description="Normalize loudness to -27 LUFS"
    )
    
    # Base/Multilingual model parameters (ignored for Turbo)
    cfg_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Classifier-free guidance weight (base/multilingual only)"
    )
    exaggeration: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Voice exaggeration factor (base/multilingual only)"
    )
    
    # Multilingual model specific
    language_id: str | None = Field(
        default=None,
        description="Language code for multilingual model (en, fr, de, zh, etc.). Uses voice config default if not specified."
    )


class VoiceConfig(BaseModel):
    """Voice configuration schema for per-voice settings."""
    
    voice_id: str = Field(..., description="Unique voice identifier")
    model_type: Literal["turbo", "base", "multilingual"] = Field(
        default="turbo",
        description="Model variant to use for this voice"
    )
    reference_audio: str = Field(..., description="Path to reference audio file")
    
    # Multilingual model settings
    language_id: str = Field(
        default="en",
        description="Language code for multilingual model (en, fr, de, zh, etc.)"
    )
    
    # Generation parameters
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=1000, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    norm_loudness: bool = Field(default=True)
    
    # Base/Multilingual only (ignored for Turbo)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=2.0)
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    
    # Preprocessing
    enable_preprocessing: bool = Field(
        default=True,
        description="Enable text preprocessing (cleanup, etc.)"
    )


class VoiceInfo(BaseModel):
    """Voice metadata for API responses."""
    
    voice_id: str
    model_type: Literal["turbo", "base", "multilingual"]
    reference_audio: str
    duration_seconds: float | None = None
    sample_rate: int | None = None


class VoicesListResponse(BaseModel):
    """Response schema for listing voices."""
    
    voices: list[VoiceInfo]


class ModelInfoResponse(BaseModel):
    """OpenAI-compatible model information."""
    
    id: str
    object: str = "model"
    created: int
    owned_by: str = "resemble-ai"


class ModelsListResponse(BaseModel):
    """OpenAI-compatible models list response."""
    
    object: str = "list"
    data: list[ModelInfoResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_type: str
    device: str
    version: str = "0.1.0"


class CreateVoiceRequest(BaseModel):
    """Request schema for creating a voice configuration."""
    
    voice_id: str = Field(..., description="Unique voice identifier")
    model_type: Literal["turbo", "base", "multilingual"] = Field(default="turbo")
    language_id: str = Field(default="en", description="Language code for multilingual model")
    temperature: float = Field(default=0.8)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=1000)
    repetition_penalty: float = Field(default=1.2)
    norm_loudness: bool = Field(default=True)
    cfg_weight: float = Field(default=0.5)
    exaggeration: float = Field(default=0.5)
    enable_preprocessing: bool = Field(default=True)


class CreateVoiceResponse(BaseModel):
    """Response schema for voice creation."""
    
    status: str
    voice: VoiceInfo
    conditionals_cached: bool


class DeleteVoiceResponse(BaseModel):
    """Response schema for voice deletion."""
    
    status: str
    message: str

"""
Chatterbox TTS Microservice - FastAPI Application

OpenAI-compatible /v1/audio/speech endpoint with advanced features:
- Voice cloning from audio files
- Voice profile management with per-voice configuration
- Support for Turbo, Base, and Multilingual models
- Local-first model loading with auto-download fallback
- Extensible text preprocessing pipeline
- ROCm/CUDA/CPU backend support
- Voice conditionals caching for fast repeated generation
- Web UI for voice management

Environment Variables:
    CHATTERBOX_DEVICE: torch device (cuda, rocm, mps, cpu) - default: cuda
    CHATTERBOX_PORT: Server port - default: 8082
    CHATTERBOX_HOST: Bind address - default: 127.0.0.1
    CHATTERBOX_MODEL_PATH: Local path to model weights
    CHATTERBOX_MODEL_TYPE: Default model variant (turbo, base, multilingual)
    CHATTERBOX_VOICES_DIR: Directory for reference audio
    CHATTERBOX_MAX_CONCURRENT: Generation semaphore limit
    CHATTERBOX_MAX_TEXT_CHARS: Input length limit
    HF_HOME: HuggingFace cache directory
"""

from __future__ import annotations

import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import CONFIG
from models import (
    SpeechRequest,
    CreateVoiceRequest,
    CreateVoiceResponse,
    DeleteVoiceResponse,
    VoicesListResponse,
    VoiceInfo,
    VoiceConfig,
    ModelsListResponse,
    ModelInfoResponse,
    HealthResponse,
)
from voice_manager import VOICE_MANAGER
from generator import get_generator, TTSGenerator, GENERATOR

# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with graceful startup/shutdown."""
    # Startup
    logger.info("Starting Chatterbox TTS Server...")
    logger.info(f"Configuration: device={CONFIG.device}, model_type={CONFIG.model_type}")
    logger.info(f"Voices directory: {CONFIG.voices_dir}")
    
    try:
        # Initialize generator (loads default model)
        generator = await get_generator()
        logger.success(f"Server ready at http://{CONFIG.host}:{CONFIG.port}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        # Don't raise - let server start anyway, model will load on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chatterbox TTS Server...")


app = FastAPI(
    title="Chatterbox TTS Server",
    description="OpenAI-compatible TTS microservice with voice cloning support",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_raw_requests(request: Request, call_next):
    if "/v1/audio/speech" not in request.url.path: return await call_next(request)
    
    try:
        body = await request.body()
        logger.info(f"Raw request body: {body.decode('utf-8')[:5000]}")
        
        # Also capture headers and method
        logger.info(f"Headers: {dict(request.headers)}")
    except Exception as e:
        logger.exception(f"Failed to read raw body: {e}")
    
    return await call_next(request)

# Serve static files for web UI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# Web UI
# ============================================================================

@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """Web UI for voice management."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>UI not found</h1><p>Ensure static/index.html exists</p>"


@app.get("/")
async def root():
    """Redirect root to UI."""
    return {"message": "Chatterbox TTS Server", "ui": "/ui", "api": "/docs"}


# ============================================================================
# Health & Model Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    generator = await get_generator()
    loaded_models = list(generator._models.keys()) if generator else []
    
    return HealthResponse(
        status="healthy",
        model_loaded=len(loaded_models) > 0,
        model_type=CONFIG.model_type,
        device=generator.device if generator else CONFIG.device,
    )


@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """List available TTS models (OpenAI-compatible)."""
    return ModelsListResponse(
        object="list",
        data=[
            ModelInfoResponse(
                id="chatterbox-turbo",
                created=int(time.time()),
                owned_by="resemble-ai",
            ),
            ModelInfoResponse(
                id="chatterbox-base",
                created=int(time.time()),
                owned_by="resemble-ai",
            ),
            ModelInfoResponse(
                id="chatterbox-multilingual",
                created=int(time.time()),
                owned_by="resemble-ai",
            ),
        ]
    )


# ============================================================================
# Speech Generation (OpenAI-Compatible)
# ============================================================================

@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    """
    Convert text to speech (OpenAI-compatible endpoint).
    
    Supports extended Chatterbox parameters via the request body:
    - temperature: Sampling temperature (0.05-2.0)
    - top_p: Nucleus sampling (0.0-1.0)
    - top_k: Top-k sampling (0-1000)
    - repetition_penalty: Penalty for repetition (1.0-2.0)
    - norm_loudness: Normalize to -27 LUFS (bool)
    - cfg_weight: Classifier-free guidance (base/multilingual only)
    - exaggeration: Voice exaggeration (base/multilingual only)
    """
    # Validate input
    if not request.input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty"
        )
    
    # Validate voice exists
    if not VOICE_MANAGER.voice_exists(request.voice):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice not found: {request.voice}"
        )
    
    try:
        # Load voice config
        voice_config = VOICE_MANAGER.load_voice_config(request.voice)
        
        # Override model type if specified in request
        model_type = request.model.replace("chatterbox-", "")
        if model_type in ("turbo", "base", "multilingual"):
            voice_config.model_type = model_type  # type: ignore
        
        # Get generator
        generator = await get_generator()
        
        # Build override parameters from request
        override_params = {}
        if request.temperature is not None:
            override_params["temperature"] = request.temperature
        if request.top_p is not None:
            override_params["top_p"] = request.top_p
        if request.top_k is not None:
            override_params["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            override_params["repetition_penalty"] = request.repetition_penalty
        if request.norm_loudness is not None:
            override_params["norm_loudness"] = request.norm_loudness
        if request.cfg_weight is not None:
            override_params["cfg_weight"] = request.cfg_weight
        if request.exaggeration is not None:
            override_params["exaggeration"] = request.exaggeration
        if request.language_id is not None:
            override_params["language_id"] = request.language_id

        # Generate audio
        audio_bytes, mime_type = await generator.generate(
            text=request.input,
            voice_config=voice_config,
            output_format=request.response_format,
            speed=request.speed,
            **override_params
        )
        
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            }
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Speech generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}"
        )


@app.get("/v1/audio/voices")
async def list_voices_simple():
    """
    List available voices as simple array (most compatible format).
    
    This matches the format expected by many TTS clients.
    """
    voices = VOICE_MANAGER.list_voices()
    return [voice.voice_id for voice in voices]


# ============================================================================
# Voice Management
# ============================================================================

@app.get("/v1/voices", response_model=VoicesListResponse)
async def list_voices():
    """List all available voices with full metadata."""
    voices = VOICE_MANAGER.list_voices()
    return VoicesListResponse(voices=voices)


@app.post("/v1/voices", response_model=CreateVoiceResponse)
async def create_voice(
    voice_id: str = Form(..., description="Unique voice identifier"),
    model_type: str = Form(default="turbo", description="Model variant (turbo, base, multilingual)"),
    language_id: str = Form(default="en", description="Language code for multilingual model (en, fr, de, etc.)"),
    temperature: float = Form(default=0.8, description="Default temperature"),
    top_p: float = Form(default=0.95, description="Default top_p"),
    top_k: int = Form(default=1000, description="Default top_k"),
    repetition_penalty: float = Form(default=1.2, description="Default repetition penalty"),
    norm_loudness: bool = Form(default=True, description="Normalize loudness"),
    cfg_weight: float = Form(default=0.5, description="CFG weight (base/multilingual only)"),
    exaggeration: float = Form(default=0.5, description="Exaggeration (base/multilingual only)"),
    enable_preprocessing: bool = Form(default=True, description="Enable text preprocessing"),
    audio: UploadFile = File(..., description="Reference audio file (wav, mp3, flac, ogg)"),
):
    """
    Create a new voice from reference audio.
    
    The audio file should contain clear speech (5+ seconds recommended).
    The system will store the audio and create a voice configuration.
    """
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = Path(audio.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio format. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Sanitize voice ID
    import re
    safe_voice_id = re.sub(r'[^a-zA-Z0-9_-]', '_', voice_id)
    safe_voice_id = safe_voice_id.strip('_')
    if not safe_voice_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice ID"
        )
    
    # Validate model type
    if model_type not in ("turbo", "base", "multilingual"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model type must be: turbo, base, or multilingual"
        )
    
    # Validate language_id for multilingual
    if model_type == "multilingual":
        supported_langs = {
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh"
        }
        if language_id.lower() not in supported_langs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language_id '{language_id}'. Supported: {', '.join(sorted(supported_langs))}"
            )
    
    # Create voice configuration
    config = VoiceConfig(
        voice_id=safe_voice_id,
        model_type=model_type,  # type: ignore
        language_id=language_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        enable_preprocessing=enable_preprocessing,
        reference_audio="",  # Will be set by voice_manager
    )
    
    try:
        # Read uploaded audio
        audio_bytes = await audio.read()
        
        # Create voice
        metadata = VOICE_MANAGER.create_voice(
            voice_id=safe_voice_id,
            audio_source=audio_bytes,
            config=config,
        )
        
        return CreateVoiceResponse(
            status="success",
            voice=VoiceInfo(
                voice_id=metadata.voice_id,
                model_type=metadata.model_type,  # type: ignore
                reference_audio=str(metadata.audio_path),
                duration_seconds=metadata.duration_seconds,
                sample_rate=metadata.sample_rate,
            ),
            conditionals_cached=False,  # Will be computed on first use
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Voice creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create voice: {str(e)}"
        )


@app.get("/v1/voices/{voice_id}")
async def get_voice(voice_id: str):
    """Get voice configuration and metadata."""
    if not VOICE_MANAGER.voice_exists(voice_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice not found: {voice_id}"
        )
    
    try:
        config = VOICE_MANAGER.load_voice_config(voice_id)
        audio_path = VOICE_MANAGER.get_voice_audio_path(voice_id)
        
        # Get audio metadata
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio) / sr
        
        return {
            "voice_id": voice_id,
            "config": config.model_dump(),
            "audio": {
                "path": str(audio_path),
                "duration_seconds": float(duration),
                "sample_rate": int(sr),
            },
            "conditionals_cached": VOICE_MANAGER.is_conditionals_cached(voice_id),
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load voice: {str(e)}"
        )


@app.delete("/v1/voices/{voice_id}", response_model=DeleteVoiceResponse)
async def delete_voice(voice_id: str):
    """Delete a voice and all associated files."""
    if not VOICE_MANAGER.voice_exists(voice_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice not found: {voice_id}"
        )
    
    try:
        VOICE_MANAGER.delete_voice(voice_id)
        return DeleteVoiceResponse(
            status="success",
            message=f"Voice '{voice_id}' deleted"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {str(e)}"
        )


@app.get("/v1/voices/{voice_id}/preview")
async def preview_voice(voice_id: str):
    """
    Return the voice's reference audio for preview/verification.
    """
    if not VOICE_MANAGER.voice_exists(voice_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice not found: {voice_id}"
        )
    
    try:
        audio_path = VOICE_MANAGER.get_voice_audio_path(voice_id)
        
        # Read and return audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Determine MIME type from extension
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        mime_type = mime_types.get(audio_path.suffix.lower(), "audio/wav")
        
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{voice_id}_preview{audio_path.suffix}"'
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load voice preview: {str(e)}"
        )


@app.post("/v1/voices/{voice_id}/conditionals")
async def cache_voice_conditionals(voice_id: str):
    """
    Pre-compute and cache conditionals for a voice.
    
    This speeds up subsequent generation by computing speaker embeddings
    ahead of time.
    """
    if not VOICE_MANAGER.voice_exists(voice_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice not found: {voice_id}"
        )
    
    try:
        # Load voice config
        voice_config = VOICE_MANAGER.load_voice_config(voice_id)
        
        # Get generator and load model
        generator = await get_generator()
        model = await generator._load_model(voice_config.model_type)
        
        # Compute conditionals (this will cache them)
        await generator._prepare_conditionals(model, voice_config)
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "conditionals_cached": True,
        }
    
    except Exception as e:
        logger.exception("Failed to cache conditionals")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cache conditionals: {str(e)}"
        )


@app.delete("/v1/voices/{voice_id}/conditionals")
async def clear_voice_conditionals(voice_id: str):
    """Clear cached conditionals for a voice."""
    VOICE_MANAGER.clear_conditionals_cache(voice_id)
    return {
        "status": "success",
        "voice_id": voice_id,
        "message": "Conditionals cache cleared"
    }


# ============================================================================
# Admin/Debug Endpoints
# ============================================================================

@app.get("/v1/admin/cache")
async def get_cache_stats():
    """Get conditionals cache statistics."""
    return VOICE_MANAGER.get_cache_stats()


@app.delete("/v1/admin/cache")
async def clear_all_caches():
    """Clear all conditionals caches."""
    VOICE_MANAGER.clear_conditionals_cache()
    return {"status": "success", "message": "All caches cleared"}


@app.get("/v1/admin/model-info")
async def get_model_info():
    """Get information about loaded models."""
    generator = await get_generator()
    return await generator.get_model_info()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "chatterbox_server:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=1,  # Single worker for shared model state
        log_level="info",
    )

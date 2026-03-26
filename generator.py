"""
TTS generation logic for Chatterbox models.

Handles:
- Model loading (local-first, auto-download fallback)
- Unified interface for Turbo, Base, and Multilingual models
- Voice conditioning and caching
- Async generation with semaphore control
- Audio encoding to various formats
"""

from __future__ import annotations

import io
import os
import re
import asyncio
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from loguru import logger

from config import CONFIG, ServerConfig
from models import VoiceConfig
from voice_manager import VOICE_MANAGER
from preprocessor import preprocess_text


# Model repository IDs on HuggingFace
MODEL_REPOS = {
    "turbo": "ResembleAI/chatterbox-turbo",
    "base": "ResembleAI/chatterbox",
    "multilingual": "ResembleAI/chatterbox",
}


@dataclass
class GenerationResult:
    """Result of TTS generation."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float


class TTSGenerator:
    """
    Unified TTS generator for all Chatterbox model variants.
    
    Supports:
    - Chatterbox Turbo (350M, English, fastest)
    - Chatterbox Base (500M, English, CFG/exaggeration control)
    - Chatterbox Multilingual (500M, 23+ languages)
    """
    
    def __init__(self, config: ServerConfig | None = None):
        self.config = config or CONFIG
        self.device = self._resolve_device()
        
        # Model instances (loaded on demand)
        self._models: dict[str, Any] = {}
        self._model_lock = asyncio.Lock()
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        
        logger.info(f"TTSGenerator initialized (device: {self.device})")
    
    def _resolve_device(self) -> str:
        """Resolve device string to valid torch device."""
        requested = self.config.device.lower()
        
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        
        elif requested == "rocm":
            # ROCm uses CUDA API on Linux
            if torch.cuda.is_available():
                logger.info("Using ROCm (via CUDA API)")
                return "cuda"
            else:
                logger.warning("ROCm requested but not available, falling back to CPU")
                return "cpu"
        
        elif requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
        
        elif requested == "cpu":
            return "cpu"
        
        else:
            logger.warning(f"Unknown device '{requested}', defaulting to CPU")
            return "cpu"
    
    async def _load_model(self, model_type: Literal["turbo", "base", "multilingual"]) -> Any:
        """
        Load a Chatterbox model (with caching).
        
        Local-first loading:
        1. Check CHATTERBOX_MODEL_PATH for local weights
        2. Check HF cache directory
        3. Download from HuggingFace (and save to custom path if set)
        
        ROCm Workaround: If CHATTERBOX_ROCM_HYBRID=1, load T3 on GPU and S3Gen on CPU
        to avoid MIOpen convolution bugs in the decoder while keeping fast text encoding.
        """
        async with self._model_lock:
            if model_type in self._models:
                return self._models[model_type]
            
            logger.info(f"Loading Chatterbox {model_type} model...")
            
            # Import here to avoid loading on startup if not needed
            if model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                model_class = ChatterboxTurboTTS
            elif model_type == "base":
                from chatterbox.tts import ChatterboxTTS
                model_class = ChatterboxTTS
            else:  # multilingual
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                model_class = ChatterboxMultilingualTTS
            
            # Try local loading first
            model = await self._try_load_local(model_type, model_class)
            
            if model is None:
                # Fall back to downloading
                model = await self._download_model(model_type, model_class)
            
            # ROCm Hybrid Mode: Keep T3 on GPU, move S3Gen to CPU
            # This avoids MIOpen conv1d bugs in the decoder while keeping fast text encoding
            if os.getenv("CHATTERBOX_ROCM_HYBRID") == "1" and self.device == "cuda":
                logger.info("ROCm Hybrid Mode: T3 on GPU, S3Gen on CPU")
                if hasattr(model, 's3gen') and hasattr(model, 't3'):
                    import torch
                    # Move S3Gen and all its submodules to CPU (avoids MIOpen conv bug)
                    model.s3gen = model.s3gen.cpu()
                    # Also move voice encoder components that feed into s3gen
                    if hasattr(model, 've'):
                        model.ve = model.ve.cpu()
                    # Keep T3 on GPU (fast text encoding)
                    model.t3 = model.t3.cuda()
                    # Mark model as hybrid for generation logic
                    model._rocm_hybrid = True
                    model._device_cpu = torch.device("cpu")
                    model._device_gpu = torch.device("cuda")
                    # Clear any cached conditionals - they need to be recomputed on CPU
                    VOICE_MANAGER.clear_conditionals_cache()
                    logger.success("ROCm Hybrid mode enabled: T3→GPU, S3Gen→CPU (conditionals cleared for CPU recomputation)")
            
            self._models[model_type] = model
            logger.success(f"Chatterbox {model_type} model loaded successfully")
            
            return model
    
    def _get_model_sentinel_files(self, model_type: str) -> list[str]:
        """Get list of sentinel files that indicate a complete model download."""
        sentinels = {
            "turbo": ["ve.safetensors", "t3_turbo_v1.safetensors", "conds.pt"],
            "base": ["ve.pt", "s3gen.pt", "t3_cfg.safetensors"],
            "multilingual": ["ve.pt", "s3gen.pt", "t3_mtl23ls_v2.safetensors"],
        }
        return sentinels.get(model_type, ["ve.pt", "ve.safetensors"])
    
    async def _try_load_local(
        self,
        model_type: str,
        model_class: type
    ) -> Any | None:
        print("Trying to load local model...")
        """Try to load model from local path."""
        # Check explicit model path
        if self.config.model_path:
            print("Has local path config")
            model_path = self.config.model_path / model_type
            model_path.mkdir(parents=True, exist_ok=True)

            if model_path.exists():
                print("Local model path exists, checking files...")
                # Verify it contains actual model files (not just empty directory)
                # Different models have different file naming conventions
                sentinel_files = self._get_model_sentinel_files(model_type)
                has_model_files = all(
                    (model_path / f).exists() 
                    for f in sentinel_files
                )
                
                if not has_model_files:
                    logger.debug(f"Model path exists but missing model files for {model_type}: {model_path}")
                    return None
                
                try:
                    logger.info(f"Loading model from {model_path}")
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(
                        None,
                        lambda: model_class.from_local(
                            str(model_path),
                            self.device
                        )
                    )
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load from custom path: {e}")
        
        # Check HF cache for this specific model
        try:
            from huggingface_hub import snapshot_download
            repo_id = MODEL_REPOS[model_type]
            
            # Get cache directory without downloading
            local_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_files_only=True,  # Don't download, only check cache
                )
            )
            
            if local_path and Path(local_path).exists():
                logger.info(f"Loading model from HF cache: {local_path}")
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(
                    None,
                    lambda: model_class.from_local(local_path, self.device)
                )
                return model
                
        except Exception as e:
            logger.debug(f"Model not found in cache: {e}")
        
        return None
    
    async def _download_model(self, model_type: str, model_class: type) -> Any:
        """Download model from HuggingFace to local path."""
        repo_id = MODEL_REPOS[model_type]
        logger.info(f"Downloading {model_type} model from HuggingFace ({repo_id})...")
        
        try:
            from huggingface_hub import snapshot_download
            
            loop = asyncio.get_event_loop()
            
            # Determine target path: custom path if set, else HF cache
            if self.config.model_path:
                target_path = self.config.model_path / model_type
                logger.info(f"Downloading model files to {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path = None  # Use HF cache default
            
            # Download files to target path (or HF cache if no custom path)
            local_path = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=str(target_path) if target_path else None,
                    local_files_only=False,
                )
            )
            
            logger.success(f"Model files downloaded to {local_path}")
            
            # Load from the local path we just downloaded to
            logger.info(f"Loading model from {local_path}")
            model = await loop.run_in_executor(
                None,
                lambda: model_class.from_local(local_path, self.device)
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not load or download {model_type} model: {e}")
    
    def _prepare_conditionals_sync(
        self,
        model: Any,
        voice_config: VoiceConfig,
    ) -> Any:
        """Synchronous version of prepare_conditionals for both normal and hybrid modes."""
        import torch
        
        voice_id = voice_config.voice_id
        
        # Check cache first
        cached = VOICE_MANAGER.get_cached_conditionals(voice_id)
        if cached is not None:
            logger.debug(f"Using cached conditionals for voice '{voice_id}'")
            model.conds = cached
            return cached
        
        # Get reference audio path
        audio_path = Path(voice_config.reference_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        # Compute conditionals
        logger.info(f"Computing conditionals for voice '{voice_id}'...")
        
        # ROCm Hybrid Mode: Compute conditionals on CPU to match S3Gen device
        is_hybrid = getattr(model, '_rocm_hybrid', False)
        target_device = torch.device("cpu") if is_hybrid else model.t3.device
        
        # Get model-specific parameters
        if hasattr(model, 'prepare_conditionals'):
            exaggeration = voice_config.exaggeration if voice_config.model_type != "turbo" else 0.0
            
            # Load and prepare audio
            import librosa
            from chatterbox.tts_turbo import S3_SR, S3GEN_SR
            
            # Load audio
            s3gen_ref_wav, _sr = librosa.load(str(audio_path), sr=S3GEN_SR)
            
            if voice_config.norm_loudness:
                s3gen_ref_wav = self._norm_loudness_sync(s3gen_ref_wav, _sr)
            
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
            
            # Prepare for s3gen (on CPU for hybrid mode)
            s3gen_ref_wav = s3gen_ref_wav[:10 * S3GEN_SR]  # DEC_COND_LEN
            
            if is_hybrid:
                # CPU path for hybrid mode
                s3gen_ref_wav_tensor = torch.from_numpy(s3gen_ref_wav).float().cpu()
                s3gen_cpu = model.s3gen.cpu()
                s3gen_ref_dict = s3gen_cpu.embed_ref(s3gen_ref_wav_tensor.numpy(), S3GEN_SR, device="cpu")
                
                # Ensure all values are CPU tensors (embed_ref may return tensors or numpy)
                def ensure_tensor_cpu(v):
                    if torch.is_tensor(v):
                        return v.cpu()
                    elif hasattr(v, 'dtype') and hasattr(v, 'shape'):  # numpy array
                        return torch.from_numpy(v).cpu()
                    return v
                
                s3gen_ref_dict = {k: ensure_tensor_cpu(v) for k, v in s3gen_ref_dict.items()}
                
                # T3 conditionals on CPU (will be moved to GPU during generation)
                ve = model.ve.cpu()
                ve_embed_raw = ve.embeds_from_wavs([ref_16k_wav[:15*S3_SR]], sample_rate=S3_SR)
                if not torch.is_tensor(ve_embed_raw):
                    ve_embed_raw = torch.from_numpy(ve_embed_raw)
                ve_embed = ve_embed_raw.mean(axis=0, keepdim=True).cpu()
                
                from chatterbox.models.t3.modules.cond_enc import T3Cond
                t3_cond = T3Cond(
                    speaker_emb=ve_embed,
                    cond_prompt_speech_tokens=None,  # Will be computed on GPU
                    emotion_adv=torch.tensor([exaggeration]).view(1, 1, 1),
                )
                
                # Re-encode with s3 tokenizer for T3
                plen = model.t3.hp.speech_cond_prompt_len
                if plen:
                    s3_tokzr = s3gen_cpu.tokenizer
                    t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:15*S3_SR]], max_len=plen)
                    if not torch.is_tensor(t3_cond_prompt_tokens):
                        t3_cond_prompt_tokens = torch.from_numpy(t3_cond_prompt_tokens)
                    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).cpu()
                    t3_cond.cond_prompt_speech_tokens = t3_cond_prompt_tokens
                
                from chatterbox.tts_turbo import Conditionals
                model.conds = Conditionals(t3_cond, s3gen_ref_dict)
                
            else:
                # Normal GPU path
                model.prepare_conditionals(str(audio_path))

        else:
            # Multilingual might have different API
            logger.warning("Using fallback conditionals preparation for multilingual model")
            model.prepare_conditionals(str(audio_path))
        
        # Cache the conditionals
        conditionals = model.conds
        VOICE_MANAGER.cache_conditionals(voice_id, conditionals)
        
        return conditionals
    
    def _norm_loudness_sync(self, wav, sr, target_lufs=-27):
        """Synchronous loudness normalization."""
        try:
            import pyloudnorm as ln
            import math
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            logger.warning(f"Error in norm_loudness, skipping: {e}")
        return wav

    async def _prepare_conditionals(
        self,
        model: Any,
        voice_config: VoiceConfig,
    ) -> Any:
        """
        Prepare or retrieve cached conditionals for a voice.
        
        Conditionals include speaker embeddings computed from reference audio.
        Uses ROCm hybrid-aware logic when in hybrid mode.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._prepare_conditionals_sync(model, voice_config)
        )
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> list[str]:
        """
        Split text into sentence-aware chunks for TTS processing.
        
        Respects max_chunk_size while merging small sentences and splitting
        oversized ones on word boundaries.
        
        Args:
            text: Input text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Normalize newlines to spaces for sentence detection
        normalized_text = text.replace('\n', ' ').strip()
        
        if len(normalized_text) <= max_chunk_size:
            return [normalized_text] if normalized_text else []
        
        # Split on sentence boundaries (periods, exclamation, question marks)
        sentence_regex = r'[^.!?]+(?:[.!?]+|\n|$)'
        raw_sentences = re.findall(sentence_regex, normalized_text)
        
        # Clean up sentences
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If single sentence exceeds max, split on word boundaries
            if len(sentence) > max_chunk_size:
                # Flush current chunk if any
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split oversized sentence on word boundaries
                words = sentence.split()
                oversized_chunk = ""
                
                for word in words:
                    test_chunk = f"{oversized_chunk} {word}".strip()
                    if len(test_chunk) > max_chunk_size:
                        if oversized_chunk:
                            chunks.append(oversized_chunk.strip())
                        oversized_chunk = word
                    else:
                        oversized_chunk = test_chunk
                
                if oversized_chunk:
                    current_chunk = oversized_chunk
                continue
            
            # Try to add sentence to current chunk
            potential_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            
            if len(potential_chunk) <= max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk: chunks.append(current_chunk.strip())

        if len(chunks) > 1: print(f"Split input into {len(chunks)} chunks")
        return [c for c in chunks if c]
    
    def _insert_inter_chunk_silence(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int,
        delay_ms: float
    ) -> np.ndarray:
        """
        Insert silence between audio chunks.
        
        Args:
            audio_chunks: List of audio arrays
            sample_rate: Sample rate of audio
            delay_ms: Milliseconds of silence to insert between chunks
            
        Returns:
            Concatenated audio with silence between chunks
        """
        if delay_ms <= 0 or len(audio_chunks) <= 1:
            return np.concatenate(audio_chunks)
        
        # Calculate zero samples for the delay
        delay_samples = int(sample_rate * (delay_ms / 1000.0))
        silence = np.zeros(delay_samples, dtype=audio_chunks[0].dtype)
        
        # Interleave chunks with silence
        interleaved = []
        for i, chunk in enumerate(audio_chunks):
            interleaved.append(chunk)
            if i < len(audio_chunks) - 1:  # Don't add silence after last chunk
                interleaved.append(silence)
        
        return np.concatenate(interleaved)
    
    async def _generate_audio_array(
        self,
        text: str,
        voice_config: VoiceConfig,
        gen_params: dict,
        loop: asyncio.AbstractEventLoop
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio array for a single text chunk.
        
        Args:
            text: Text to synthesize (single chunk)
            voice_config: Voice configuration
            gen_params: Generation parameters
            loop: Event loop for executor
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Check if we need CPU fallback BEFORE loading model
        needs_cpu_fallback = (
            os.getenv("CHATTERBOX_ROCM_HYBRID") == "1" and
            self.device == "cuda" and
            voice_config.model_type in ("base", "multilingual")
        )
        
        if needs_cpu_fallback:
            # Base/Multilingual + ROCm hybrid: use CPU-only fallback
            logger.debug(f"Routing {voice_config.model_type} directly to CPU fallback (ROCm hybrid)")
            wav, sample_rate = await self._generate_with_cpu_fallback(
                text, voice_config, gen_params, loop
            )
        else:
            # Load model (normal GPU or hybrid for Turbo)
            model = await self._load_model(voice_config.model_type)
            
            # Prepare conditionals
            await self._prepare_conditionals(model, voice_config)
            
            # Generate audio
            logger.debug(f"Generating audio for text chunk ({len(text)} chars)")
            
            # ROCm Hybrid Mode for Turbo: use hybrid GPU→CPU generation
            if getattr(model, '_rocm_hybrid', False):
                wav = await loop.run_in_executor(
                    None,
                    lambda: self._generate_hybrid(model, text, gen_params)
                )
            elif voice_config.model_type == "turbo":
                wav = await loop.run_in_executor(
                    None,
                    lambda: model.generate(text, **gen_params)
                )
            elif voice_config.model_type == "multilingual":
                wav = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        text,
                        language_id=gen_params.get("language_id", "en"),
                        audio_prompt_path=gen_params.get("audio_prompt_path"),
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        temperature=gen_params.get("temperature", 0.8),
                        repetition_penalty=gen_params.get("repetition_penalty", 2.0),
                        top_p=gen_params.get("top_p", 1.0),
                    )
                )
            else:
                # Base model
                wav = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        text,
                        audio_prompt_path=gen_params.get("audio_prompt_path"),
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        temperature=gen_params.get("temperature", 0.8),
                        repetition_penalty=gen_params.get("repetition_penalty", 2.0),
                        top_p=gen_params.get("top_p", 1.0),
                    )
                )
            
            # Get sample rate from model (unless CPU fallback which sets it)
            sample_rate = model.sr
        
        # Convert to numpy array
        if torch.is_tensor(wav):
            audio_data = wav.squeeze().detach().cpu().numpy()
        else:
            audio_data = np.array(wav).squeeze()
        
        return audio_data, sample_rate
    
    async def generate(
        self,
        text: str,
        voice_config: VoiceConfig,
        output_format: str = "wav",
        speed: float = 1.0,
        **override_params
    ) -> tuple[bytes, str]:
        """
        Generate speech from text with optional chunking for long inputs.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            output_format: Output audio format (wav, mp3, flac, opus)
            speed: Playback speed modifier
            **override_params: Override any generation parameters
            
        Returns:
            Tuple of (audio_bytes, mime_type)
        """
        async with self._semaphore:
            # Preprocess text
            if voice_config.enable_preprocessing:
                text = preprocess_text(text)
            
            # Build generation parameters
            gen_params = self._build_generation_params(voice_config, override_params)
            
            # Check if chunking is needed
            chunk_size = self.config.chunk_size
            enable_chunking = self.config.enable_chunking
            
            # Determine if we need to chunk
            needs_chunking = enable_chunking and len(text) > chunk_size
            
            loop = asyncio.get_event_loop()
            
            if not needs_chunking:
                # Single generation for short text
                logger.debug(f"Generating audio for text ({len(text)} chars, no chunking)")
                audio_data, sample_rate = await self._generate_audio_array(
                    text, voice_config, gen_params, loop
                )
            else:
                # Chunked generation for long text
                chunks = self._split_text_into_chunks(text, chunk_size)
                logger.info(f"Text length {len(text)} exceeds chunk size {chunk_size}, "
                           f"splitting into {len(chunks)} chunks")
                
                audio_chunks = []
                sample_rate = None
                
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                    try:
                        chunk_audio, chunk_sr = await self._generate_audio_array(
                            chunk, voice_config, gen_params, loop
                        )
                        audio_chunks.append(chunk_audio)
                        
                        # Capture sample rate from first chunk
                        if sample_rate is None:
                            sample_rate = chunk_sr
                    except Exception as e:
                        logger.error(f"Failed to generate chunk {i+1}: {e}")
                        # Continue with successfully generated chunks
                        break
                
                if not audio_chunks:
                    raise RuntimeError("All chunks failed to generate")
                
                # Insert silence between chunks if configured
                inter_delay_ms = self.config.chunk_inter_delay_ms
                if inter_delay_ms > 0 and len(audio_chunks) > 1:
                    logger.debug(f"Inserting {inter_delay_ms}ms silence between chunks")
                    audio_data = self._insert_inter_chunk_silence(
                        audio_chunks, sample_rate, inter_delay_ms
                    )
                else:
                    audio_data = np.concatenate(audio_chunks)
            
            # Apply speed modification if needed
            if speed != 1.0:
                audio_data = self._resample_audio(audio_data, speed)
            
            # Encode to requested format
            audio_bytes = self._encode_audio(audio_data, sample_rate, output_format)
            mime_type = self._get_mime_type(output_format)
            
            return audio_bytes, mime_type
    
    def _build_generation_params(
        self,
        voice_config: VoiceConfig,
        overrides: dict
    ) -> dict:
        """Build generation parameter dict from config and overrides."""
        params = {
            "temperature": overrides.get("temperature", voice_config.temperature),
            "top_p": overrides.get("top_p", voice_config.top_p),
            "top_k": overrides.get("top_k", voice_config.top_k),
            "repetition_penalty": overrides.get("repetition_penalty", voice_config.repetition_penalty),
            "norm_loudness": overrides.get("norm_loudness", voice_config.norm_loudness),
        }
        
        # Add model-specific parameters
        if voice_config.model_type in ("base", "multilingual"):
            params["cfg_weight"] = overrides.get("cfg_weight", voice_config.cfg_weight)
            params["exaggeration"] = overrides.get("exaggeration", voice_config.exaggeration)
        
        # Multilingual specific: language_id is REQUIRED
        if voice_config.model_type == "multilingual":
            params["language_id"] = overrides.get("language_id", voice_config.language_id)
        
        # Reference audio is handled via conditionals caching
        # But we need to include it for the first call
        if not VOICE_MANAGER.is_conditionals_cached(voice_config.voice_id):
            params["audio_prompt_path"] = voice_config.reference_audio
        
        return params
    
    def _generate_hybrid(self, model: Any, text: str, gen_params: dict) -> Any:
        """
        Generate audio in ROCm Hybrid mode.
        
        T3 runs on GPU (fast), S3Gen runs on CPU (avoids MIOpen bug).
        Handles tensor device movement automatically.
        """
        import torch
        
        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        
        # Step 1: Generate speech tokens on GPU (T3)
        text = gen_params.get('text', text)
        
        # Normalize and tokenize
        from chatterbox.tts_turbo import punc_norm
        text = punc_norm(text)
        text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(gpu_device)
        
        # Move T3 conditionals to GPU for inference
        t3_cond_gpu = model.conds.t3.to(device=gpu_device)
        
        # Generate speech tokens with T3 (GPU)
        speech_tokens = model.t3.inference_turbo(
            t3_cond=t3_cond_gpu,
            text_tokens=text_tokens,
            temperature=gen_params.get('temperature', 0.8),
            top_k=gen_params.get('top_k', 1000),
            top_p=gen_params.get('top_p', 0.95),
            repetition_penalty=gen_params.get('repetition_penalty', 1.2),
        )
        
        # Step 2: Move everything to CPU for S3Gen
        speech_tokens = speech_tokens.cpu()
        
        # Ensure s3gen is on CPU
        model.s3gen = model.s3gen.cpu()
        
        # Move conditionals to CPU - handle nested structure
        def ensure_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(ensure_cpu(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: ensure_cpu(v) for k, v in obj.items()}
            return obj
        
        # Move all conditionals to CPU (avoid deepcopy - tensors aren't leaf tensors)
        gen_conds_cpu = ensure_cpu(model.conds.gen)
        
        # Step 3: Decode to audio on CPU (S3Gen - avoids MIOpen)
        from chatterbox.models.s3gen.const import S3GEN_SIL
        
        # Remove OOV tokens and add silence
        speech_tokens = speech_tokens[speech_tokens < 6561]
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(cpu_device)
        speech_tokens = torch.cat([speech_tokens.to(cpu_device), silence])
        
        # Run S3Gen inference on CPU
        with torch.no_grad():
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=gen_conds_cpu,
                n_cfm_timesteps=2,
            )
        
        # Step 4: Apply watermark and return
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
        
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    async def _generate_with_cpu_fallback(
        self,
        text: str,
        voice_config: VoiceConfig,
        gen_params: dict,
        loop: asyncio.AbstractEventLoop
    ) -> tuple[Any, int]:
        """
        Generate audio using CPU-only mode for Base/Multilingual in hybrid deployments.
        
        Strategy: Temporarily unload GPU model, load CPU model, generate, cleanup.
        This avoids device mismatch issues with complex conditional structures.
        
        Returns:
            Tuple of (wav_tensor, sample_rate)
        """
        model_type = voice_config.model_type
        
        logger.warning(f"ROCm hybrid mode: {model_type} requires CPU fallback, unloading GPU model...")
        
        # Step 1: Save original state and unload GPU model
        original_device = self.device
        async with self._model_lock:
            if model_type in self._models:
                del self._models[model_type]
                logger.info(f"Unloaded {model_type} model from GPU cache")
        
        # Step 2: Temporarily switch to CPU mode
        self.device = "cpu"
        
        # Step 3: Clear any cached conditionals (they were computed on GPU)
        VOICE_MANAGER.clear_conditionals_cache(voice_config.voice_id)
        
        try:
            # Step 4: Load fresh model on CPU
            logger.info(f"Loading {model_type} model on CPU...")
            cpu_model = await self._load_model(model_type)
            
            # Step 5: Prepare conditionals on CPU
            await self._prepare_conditionals(cpu_model, voice_config)
            
            # Step 6: Generate on CPU
            logger.info(f"Generating audio on CPU for {model_type}...")
            if model_type == "multilingual":
                wav = await loop.run_in_executor(
                    None,
                    lambda: cpu_model.generate(
                        text,
                        language_id=gen_params.get("language_id", "en"),
                        audio_prompt_path=gen_params.get("audio_prompt_path"),
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        temperature=gen_params.get("temperature", 0.8),
                        repetition_penalty=gen_params.get("repetition_penalty", 2.0),
                        top_p=gen_params.get("top_p", 1.0),
                    )
                )
            else:  # base model
                wav = await loop.run_in_executor(
                    None,
                    lambda: cpu_model.generate(
                        text,
                        audio_prompt_path=gen_params.get("audio_prompt_path"),
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        temperature=gen_params.get("temperature", 0.8),
                        repetition_penalty=gen_params.get("repetition_penalty", 2.0),
                        top_p=gen_params.get("top_p", 1.0),
                    )
                )
            
            # Capture sample rate before cleanup
            sample_rate = cpu_model.sr
            return wav, sample_rate
            
        finally:
            # Step 7: Cleanup - unload CPU model and restore GPU state
            logger.info(f"Cleaning up CPU {model_type} model, restoring GPU state...")
            async with self._model_lock:
                if model_type in self._models:
                    del self._models[model_type]
            
            # Restore original device
            self.device = original_device
            
            # Clear CPU conditionals cache
            VOICE_MANAGER.clear_conditionals_cache(voice_config.voice_id)
            
            logger.info("CPU fallback complete, GPU state restored")

    @staticmethod
    def _resample_audio(audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Resample audio to change playback speed."""
        from scipy import signal
        new_length = int(len(audio) / speed_factor)
        return signal.resample(audio, new_length)
    
    @staticmethod
    def _encode_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        output_format: str,
    ) -> bytes:
        """Encode audio data to requested format."""
        buffer = io.BytesIO()
        
        if output_format == "wav":
            sf.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        elif output_format == "mp3":
            sf.write(buffer, audio_data, sample_rate, format="MP3")
        elif output_format == "flac":
            sf.write(buffer, audio_data, sample_rate, format="FLAC")
        elif output_format == "opus":
            sf.write(buffer, audio_data, sample_rate, format="OGG", subtype="OPUS")
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        
        return buffer.getvalue()
    
    @staticmethod
    def _get_mime_type(output_format: str) -> str:
        """Get MIME type for audio format."""
        mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "opus": "audio/opus",
        }
        return mime_types.get(output_format, "audio/wav")
    
    async def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "device": self.device,
            "loaded_models": list(self._models.keys()),
            "available_models": list(MODEL_REPOS.keys()),
        }


# Global generator instance
GENERATOR: TTSGenerator | None = None


async def get_generator() -> TTSGenerator:
    """Get or create the global TTS generator."""
    global GENERATOR
    if GENERATOR is None:
        GENERATOR = TTSGenerator()
    return GENERATOR

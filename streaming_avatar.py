"""
Streaming Avatar Pipeline
Handles real-time audio processing and video generation
"""

import os
import asyncio
import logging
import base64
import io
import time
import traceback
from typing import Optional, AsyncGenerator, Tuple
from collections import deque

import numpy as np
import cv2
import torch
from openai import AsyncOpenAI, OpenAIError
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

from idle_animation import IdleAnimationManager, TransitionManager

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track performance metrics for monitoring"""
    
    def __init__(self):
        self.transcription_times = deque(maxlen=100)
        self.llm_response_times = deque(maxlen=100)
        self.tts_generation_times = deque(maxlen=100)
        self.lipsync_generation_times = deque(maxlen=100)
        self.total_latencies = deque(maxlen=100)
        self.frame_generation_times = deque(maxlen=1000)
    
    def record_transcription(self, duration_ms: float):
        self.transcription_times.append(duration_ms)
        logger.info(f"⏱️  Transcription: {duration_ms:.0f}ms")
    
    def record_llm_response(self, duration_ms: float):
        self.llm_response_times.append(duration_ms)
        logger.info(f"⏱️  LLM Response: {duration_ms:.0f}ms")
    
    def record_tts_generation(self, duration_ms: float):
        self.tts_generation_times.append(duration_ms)
        logger.info(f"⏱️  TTS Generation: {duration_ms:.0f}ms")
    
    def record_lipsync(self, duration_ms: float):
        self.lipsync_generation_times.append(duration_ms)
        logger.info(f"⏱️  Lip-sync: {duration_ms:.0f}ms")
    
    def record_total_latency(self, duration_ms: float):
        self.total_latencies.append(duration_ms)
        logger.info(f"⏱️  Total Latency: {duration_ms:.0f}ms")
    
    def record_frame_time(self, duration_ms: float):
        self.frame_generation_times.append(duration_ms)
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            "transcription_avg_ms": avg(self.transcription_times),
            "llm_response_avg_ms": avg(self.llm_response_times),
            "tts_generation_avg_ms": avg(self.tts_generation_times),
            "lipsync_avg_ms": avg(self.lipsync_generation_times),
            "total_latency_avg_ms": avg(self.total_latencies),
            "frame_generation_avg_ms": avg(self.frame_generation_times),
            "fps": 1000 / avg(self.frame_generation_times) if self.frame_generation_times else 0
        }


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection
    Detects when user stops speaking based on silence duration
    """
    
    def __init__(self, silence_duration_ms: int = 2000):
        self.silence_duration_ms = silence_duration_ms
        self.last_speech_time = None
        self.is_speaking = False
    
    def process_audio_chunk(self, audio_data: bytes, has_speech: bool = True) -> bool:
        """
        Process audio chunk and determine if speech has ended
        
        Args:
            audio_data: Raw audio bytes
            has_speech: Whether this chunk contains speech (simplified)
        
        Returns:
            True if speech has ended (silence detected)
        """
        current_time = time.time() * 1000  # milliseconds
        
        if has_speech:
            self.last_speech_time = current_time
            self.is_speaking = True
            return False
        else:
            if self.is_speaking and self.last_speech_time:
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= self.silence_duration_ms:
                    self.is_speaking = False
                    return True  # Speech ended
        
        return False
    
    def reset(self):
        """Reset VAD state"""
        self.last_speech_time = None
        self.is_speaking = False


class AudioBuffer:
    """
    Buffers audio chunks until processing is triggered
    """
    
    def __init__(self, max_duration_seconds: int = 30):
        self.chunks = deque()
        self.max_duration_seconds = max_duration_seconds
        self.total_duration = 0
    
    def add_chunk(self, audio_data: bytes, duration_ms: float):
        """Add audio chunk to buffer"""
        self.chunks.append((audio_data, duration_ms))
        self.total_duration += duration_ms
        
        # Prevent buffer from growing too large
        while self.total_duration > self.max_duration_seconds * 1000:
            removed_chunk = self.chunks.popleft()
            self.total_duration -= removed_chunk[1]
    
    def get_all_audio(self) -> bytes:
        """Get all buffered audio as single bytes object"""
        return b''.join(chunk[0] for chunk in self.chunks)
    
    def clear(self):
        """Clear buffer"""
        self.chunks.clear()
        self.total_duration = 0
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.chunks) == 0


class StreamingAvatar:
    """
    Streaming Avatar Pipeline
    Processes audio input and generates synchronized video output
    """
    
    def __init__(
        self,
        session_id: str,
        avatar_instance,
        models,
        openai_api_key: str,
        elevenlabs_api_key: str,
        silence_duration_ms: int = 2000,
        fps: int = 25
    ):
        self.session_id = session_id
        self.avatar = avatar_instance
        self.models = models
        self.fps = fps
        
        # Initialize API clients with proper configuration
        # Following official documentation:
        # - OpenAI: https://platform.openai.com/docs/api-reference
        # - ElevenLabs: https://elevenlabs.io/docs/api-reference/introduction
        try:
            self.openai_client = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=30.0,  # 30 second timeout
                max_retries=0  # We handle retries manually
            )
            
            self.elevenlabs_client = AsyncElevenLabs(
                api_key=elevenlabs_api_key,
                timeout=30.0  # 30 second timeout
            )
            
            logger.info("✅ API clients initialized (OpenAI + ElevenLabs)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API clients: {e}")
            raise
        
        # Audio processing
        self.audio_buffer = AudioBuffer()
        self.vad = VoiceActivityDetector(silence_duration_ms=silence_duration_ms)
        
        # Idle animation
        self.idle_animation = IdleAnimationManager(avatar_instance, fps=fps)
        self.transition_manager = TransitionManager(transition_frames=5)
        
        # Conversation history
        self.conversation_history = []
        
        # State
        self.is_processing = False
        self.current_state = "idle"  # idle, listening, processing, speaking
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Error tracking
        self.error_count = 0
        self.max_consecutive_errors = 3
        
        logger.info(f"✅ StreamingAvatar initialized for session {session_id}")
    
    async def initialize(self):
        """Initialize avatar and generate idle frames"""
        try:
            logger.info("Initializing idle animation...")
            success = self.idle_animation.generate_idle_frames(num_frames=75)
            if success:
                logger.info("✅ Idle animation ready")
            else:
                logger.warning("⚠️  Idle animation using placeholders")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to initialize idle animation: {e}")
            return False
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict]:
        """
        Process incoming audio chunk
        
        Args:
            audio_data: Raw audio bytes (base64 encoded)
        
        Returns:
            Status update if state changed
        """
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return None
        
        # Add to buffer
        chunk_duration_ms = 100  # Assuming 100ms chunks
        self.audio_buffer.add_chunk(audio_bytes, chunk_duration_ms)
        
        # Update state
        if self.current_state == "idle":
            self.current_state = "listening"
            return {"type": "status", "state": "listening"}
        
        # Check for speech end (simplified VAD)
        # In production, use proper VAD like WebRTC VAD or Silero VAD
        speech_ended = self.vad.process_audio_chunk(audio_bytes, has_speech=True)
        
        if speech_ended and not self.is_processing:
            # Trigger processing
            asyncio.create_task(self._process_buffered_audio())
            return {"type": "status", "state": "processing"}
        
        return None
    
    async def _process_buffered_audio(self):
        """
        Process buffered audio through the pipeline:
        1. Speech-to-Text (Whisper)
        2. LLM Response (OpenAI)
        3. Text-to-Speech (ElevenLabs)
        4. Lip-sync Video Generation (MuseTalk)
        """
        if self.is_processing:
            logger.warning("⚠️  Already processing, skipping")
            return
        
        self.is_processing = True
        self.current_state = "processing"
        pipeline_start_time = time.time()
        
        try:
            logger.info("🔄 Starting processing pipeline...")
            # Get buffered audio
            audio_data = self.audio_buffer.get_all_audio()
            
            if len(audio_data) == 0:
                logger.warning("⚠️  No audio data to process")
                return
            
            logger.info(f"📊 Processing {len(audio_data)} bytes of audio")
            
            # Step 1: Speech-to-Text
            stt_start = time.time()
            transcript = await self._transcribe_audio(audio_data)
            stt_duration = (time.time() - stt_start) * 1000
            self.metrics.record_transcription(stt_duration)
            
            if not transcript or len(transcript.strip()) == 0:
                logger.warning("⚠️  Empty transcript, skipping")
                return
            
            logger.info(f"📝 Transcript: {transcript}")
            
            # Send transcript to client
            yield {
                "type": "transcript",
                "text": transcript,
                "is_final": True,
                "timestamp": time.time()
            }
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": transcript
            })
            
            # Step 2: Get LLM response
            llm_start = time.time()
            ai_response = await self._get_ai_response(transcript)
            llm_duration = (time.time() - llm_start) * 1000
            self.metrics.record_llm_response(llm_duration)
            
            logger.info(f"🤖 AI Response: {ai_response}")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Start transition from idle to talking
            self.transition_manager.start_transition("idle", "talking")
            
            # Step 3 & 4: Generate TTS and Lip-sync (streamed together)
            async for frame_data in self._generate_talking_video(ai_response):
                # Add timestamp for sync
                if "timestamp" not in frame_data:
                    frame_data["timestamp"] = time.time()
                # This will be sent via WebSocket
                yield frame_data
            
            # Start transition from talking back to idle
            self.transition_manager.start_transition("talking", "idle")
            
            # Record total latency
            total_latency = (time.time() - pipeline_start_time) * 1000
            self.metrics.record_total_latency(total_latency)
            
            # Reset error count on success
            self.error_count = 0
            
            logger.info(f"✅ Pipeline complete: {total_latency:.0f}ms total")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in processing pipeline ({self.error_count}/{self.max_consecutive_errors}): {e}")
            logger.error(traceback.format_exc())
            
            # Check if too many consecutive errors
            if self.error_count >= self.max_consecutive_errors:
                error_msg = f"Too many consecutive errors ({self.error_count}). Please restart session."
                logger.error(f"🚨 {error_msg}")
                yield {
                    "type": "error",
                    "message": error_msg,
                    "fatal": True,
                    "timestamp": time.time()
                }
            else:
                yield {
                    "type": "error",
                    "message": str(e),
                    "fatal": False,
                    "timestamp": time.time()
                }
        
        finally:
            # Reset state
            self.audio_buffer.clear()
            self.vad.reset()
            self.is_processing = False
            self.current_state = "idle"
            
            yield {"type": "status", "state": "idle"}
    
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_data: Raw audio bytes
        
        Returns:
            Transcribed text
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🎤 Transcribing audio (attempt {attempt + 1}/{max_retries})...")
                # Save audio to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    audio_path = f.name
                
                try:
                    # Use OpenAI's Whisper API for transcription
                    with open(audio_path, "rb") as audio_file:
                        transcript = await self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )
                    
                    return transcript.strip()
                    
                finally:
                    # Cleanup temp file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  Transcription attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ Transcription failed after {max_retries} attempts: {e}")
                    raise
    
    async def _get_ai_response(self, user_message: str) -> str:
        """
        Get AI response from OpenAI
        
        Args:
            user_message: User's message
        
        Returns:
            AI response text
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Getting AI response (attempt {attempt + 1}/{max_retries})...")
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful, friendly AI assistant. Keep responses concise and natural, as if having a conversation."
                    }
                ] + self.conversation_history[-10:]  # Keep last 10 messages for context
                
                # Get completion using OpenAI Chat Completions API
                # Reference: https://platform.openai.com/docs/api-reference/chat/create
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                    stream=False  # We're not using streaming for chat responses
                )
                
                return response.choices[0].message.content
                
            except OpenAIError as e:
                # Handle OpenAI-specific errors
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  OpenAI API error (attempt {attempt + 1}): {e}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ OpenAI API failed after {max_retries} attempts: {e}")
                    return "I'm sorry, I'm having trouble processing that right now."
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  OpenAI API attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ OpenAI API failed after {max_retries} attempts: {e}")
                    return "I'm sorry, I'm having trouble processing that right now."
    
    async def _generate_talking_video(self, text: str) -> AsyncGenerator[dict, None]:
        """
        Generate talking video with lip-sync
        
        Args:
            text: Text to speak
        
        Yields:
            Video frames and audio chunks
        """
        tts_start = time.time()
        lipsync_start = None
        
        try:
            self.current_state = "speaking"
            yield {
                "type": "status",
                "state": "speaking",
                "timestamp": time.time()
            }
            
            # Step 1: Generate TTS audio from ElevenLabs
            # Using the official streaming method as per ElevenLabs docs:
            # https://elevenlabs.io/docs/api-reference/streaming
            logger.info("🔊 Generating TTS audio with ElevenLabs streaming...")
            
            # Retry logic for ElevenLabs TTS
            max_tts_retries = 3
            tts_retry_delay = 1.0
            audio_chunks = []
            
            for tts_attempt in range(max_tts_retries):
                try:
                    # Use the official stream method from ElevenLabs SDK
                    # Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/stream
                    # Note: convert_as_stream returns an async generator, don't await it!
                    audio_generator = self.elevenlabs_client.text_to_speech.convert_as_stream(
                        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice (default)
                        text=text,
                        model_id="eleven_turbo_v2_5",  # Ultra-low latency (~75ms) for real-time
                        voice_settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=0.75,
                            style=0.0,
                            use_speaker_boost=True
                        ),
                        # Optional: Specify output format for better quality/compatibility
                        output_format="mp3_44100_128"  # High quality MP3 at 44.1kHz
                    )
                    
                    # Stream audio chunks as they arrive
                    # This enables low-latency playback on the client
                    audio_chunks = []
                    async for chunk in audio_generator:
                        if chunk:
                            audio_chunks.append(chunk)
                            # Send audio chunk to client immediately for streaming playback
                            yield {
                                "type": "audio_chunk",
                                "data": base64.b64encode(chunk).decode('utf-8'),
                                "timestamp": time.time()
                            }
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    if tts_attempt < max_tts_retries - 1:
                        logger.warning(f"⚠️  ElevenLabs TTS attempt {tts_attempt + 1} failed: {e}, retrying in {tts_retry_delay}s...")
                        await asyncio.sleep(tts_retry_delay)
                        tts_retry_delay *= 2
                    else:
                        logger.error(f"❌ ElevenLabs TTS failed after {max_tts_retries} attempts: {e}")
                        raise
            
            # Combine audio for lip-sync generation
            full_audio = b''.join(audio_chunks)
            tts_duration = (time.time() - tts_start) * 1000
            self.metrics.record_tts_generation(tts_duration)
            
            logger.info(f"✅ TTS audio generated ({len(full_audio)} bytes, {len(audio_chunks)} chunks)")
            
            # Step 2: Generate lip-sync video
            lipsync_start = time.time()
            
            # Save audio to temp file
            import tempfile
            audio_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(full_audio)
                    audio_path = f.name
                
                logger.info("🎬 Generating lip-sync video...")
                
                # TODO: Integrate actual MuseTalk lip-sync generation
                # This requires the Avatar instance to be properly initialized
                
                # Placeholder: Send some frames with proper timing
                frame_duration = 1.0 / self.fps  # seconds per frame
                num_frames = int(len(full_audio) / 4000 * self.fps)  # Estimate frames from audio length
                
                for i in range(num_frames):
                    frame_start = time.time()
                    
                    # In production, these would be actual generated frames
                    yield {
                        "type": "video_frame",
                        "data": "placeholder_frame_data",
                        "frame_index": i,
                        "timestamp": time.time(),
                        "is_idle": False
                    }
                    
                    # Track frame generation time
                    frame_time = (time.time() - frame_start) * 1000
                    self.metrics.record_frame_time(frame_time)
                    
                    # Maintain frame rate
                    await asyncio.sleep(frame_duration)
                
                lipsync_duration = (time.time() - lipsync_start) * 1000
                self.metrics.record_lipsync(lipsync_duration)
                
                logger.info(f"✅ Lip-sync video generated ({num_frames} frames)")
                
            finally:
                # Cleanup temp file
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            logger.error(traceback.format_exc())
            yield {
                "type": "error",
                "message": f"Video generation failed: {str(e)}",
                "timestamp": time.time()
            }
    
    async def get_idle_frame(self) -> dict:
        """
        Get an idle frame (for when not speaking)
        
        Returns:
            Video frame data with timestamp
        """
        return self.idle_animation.get_next_frame()
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.get_stats()


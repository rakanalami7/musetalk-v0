"""
MuseTalk FastAPI Server
Real-time interactive avatar with WebSocket streaming
"""

import os
import sys
import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from transformers import WhisperModel

# Add MuseTalk modules to path
sys.path.insert(0, os.path.dirname(__file__))

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from session_manager import SessionManager, SessionStatus
from streaming_avatar import StreamingAvatar

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model storage
class ModelStore:
    """Global storage for loaded models"""
    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.weight_dtype = None
        self.timesteps = None
        self.is_initialized = False

models = ModelStore()

# Global session manager
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    Handles startup and shutdown events
    """
    global session_manager
    
    # Startup
    logger.info("🚀 Starting MuseTalk server...")
    
    try:
        # Configure device
        gpu_id = int(os.getenv("GPU_ID", "0"))
        models.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {models.device}")
        
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available! Running on CPU (will be slow)")
        
        # Determine model paths
        project_dir = os.path.dirname(__file__)
        models_dir = os.path.join(project_dir, "models")
        
        # Check if models exist
        unet_path = os.path.join(models_dir, "musetalkV15", "unet.pth")
        unet_config = os.path.join(models_dir, "musetalkV15", "musetalk.json")
        whisper_dir = os.path.join(models_dir, "whisper")
        
        if not os.path.exists(unet_path):
            logger.error(f"❌ UNet model not found at {unet_path}")
            logger.error("Please run download_weights.sh to download models")
            raise FileNotFoundError(f"UNet model not found: {unet_path}")
        
        logger.info("📦 Loading MuseTalk models...")
        
        # Load VAE, UNet, and Position Encoder
        models.vae, models.unet, models.pe = load_all_model(
            unet_model_path=unet_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=models.device
        )
        
        # Determine weight dtype
        use_float16 = os.getenv("USE_FLOAT16", "true").lower() == "true"
        if use_float16 and torch.cuda.is_available():
            logger.info("Using float16 for faster inference")
            models.pe = models.pe.half()
            models.vae.vae = models.vae.vae.half()
            models.unet.model = models.unet.model.half()
            models.weight_dtype = torch.float16
        else:
            logger.info("Using float32")
            models.weight_dtype = torch.float32
        
        # Move models to device
        models.pe = models.pe.to(models.device)
        models.vae.vae = models.vae.vae.to(models.device)
        models.unet.model = models.unet.model.to(models.device)
        
        # Set to eval mode
        models.pe.eval()
        models.vae.vae.eval()
        models.unet.model.eval()
        
        # Disable gradients
        models.pe.requires_grad_(False)
        models.vae.vae.requires_grad_(False)
        models.unet.model.requires_grad_(False)
        
        # Initialize timesteps
        models.timesteps = torch.tensor([0], device=models.device)
        
        logger.info("📦 Loading Whisper model...")
        
        # Initialize audio processor and Whisper
        models.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        models.whisper = WhisperModel.from_pretrained(whisper_dir)
        models.whisper = models.whisper.to(device=models.device, dtype=models.weight_dtype).eval()
        models.whisper.requires_grad_(False)
        
        models.is_initialized = True
        logger.info("✅ All models loaded successfully!")
        logger.info(f"📊 GPU Memory: {torch.cuda.memory_allocated(models.device) / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
        
        # Initialize session manager
        max_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS", "3"))
        timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "5"))
        session_manager = SessionManager(
            max_concurrent_sessions=max_sessions,
            session_timeout_minutes=timeout_minutes
        )
        await session_manager.start_cleanup_task()
        logger.info("✅ Session manager initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MuseTalk server...")
    
    # Cleanup sessions
    if session_manager:
        await session_manager.stop_cleanup_task()
        await session_manager.cleanup_all_sessions()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MuseTalk Server",
    description="Real-time interactive avatar with lip-sync",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MuseTalk Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns server status and model initialization state
    """
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(models.device),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(models.device) / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(models.device) / 1024**3, 2),
        }
    else:
        gpu_info = {"available": False}
    
    return {
        "status": "healthy",
        "models_initialized": models.is_initialized,
        "device": str(models.device),
        "gpu": gpu_info,
        "weight_dtype": str(models.weight_dtype) if models.weight_dtype else None,
    }


@app.get("/api/info")
async def get_info():
    """
    Get server information
    """
    active_sessions = session_manager.get_active_session_count() if session_manager else 0
    max_sessions = session_manager.max_concurrent_sessions if session_manager else 0
    
    return {
        "server": "MuseTalk",
        "version": "1.0.0",
        "models_loaded": models.is_initialized,
        "device": str(models.device),
        "sessions": {
            "active": active_sessions,
            "max": max_sessions,
        },
        "capabilities": {
            "websocket_streaming": True,
            "session_management": True,
            "realtime_lipsync": True,
        }
    }


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.post("/api/session/start")
async def start_session(
    video: UploadFile = File(...),
    bbox_shift: int = Form(0)
):
    """
    Start a new avatar session
    
    Args:
        video: Video file for the avatar
        bbox_shift: Bounding box shift parameter (default: 0)
    
    Returns:
        session_id and status
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    if not models.is_initialized:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Save uploaded video
    try:
        # Create uploads directory
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save video file
        video_filename = f"{int(time.time())}_{video.filename}"
        video_path = os.path.join(uploads_dir, video_filename)
        
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        logger.info(f"Saved uploaded video: {video_path}")
        
    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Create session
    session_id, error = await session_manager.create_session(
        video_path=video_path,
        bbox_shift=bbox_shift
    )
    
    if error:
        # Cleanup video file
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=400, detail=error)
    
    # Start avatar preparation in background
    asyncio.create_task(prepare_avatar(session_id, video_path, bbox_shift))
    
    return {
        "session_id": session_id,
        "status": "initializing",
        "message": "Session created, preparing avatar..."
    }


@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    """
    Get session status
    
    Args:
        session_id: Session ID
    
    Returns:
        Session information
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """
    End a session and cleanup resources
    
    Args:
        session_id: Session ID
    
    Returns:
        Success message
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    success = await session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "message": "Session ended successfully",
        "session_id": session_id
    }


@app.get("/api/sessions")
async def list_sessions():
    """
    List all active sessions
    
    Returns:
        Dictionary of all sessions
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    return {
        "sessions": session_manager.get_all_sessions(),
        "active_count": session_manager.get_active_session_count(),
        "max_sessions": session_manager.max_concurrent_sessions
    }


# ============================================================================
# Avatar Preparation
# ============================================================================

async def prepare_avatar(session_id: str, video_path: str, bbox_shift: int):
    """
    Prepare avatar in background
    This runs the Avatar initialization from realtime_inference.py
    """
    import time
    
    try:
        logger.info(f"Starting avatar preparation for session {session_id}")
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=10)
        
        # Import Avatar class
        from scripts.realtime_inference import Avatar
        from musetalk.utils.face_parsing import FaceParsing
        
        # Initialize face parser
        fp = FaceParsing(
            left_cheek_width=int(os.getenv("LEFT_CHEEK_WIDTH", "90")),
            right_cheek_width=int(os.getenv("RIGHT_CHEEK_WIDTH", "90"))
        )
        
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=20)
        
        # Create Avatar instance
        batch_size = int(os.getenv("BATCH_SIZE", "8"))
        
        logger.info(f"Creating Avatar instance for session {session_id}")
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=50)
        
        # Initialize Avatar with the uploaded video
        avatar = Avatar(
            avatar_id=video_path,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            preparation=True
        )
        
        # Store avatar instance in session
        session = session_manager.get_session(session_id)
        if session:
            session.avatar_instance = avatar
        
        session_manager.update_session_status(session_id, SessionStatus.READY, progress=100)
        logger.info(f"Avatar ready for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to prepare avatar for session {session_id}: {e}")
        session_manager.update_session_status(
            session_id,
            SessionStatus.ERROR,
            error_message=str(e)
        )


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================

@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for bidirectional audio/video streaming
    
    Protocol:
        Client → Server:
            {"type": "audio_chunk", "data": "<base64_audio>", "timestamp": 123456}
            {"type": "start_speaking"}
            {"type": "stop_speaking"}
        
        Server → Client:
            {"type": "status", "state": "listening|processing|speaking"}
            {"type": "transcript", "text": "...", "is_final": true}
            {"type": "video_frame", "data": "<base64_jpeg>", "frame_index": 0, "timestamp": 123456}
            {"type": "audio_chunk", "data": "<base64_audio>", "timestamp": 123456}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")
    
    # Validate session
    if not session_manager:
        await websocket.send_json({"type": "error", "message": "Session manager not initialized"})
        await websocket.close()
        return
    
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    
    if session.status != SessionStatus.READY:
        await websocket.send_json({
            "type": "error",
            "message": f"Session not ready. Current status: {session.status.value}"
        })
        await websocket.close()
        return
    
    # Update session status
    session_manager.update_session_status(session_id, SessionStatus.ACTIVE)
    
    # Initialize streaming avatar
    streaming_avatar = StreamingAvatar(
        session_id=session_id,
        avatar_instance=session.avatar_instance,
        models=models,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        silence_duration_ms=int(os.getenv("SILENCE_DURATION_MS", "2000")),
        fps=int(os.getenv("FPS", "25"))
    )
    
    # Initialize idle animation
    await streaming_avatar.initialize()
    
    # Send initial status
    await websocket.send_json({"type": "status", "state": "ready", "timestamp": time.time()})
    
    # Start idle frame streaming task
    idle_task = asyncio.create_task(stream_idle_frames(websocket, streaming_avatar))
    
    try:
        while True:
            # Receive message from client
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            
            message_type = data.get("type")
            
            if message_type == "audio_chunk":
                # Process audio chunk
                audio_data = data.get("data")
                if audio_data:
                    # Stop idle streaming if active
                    if not idle_task.done():
                        idle_task.cancel()
                    
                    # Process audio
                    status_update = await streaming_avatar.process_audio_chunk(audio_data)
                    if status_update:
                        await websocket.send_json(status_update)
                    
                    # If processing started, stream results
                    if streaming_avatar.is_processing:
                        async for result in streaming_avatar._process_buffered_audio():
                            await websocket.send_json(result)
                        
                        # Resume idle streaming
                        idle_task = asyncio.create_task(stream_idle_frames(websocket, streaming_avatar))
            
            elif message_type == "start_speaking":
                # User started speaking
                if not idle_task.done():
                    idle_task.cancel()
                await websocket.send_json({"type": "status", "state": "listening"})
            
            elif message_type == "stop_speaking":
                # User stopped speaking - trigger processing
                if streaming_avatar.audio_buffer and not streaming_avatar.audio_buffer.is_empty():
                    async for result in streaming_avatar._process_buffered_audio():
                        await websocket.send_json(result)
                
                # Resume idle streaming
                idle_task = asyncio.create_task(stream_idle_frames(websocket, streaming_avatar))
            
            elif message_type == "ping":
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if not idle_task.done():
            idle_task.cancel()
        
        # Update session status
        session_manager.update_session_status(session_id, SessionStatus.IDLE)
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"WebSocket closed for session {session_id}")


async def stream_idle_frames(websocket: WebSocket, streaming_avatar: StreamingAvatar):
    """
    Stream idle frames while waiting for user input
    """
    try:
        frame_duration = streaming_avatar.idle_animation.get_frame_duration_ms() / 1000
        
        while True:
            frame_start = time.time()
            
            # Get idle frame
            idle_frame = await streaming_avatar.get_idle_frame()
            
            # Send to client
            try:
                await websocket.send_json(idle_frame)
            except Exception as e:
                logger.warning(f"Failed to send idle frame: {e}")
                break
            
            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    except asyncio.CancelledError:
        # Task was cancelled (user started speaking)
        logger.debug("Idle frame streaming cancelled")
    except Exception as e:
        logger.error(f"Error streaming idle frames: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )


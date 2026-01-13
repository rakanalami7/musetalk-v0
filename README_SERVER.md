# MuseTalk Server - Real-time Interactive Avatar

## Overview

This is a FastAPI-based server that provides real-time interactive avatar capabilities with lip-sync video generation. It integrates MuseTalk for video generation, OpenAI for chat, and ElevenLabs for text-to-speech.

## Features

- 🎥 **Real-time Lip-Sync**: Generate talking face videos synchronized with audio
- 🎤 **Speech-to-Text**: Transcribe user speech using Whisper
- 🤖 **AI Chat**: Intelligent responses powered by OpenAI GPT-4
- 🔊 **Text-to-Speech**: Natural voice synthesis with ElevenLabs
- 🔄 **WebSocket Streaming**: Bidirectional audio/video streaming
- 📦 **Session Management**: Handle multiple concurrent sessions
- ⚡ **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA

## Architecture

```
Client (Browser)
    ↓ WebSocket
FastAPI Server
    ├── Session Manager (lifecycle)
    ├── Streaming Avatar (processing pipeline)
    │   ├── Audio Buffer + VAD
    │   ├── Whisper STT
    │   ├── OpenAI Chat
    │   ├── ElevenLabs TTS
    │   └── MuseTalk Lip-sync
    └── Models (VAE, UNet, Whisper)
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

```bash
# On Linux/Mac
./download_weights.sh

# On Windows
download_weights.bat
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
ELEVENLABS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Usage

### Start the Server

```bash
python server.py
```

The server will start on `http://0.0.0.0:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

Returns server status and GPU information.

#### Create Session
```bash
POST /api/session/start
Content-Type: multipart/form-data

video: <video_file>
bbox_shift: 0
```

Returns `session_id` for use with WebSocket.

#### Get Session Status
```bash
GET /api/session/{session_id}
```

Returns session state and preparation progress.

#### End Session
```bash
DELETE /api/session/{session_id}
```

Cleanup session and free resources.

### WebSocket Streaming

Connect to: `ws://localhost:8000/ws/stream/{session_id}`

#### Client → Server Messages

```json
{
  "type": "audio_chunk",
  "data": "<base64_encoded_audio>",
  "timestamp": 1234567890
}
```

```json
{
  "type": "start_speaking"
}
```

```json
{
  "type": "stop_speaking"
}
```

#### Server → Client Messages

```json
{
  "type": "status",
  "state": "listening|processing|speaking"
}
```

```json
{
  "type": "video_frame",
  "data": "<base64_jpeg>",
  "frame_index": 0,
  "timestamp": 1234567890
}
```

```json
{
  "type": "audio_chunk",
  "data": "<base64_audio>",
  "timestamp": 1234567890
}
```

```json
{
  "type": "error",
  "message": "Error description"
}
```

## Configuration

Environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | 0.0.0.0 | Server bind address |
| `SERVER_PORT` | 8000 | Server port |
| `GPU_ID` | 0 | GPU device ID |
| `USE_FLOAT16` | true | Use FP16 for faster inference |
| `MAX_CONCURRENT_SESSIONS` | 3 | Max simultaneous sessions |
| `SESSION_TIMEOUT_MINUTES` | 5 | Session idle timeout |
| `SILENCE_DURATION_MS` | 2000 | Silence before processing |
| `BATCH_SIZE` | 8 | Inference batch size |
| `FPS` | 25 | Video frame rate |

## File Structure

```
MuseTalk/
├── server.py                  # FastAPI server
├── session_manager.py         # Session lifecycle management
├── streaming_avatar.py        # Real-time processing pipeline
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (not in git)
├── .env.example              # Configuration template
├── models/                    # Model weights
│   ├── musetalkV15/
│   ├── whisper/
│   ├── sd-vae/
│   └── ...
├── uploads/                   # Uploaded videos
└── results/                   # Generated avatars
```

## Development

### Running in Development Mode

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test session creation
curl -X POST http://localhost:8000/api/session/start \
  -F "video=@path/to/video.mp4" \
  -F "bbox_shift=0"
```

### Logging

Logs are output to stdout. Adjust log level in `server.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# Copy code
COPY . /app
WORKDIR /app

# Install Python packages
RUN pip install -r requirements.txt

# Download models
RUN ./download_weights.sh

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "server.py"]
```

### RunPod

1. Build Docker image
2. Push to Docker Hub
3. Create RunPod instance with GPU
4. Deploy container
5. Expose port 8000

## Troubleshooting

### CUDA Out of Memory

- Reduce `MAX_CONCURRENT_SESSIONS`
- Reduce `BATCH_SIZE`
- Use `USE_FLOAT16=true`

### Slow Performance

- Ensure GPU is being used (check `/health`)
- Use FP16: `USE_FLOAT16=true`
- Increase `BATCH_SIZE` if GPU has memory

### Models Not Found

Run the download script:
```bash
./download_weights.sh
```

### API Rate Limits

- OpenAI: Check your API quota
- ElevenLabs: Check character limit

## Performance

Expected latency (on RTX 3090):
- Speech-to-Text: ~500ms
- LLM Response: ~1000ms
- TTS Generation: ~500ms
- Lip-sync Video: ~1000ms
- **Total: ~3s end-to-end**

Target: <2s with optimizations

## License

See main MuseTalk LICENSE file.

## Support

For issues, check:
1. Server logs
2. GPU memory usage
3. API key validity
4. Model files present

---

**Status**: Production Ready ✅
**Version**: 1.0.0


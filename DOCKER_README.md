# MuseTalk Docker Deployment Guide

This guide explains how to build and run the MuseTalk server using Docker.

## 📋 Prerequisites

### Required
- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 2.0+ (included with Docker Desktop)
- **NVIDIA GPU** with CUDA support
- **nvidia-docker** runtime ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Verify GPU Support
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 Quick Start

### 1. Prepare Environment

```bash
# Navigate to MuseTalk directory
cd MuseTalk

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Add your API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
```

### 2. Download Model Weights

**Option A: Download Before Building (Recommended)**
```bash
# Download models to local directory
./download_weights.sh

# Models will be in ./models/
```

**Option B: Download During First Run**
```bash
# Models will be downloaded when container starts
# (This will take longer on first run)
```

### 3. Build Docker Image

```bash
# From project root
cd ..
docker-compose build

# Or build manually
cd MuseTalk
docker build -t musetalk-server:latest .
```

### 4. Run with Docker Compose

```bash
# Start the server
docker-compose up -d

# View logs
docker-compose logs -f musetalk

# Check status
docker-compose ps

# Stop the server
docker-compose down
```

### 5. Run with Docker (without compose)

```bash
docker run -d \
  --name musetalk-server \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/MuseTalk/models:/app/models \
  -v $(pwd)/MuseTalk/results:/app/results \
  -v $(pwd)/MuseTalk/.env:/app/.env:ro \
  musetalk-server:latest
```

## 🧪 Testing

### Check Server Health

```bash
# Health check endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","timestamp":"..."}
```

### Test API Endpoints

```bash
# Create a session
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"avatar_image": "/path/to/avatar.png"}'

# Check session status
curl http://localhost:8000/api/session/{session_id}/status
```

### Test WebSocket

```bash
# Install wscat if needed
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/ws/stream/{session_id}
```

## 📊 Monitoring

### View Logs

```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f musetalk
```

### Container Stats

```bash
# Real-time stats
docker stats musetalk-server

# GPU usage
nvidia-smi -l 1
```

### Access Container

```bash
# Open shell in running container
docker exec -it musetalk-server bash

# Check Python environment
docker exec -it musetalk-server python --version

# Test imports
docker exec -it musetalk-server python -c "import torch; print(torch.cuda.is_available())"
```

## 🔧 Configuration

### Environment Variables

Edit `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Configuration
DEVICE=cuda
BATCH_SIZE=1
FPS=25

# Session Configuration
MAX_SESSIONS=5
SESSION_TIMEOUT=3600
```

### Volume Mounts

The Docker Compose configuration mounts:

- `./MuseTalk/models` → `/app/models` (model weights)
- `./MuseTalk/results` → `/app/results` (generated videos)
- `./MuseTalk/logs` → `/app/logs` (application logs)
- `./MuseTalk/.env` → `/app/.env` (environment variables)

### Resource Limits

Edit `docker-compose.yml` to adjust resources:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
    reservations:
      cpus: '4'
      memory: 8G
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

```bash
# Reduce batch size in .env
BATCH_SIZE=1

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
```

### Models Not Loading

```bash
# Check models directory
docker exec -it musetalk-server ls -la /app/models

# Download models manually
./download_weights.sh

# Verify volume mount
docker inspect musetalk-server | grep Mounts -A 20
```

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Container Crashes

```bash
# View crash logs
docker logs musetalk-server

# Check exit code
docker inspect musetalk-server | grep ExitCode

# Restart container
docker-compose restart musetalk
```

## 🚀 Deployment

### RunPod Deployment

1. **Create RunPod Account** at [runpod.io](https://runpod.io)

2. **Push Docker Image to Registry**

```bash
# Tag image
docker tag musetalk-server:latest your-registry/musetalk-server:latest

# Push to Docker Hub
docker push your-registry/musetalk-server:latest
```

3. **Deploy on RunPod**
   - Go to RunPod Dashboard
   - Create New Pod
   - Select GPU (e.g., RTX 4090)
   - Use Custom Docker Image: `your-registry/musetalk-server:latest`
   - Set environment variables
   - Expose port 8000
   - Deploy

4. **Configure Networking**
   - Enable HTTP port 8000
   - Get public URL
   - Update client `.env.local` with RunPod URL

### AWS/GCP/Azure Deployment

See deployment guides:
- [AWS ECS with GPU](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)
- [GCP Cloud Run with GPU](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify GPU: `nvidia-smi`
3. Test health: `curl http://localhost:8000/health`
4. Review this guide's troubleshooting section
5. Open an issue on GitHub

---

**Happy Deploying! 🎉**


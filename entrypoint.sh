#!/bin/bash
set -e

echo "🚀 Starting MuseTalk Server..."

# Check if models directory exists and has content
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "⚠️  Models directory is empty. Downloading model weights..."
    ./download_weights.sh || echo "⚠️  Failed to download weights. Please mount models directory."
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected. Running on CPU (will be slow)."
fi

# Check environment variables
if [ -f ".env" ]; then
    echo "✅ Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using default configuration."
fi

# Validate API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set!"
fi

if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "⚠️  ELEVENLABS_API_KEY not set!"
fi

# Create necessary directories
mkdir -p results logs

# Start the server
echo "🎬 Starting FastAPI server on port 8000..."
exec "$@"

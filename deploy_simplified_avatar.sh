#!/bin/bash
# Deploy SimplifiedAvatar on RunPod - Single Command Setup

set -e  # Exit on error

echo "🚀 Deploying SimplifiedAvatar..."
echo ""

cd /workspace/musetalk-v0 || { echo "❌ Directory not found"; exit 1; }

# Step 1: Verify simplified_avatar.py exists
if [ ! -f "simplified_avatar.py" ]; then
    echo "❌ simplified_avatar.py not found!"
    echo "   Run: git pull origin main"
    exit 1
fi

echo "✅ Found simplified_avatar.py"

# Step 2: Backup server.py
echo "📝 Backing up server.py..."
cp server.py "server.py.backup_$(date +%s)"

# Step 3: Update server.py
echo "📝 Updating server.py..."

python3 << 'EOFPYTHON'
import re

with open('server.py', 'r') as f:
    content = f.read()

new_function = '''async def prepare_avatar(session_id: str, video_path: str, bbox_shift: int):
    """
    Prepare avatar in background using SimplifiedAvatar
    Much faster than full MuseTalk preparation
    """
    import time
    from simplified_avatar import SimplifiedAvatar
    
    try:
        logger.info(f"Starting SimplifiedAvatar preparation for session {session_id}")
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=10)
        
        # Create SimplifiedAvatar instance
        batch_size = int(os.getenv("BATCH_SIZE", "8"))
        fps = int(os.getenv("FPS", "25"))
        
        logger.info(f"Creating SimplifiedAvatar for session {session_id}")
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=20)
        
        # Initialize SimplifiedAvatar
        avatar = SimplifiedAvatar(
            avatar_id=session_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            vae=models.vae,
            unet=models.unet,
            pe=models.pe,
            audio_processor=models.audio_processor,
            whisper=models.whisper,
            device=models.device,
            weight_dtype=models.weight_dtype,
            timesteps=models.timesteps,
            batch_size=batch_size,
            fps=fps,
            max_frames=250  # ~10 seconds at 25fps
        )
        
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=40)
        
        # Prepare avatar (extract frames + encode with VAE)
        logger.info(f"Preparing avatar (extracting frames and encoding)...")
        avatar.prepare()
        
        session_manager.update_session_status(session_id, SessionStatus.INITIALIZING, progress=80)
        
        # Store avatar instance in session
        session = session_manager.get_session(session_id)
        if session:
            session.avatar_instance = avatar
        
        session_manager.update_session_status(session_id, SessionStatus.READY, progress=100)
        logger.info(f"✅ SimplifiedAvatar ready for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to prepare avatar for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        session_manager.update_session_status(
            session_id,
            SessionStatus.ERROR,
            error_message=str(e)
        )
'''

# Replace the function
pattern = r'async def prepare_avatar\(.*?\n(?:.*?\n)*?(?=\n\n# ===|@app\.|async def |def |if __name__|$)'
content = re.sub(pattern, new_function + '\n', content, count=1, flags=re.MULTILINE)

with open('server.py', 'w') as f:
    f.write(content)

print("✅ Updated server.py")
EOFPYTHON

# Step 4: Verify
echo ""
echo "🔍 Verifying changes..."
if grep -q "SimplifiedAvatar" server.py; then
    echo "✅ server.py updated successfully"
else
    echo "❌ Failed to update server.py"
    exit 1
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 SimplifiedAvatar Features:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ No mmpose required"
echo "  ✅ Fast preparation (~10-20 seconds)"
echo "  ✅ Real video frames (not placeholders)"
echo "  ✅ VAE-encoded latents for fast inference"
echo "  ✅ Ready for lip-sync with audio"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Start the server:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  python -m uvicorn server:app --host 0.0.0.0 --port 8000"
echo ""


#!/usr/bin/env python3
"""
Download all required model weights for MuseTalk
Uses Python huggingface_hub library instead of CLI for better reliability
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess

def download_models():
    """Download all required models"""
    
    checkpoints_dir = Path("models")
    
    # Create necessary directories
    dirs = [
        "musetalk", "musetalkV15", "syncnet", "dwpose", 
        "face-parse-bisent", "sd-vae", "whisper"
    ]
    for d in dirs:
        (checkpoints_dir / d).mkdir(parents=True, exist_ok=True)
    
    print("📦 Downloading MuseTalk models...")
    
    try:
        # Download MuseTalk V1.0 weights
        print("  → MuseTalk V1.0...")
        for filename in ["musetalk/musetalk.json", "musetalk/pytorch_model.bin"]:
            hf_hub_download(
                repo_id="TMElyralab/MuseTalk",
                filename=filename,
                local_dir=str(checkpoints_dir),
                local_dir_use_symlinks=False
            )
        
        # Download MuseTalk V1.5 weights (CRITICAL - unet.pth)
        print("  → MuseTalk V1.5 (unet.pth)...")
        for filename in ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]:
            hf_hub_download(
                repo_id="TMElyralab/MuseTalk",
                filename=filename,
                local_dir=str(checkpoints_dir),
                local_dir_use_symlinks=False
            )
        
        # Download SD VAE weights
        print("  → SD VAE...")
        for filename in ["config.json", "diffusion_pytorch_model.bin"]:
            hf_hub_download(
                repo_id="stabilityai/sd-vae-ft-mse",
                filename=filename,
                local_dir=str(checkpoints_dir / "sd-vae"),
                local_dir_use_symlinks=False
            )
        
        # Download Whisper weights
        print("  → Whisper...")
        for filename in ["config.json", "pytorch_model.bin", "preprocessor_config.json"]:
            hf_hub_download(
                repo_id="openai/whisper-tiny",
                filename=filename,
                local_dir=str(checkpoints_dir / "whisper"),
                local_dir_use_symlinks=False
            )
        
        # Download DWPose weights
        print("  → DWPose...")
        hf_hub_download(
            repo_id="yzd-v/DWPose",
            filename="dw-ll_ucoco_384.pth",
            local_dir=str(checkpoints_dir / "dwpose"),
            local_dir_use_symlinks=False
        )
        
        # Download SyncNet weights
        print("  → SyncNet...")
        hf_hub_download(
            repo_id="ByteDance/LatentSync",
            filename="latentsync_syncnet.pt",
            local_dir=str(checkpoints_dir / "syncnet"),
            local_dir_use_symlinks=False
        )
        
        # Download Face Parse Bisent weights using gdown and curl
        print("  → Face Parse Bisent...")
        subprocess.run([
            "gdown", "--id", "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
            "-O", str(checkpoints_dir / "face-parse-bisent" / "79999_iter.pth")
        ], check=True)
        
        subprocess.run([
            "curl", "-L", "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "-o", str(checkpoints_dir / "face-parse-bisent" / "resnet18-5c106cde.pth")
        ], check=True)
        
        print("✅ All weights have been downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)


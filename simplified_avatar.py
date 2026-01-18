"""
Simplified Avatar Implementation
No mmpose/face parsing required - fast real-time inference
"""

import os
import cv2
import torch
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class SimplifiedAvatar:
    """
    Simplified avatar that skips face parsing/detection
    Focuses on speed for real-time conversation
    """
    
    def __init__(
        self,
        avatar_id: str,
        video_path: str,
        bbox_shift: int,
        vae,
        unet,
        pe,
        audio_processor,
        whisper,
        device,
        weight_dtype,
        timesteps,
        batch_size: int = 8,
        fps: int = 25,
        max_frames: int = 250  # ~10 seconds at 25fps
    ):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.fps = fps
        self.max_frames = max_frames
        
        # Model references
        self.vae = vae
        self.unet = unet
        self.pe = pe
        self.audio_processor = audio_processor
        self.whisper = whisper
        self.device = device
        self.weight_dtype = weight_dtype
        self.timesteps = timesteps
        
        # Avatar data
        self.frames: List[np.ndarray] = []
        self.latents: Optional[torch.Tensor] = None
        self.frame_height: int = 0
        self.frame_width: int = 0
        self.is_prepared: bool = False
        
        logger.info(f"SimplifiedAvatar created for {avatar_id}")
    
    def prepare(self):
        """
        Prepare avatar by extracting frames and encoding with VAE
        This is much faster than full MuseTalk preparation
        """
        logger.info(f"Preparing SimplifiedAvatar from {self.video_path}")
        
        # Step 1: Extract frames from video
        self._extract_frames()
        
        if len(self.frames) == 0:
            raise ValueError("No frames extracted from video")
        
        logger.info(f"Extracted {len(self.frames)} frames")
        
        # Step 2: Encode frames with VAE
        self._encode_frames()
        
        self.is_prepared = True
        logger.info(f"✅ SimplifiedAvatar prepared: {len(self.frames)} frames, latents shape: {self.latents.shape}")
    
    def _extract_frames(self):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        
        frame_count = 0
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to 256x256 for MuseTalk
            frame = cv2.resize(frame, (256, 256))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(self.frames) > 0:
            self.frame_height, self.frame_width = self.frames[0].shape[:2]
    
    def _encode_frames(self):
        """Encode frames with VAE to get latents"""
        logger.info("Encoding frames with VAE...")
        
        # Convert frames to tensor
        frames_tensor = torch.from_numpy(np.array(self.frames)).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]
        frames_tensor = (frames_tensor - 0.5) * 2  # Normalize to [-1, 1]
        frames_tensor = frames_tensor.to(device=self.device, dtype=self.weight_dtype)
        
        # Encode in batches
        latent_list = []
        for i in range(0, len(self.frames), self.batch_size):
            batch = frames_tensor[i:i + self.batch_size]
            with torch.no_grad():
                latents = self.vae.encode(batch).latent_dist.sample()
                latents = latents * 0.18215  # VAE scaling factor
            latent_list.append(latents)
        
        # Concatenate all latents
        self.latents = torch.cat(latent_list, dim=0)
        
        logger.info(f"Encoded {len(self.frames)} frames to latents: {self.latents.shape}")
    
    def get_frame(self, index: int) -> np.ndarray:
        """Get a specific frame (for idle animation)"""
        if not self.is_prepared or len(self.frames) == 0:
            # Return black frame as fallback
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Loop frames
        idx = index % len(self.frames)
        return self.frames[idx].copy()
    
    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return len(self.frames)
    
    def infer_once(self, audio_feature: torch.Tensor) -> np.ndarray:
        """
        Generate a single lip-synced frame from audio features
        
        Args:
            audio_feature: Audio features from Whisper (shape: [1, seq_len, feature_dim])
        
        Returns:
            Generated frame as numpy array (H, W, 3)
        """
        if not self.is_prepared:
            raise RuntimeError("Avatar not prepared. Call prepare() first.")
        
        # Use first frame's latent as base
        base_latent = self.latents[0:1]  # (1, C, H, W)
        
        # Process audio feature with position encoder
        with torch.no_grad():
            # Get audio embedding
            audio_feature = audio_feature.to(device=self.device, dtype=self.weight_dtype)
            audio_emb = self.pe(audio_feature)  # (1, seq_len, emb_dim)
            
            # Run UNet to get modified latent
            latent_model_input = base_latent
            noise_pred = self.unet.model(
                latent_model_input,
                self.timesteps,
                encoder_hidden_states=audio_emb
            ).sample
            
            # Decode latent to image
            latents = 1 / 0.18215 * noise_pred
            image = self.vae.decode(latents).sample
            
            # Convert to numpy
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
        
        return image
    
    def infer_batch(self, audio_features: torch.Tensor) -> List[np.ndarray]:
        """
        Generate multiple lip-synced frames from audio features
        
        Args:
            audio_features: Audio features from Whisper (shape: [batch, seq_len, feature_dim])
        
        Returns:
            List of generated frames
        """
        frames = []
        for i in range(audio_features.shape[0]):
            frame = self.infer_once(audio_features[i:i+1])
            frames.append(frame)
        return frames
    
    def cleanup(self):
        """Cleanup resources"""
        self.frames = []
        self.latents = None
        self.is_prepared = False
        logger.info(f"SimplifiedAvatar {self.avatar_id} cleaned up")


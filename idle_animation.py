"""
Idle Animation System
Generates and manages idle/blinking frames for the avatar
"""

import os
import logging
import random
import time
import base64
from typing import List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class IdleAnimationManager:
    """
    Manages idle animation frames with blinking
    Pre-generates frames for smooth looping
    """
    
    def __init__(self, avatar_instance, fps: int = 25):
        self.avatar = avatar_instance
        self.fps = fps
        self.frame_duration_ms = 1000 / fps  # milliseconds per frame
        
        # Idle frame storage
        self.idle_frames: List[bytes] = []
        self.current_frame_index = 0
        
        # Blinking parameters
        self.blink_interval_frames = fps * 3  # Blink every 3 seconds
        self.blink_duration_frames = 3  # Blink lasts 3 frames (~120ms at 25fps)
        self.frames_since_blink = 0
        
        # State
        self.is_initialized = False
        
        logger.info(f"IdleAnimationManager initialized (fps={fps})")
    
    def generate_idle_frames(self, num_frames: int = 75) -> bool:
        """
        Pre-generate idle frames with blinking
        
        Args:
            num_frames: Number of frames to generate (default: 75 = 3 seconds at 25fps)
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Generating {num_frames} idle frames...")
            
            # Check if avatar has frames (SimplifiedAvatar or full Avatar)
            if not self.avatar:
                logger.warning("Avatar not provided, using placeholder frames")
                self._generate_placeholder_frames(num_frames)
                return True
            
            # Get base frame from avatar (support both SimplifiedAvatar and full Avatar)
            if hasattr(self.avatar, 'frames') and len(self.avatar.frames) > 0:
                # SimplifiedAvatar
                base_frame = self.avatar.frames[0]
            elif hasattr(self.avatar, 'frame_list_cycle') and len(self.avatar.frame_list_cycle) > 0:
                # Full Avatar
                base_frame = self.avatar.frame_list_cycle[0]
            else:
                logger.warning("Avatar not properly initialized, using placeholder frames")
                self._generate_placeholder_frames(num_frames)
                return True
            
            # Generate frames with occasional blinks
            for i in range(num_frames):
                # Determine if this frame should be a blink
                is_blink_frame = False
                
                # Blink every ~3 seconds, lasting 3 frames
                if i % self.blink_interval_frames < self.blink_duration_frames:
                    if i % self.blink_interval_frames == 0:
                        # Start of blink
                        is_blink_frame = True
                    elif i % self.blink_interval_frames < self.blink_duration_frames:
                        # Middle/end of blink
                        is_blink_frame = True
                
                # Generate frame
                if is_blink_frame:
                    frame = self._apply_blink_effect(base_frame, i % self.blink_interval_frames)
                else:
                    frame = base_frame.copy()
                
                # Add subtle movement/variation
                frame = self._add_subtle_variation(frame, i)
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                self.idle_frames.append(frame_bytes)
            
            self.is_initialized = True
            logger.info(f"✅ Generated {len(self.idle_frames)} idle frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate idle frames: {e}")
            # Fallback to placeholder
            self._generate_placeholder_frames(num_frames)
            return False
    
    def _apply_blink_effect(self, frame: np.ndarray, blink_phase: int) -> np.ndarray:
        """
        Apply blinking effect to frame
        
        Args:
            frame: Input frame
            blink_phase: 0=closing, 1=closed, 2=opening
        
        Returns:
            Frame with blink effect
        """
        # This is a simplified blink - in production, you'd use face landmarks
        # to properly close the eyes
        
        result = frame.copy()
        
        # Apply slight darkening around eye region during blink
        # This is a placeholder - proper implementation would use face detection
        if blink_phase == 1:  # Fully closed
            # Darken upper portion slightly
            h, w = result.shape[:2]
            eye_region = result[int(h*0.3):int(h*0.5), :]
            result[int(h*0.3):int(h*0.5), :] = cv2.addWeighted(
                eye_region, 0.9, eye_region, 0, 0
            )
        
        return result
    
    def _add_subtle_variation(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Add subtle variation to prevent static appearance
        
        Args:
            frame: Input frame
            frame_index: Current frame index
        
        Returns:
            Frame with subtle variation
        """
        result = frame.copy()
        
        # Add very subtle brightness variation (breathing effect)
        variation = np.sin(frame_index * 0.05) * 2  # ±2 brightness
        result = np.clip(result.astype(float) + variation, 0, 255).astype(np.uint8)
        
        return result
    
    def _generate_placeholder_frames(self, num_frames: int):
        """Generate placeholder frames when avatar not available"""
        logger.info("Generating placeholder idle frames")
        
        # Create a simple placeholder image
        placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
        placeholder[:] = (50, 50, 50)  # Dark gray
        
        # Add text
        cv2.putText(
            placeholder,
            "Avatar Loading...",
            (100, 256),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2
        )
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        # Duplicate for all frames
        self.idle_frames = [frame_bytes] * num_frames
        self.is_initialized = True
    
    def get_next_frame(self) -> dict:
        """
        Get next idle frame in sequence
        
        Returns:
            Frame data with metadata
        """
        if not self.is_initialized or len(self.idle_frames) == 0:
            logger.warning("Idle frames not initialized")
            return {
                "type": "video_frame",
                "data": "",
                "frame_index": 0,
                "timestamp": time.time(),
                "is_idle": True
            }
        
        # Get current frame
        frame_bytes = self.idle_frames[self.current_frame_index]
        
        # Encode to base64
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        
        # Prepare response
        frame_data = {
            "type": "video_frame",
            "data": frame_base64,
            "frame_index": self.current_frame_index,
            "timestamp": time.time(),
            "is_idle": True
        }
        
        # Advance to next frame (loop)
        self.current_frame_index = (self.current_frame_index + 1) % len(self.idle_frames)
        
        return frame_data
    
    def reset(self):
        """Reset to first frame"""
        self.current_frame_index = 0
    
    def get_frame_duration_ms(self) -> float:
        """Get duration per frame in milliseconds"""
        return self.frame_duration_ms


class TransitionManager:
    """
    Manages smooth transitions between idle and talking states
    """
    
    def __init__(self, transition_frames: int = 5):
        self.transition_frames = transition_frames
        self.current_transition_frame = 0
        self.is_transitioning = False
        self.transition_type = None  # 'idle_to_talking' or 'talking_to_idle'
    
    def start_transition(self, from_state: str, to_state: str):
        """
        Start a transition between states
        
        Args:
            from_state: 'idle' or 'talking'
            to_state: 'idle' or 'talking'
        """
        self.is_transitioning = True
        self.current_transition_frame = 0
        self.transition_type = f"{from_state}_to_{to_state}"
        logger.info(f"Starting transition: {self.transition_type}")
    
    def get_transition_alpha(self) -> float:
        """
        Get alpha value for current transition frame
        
        Returns:
            Alpha value between 0 and 1
        """
        if not self.is_transitioning:
            return 1.0
        
        # Linear interpolation
        alpha = self.current_transition_frame / self.transition_frames
        return min(1.0, max(0.0, alpha))
    
    def advance_transition(self) -> bool:
        """
        Advance to next transition frame
        
        Returns:
            True if transition is complete
        """
        if not self.is_transitioning:
            return True
        
        self.current_transition_frame += 1
        
        if self.current_transition_frame >= self.transition_frames:
            self.is_transitioning = False
            logger.info("Transition complete")
            return True
        
        return False
    
    def blend_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Blend two frames based on transition alpha
        
        Args:
            frame1: First frame
            frame2: Second frame
        
        Returns:
            Blended frame
        """
        alpha = self.get_transition_alpha()
        
        # Blend frames
        blended = cv2.addWeighted(
            frame1, 1 - alpha,
            frame2, alpha,
            0
        )
        
        return blended


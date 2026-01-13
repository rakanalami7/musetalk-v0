"""
Session Manager for MuseTalk Server
Handles avatar session lifecycle and resource management
"""

import os
import time
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import shutil
from uuid import uuid4

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status states"""
    INITIALIZING = "initializing"  # Session created, preparing avatar
    READY = "ready"                # Avatar prepared, ready for streaming
    ACTIVE = "active"              # Currently streaming
    IDLE = "idle"                  # No activity, but session alive
    ENDED = "ended"                # Session ended, cleaning up
    ERROR = "error"                # Error occurred


@dataclass
class SessionInfo:
    """Information about an avatar session"""
    session_id: str
    status: SessionStatus
    video_path: str
    bbox_shift: int
    created_at: datetime
    last_activity: datetime
    avatar_instance: Optional[Any] = None
    error_message: Optional[str] = None
    preparation_progress: int = 0  # 0-100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "preparation_progress": self.preparation_progress,
            "error_message": self.error_message,
        }
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_timed_out(self, timeout_minutes: int) -> bool:
        """Check if session has timed out"""
        timeout_delta = timedelta(minutes=timeout_minutes)
        return datetime.now() - self.last_activity > timeout_delta


class SessionManager:
    """
    Manages avatar sessions
    - Creates and destroys sessions
    - Tracks session state
    - Enforces resource limits
    - Handles cleanup
    """
    
    def __init__(
        self,
        max_concurrent_sessions: int = 3,
        session_timeout_minutes: int = 5,
        results_dir: str = "./results"
    ):
        self.sessions: Dict[str, SessionInfo] = {}
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout_minutes = session_timeout_minutes
        self.results_dir = results_dir
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"SessionManager initialized: max_sessions={max_concurrent_sessions}, timeout={session_timeout_minutes}min")
    
    async def start_cleanup_task(self):
        """Start background task to cleanup timed-out sessions"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started session cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped session cleanup task")
    
    async def _cleanup_loop(self):
        """Background task to periodically cleanup timed-out sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_timed_out_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_timed_out_sessions(self):
        """Remove sessions that have timed out"""
        timed_out = []
        
        for session_id, session in self.sessions.items():
            if session.status not in [SessionStatus.ENDED, SessionStatus.ERROR]:
                if session.is_timed_out(self.session_timeout_minutes):
                    timed_out.append(session_id)
        
        for session_id in timed_out:
            logger.info(f"Session {session_id} timed out, cleaning up")
            await self.delete_session(session_id)
    
    def can_create_session(self) -> tuple[bool, Optional[str]]:
        """
        Check if a new session can be created
        Returns (can_create, error_message)
        """
        active_count = sum(
            1 for s in self.sessions.values()
            if s.status not in [SessionStatus.ENDED, SessionStatus.ERROR]
        )
        
        if active_count >= self.max_concurrent_sessions:
            return False, f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached"
        
        return True, None
    
    async def create_session(
        self,
        video_path: str,
        bbox_shift: int = 0
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Create a new session
        Returns (session_id, error_message)
        """
        # Check if we can create a new session
        can_create, error_msg = self.can_create_session()
        if not can_create:
            logger.warning(f"Cannot create session: {error_msg}")
            return None, error_msg
        
        # Validate video path
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            return None, error_msg
        
        # Generate session ID
        session_id = str(uuid4())
        
        # Create session info
        session = SessionInfo(
            session_id=session_id,
            status=SessionStatus.INITIALIZING,
            video_path=video_path,
            bbox_shift=bbox_shift,
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for video: {video_path}")
        
        return session_id, None
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session:
            session.update_activity()
        return session
    
    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None,
        progress: Optional[int] = None
    ):
        """Update session status"""
        session = self.sessions.get(session_id)
        if session:
            session.status = status
            session.update_activity()
            
            if error_message:
                session.error_message = error_message
            
            if progress is not None:
                session.preparation_progress = progress
            
            logger.info(f"Session {session_id} status: {status.value}")
    
    def set_avatar_instance(self, session_id: str, avatar_instance: Any):
        """Set the Avatar instance for a session"""
        session = self.sessions.get(session_id)
        if session:
            session.avatar_instance = avatar_instance
            logger.info(f"Avatar instance set for session {session_id}")
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and cleanup resources
        Returns True if successful
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return False
        
        logger.info(f"Deleting session {session_id}")
        
        # Update status
        session.status = SessionStatus.ENDED
        
        # Cleanup avatar files
        try:
            if session.avatar_instance:
                # Get avatar path from instance
                avatar_path = getattr(session.avatar_instance, 'avatar_path', None)
                if avatar_path and os.path.exists(avatar_path):
                    shutil.rmtree(avatar_path)
                    logger.info(f"Cleaned up avatar files: {avatar_path}")
        except Exception as e:
            logger.error(f"Error cleaning up avatar files: {e}")
        
        # Remove from sessions dict
        del self.sessions[session_id]
        logger.info(f"Session {session_id} deleted")
        
        return True
    
    def get_all_sessions(self) -> Dict[str, dict]:
        """Get all sessions as dictionaries"""
        return {
            session_id: session.to_dict()
            for session_id, session in self.sessions.items()
        }
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        return sum(
            1 for s in self.sessions.values()
            if s.status not in [SessionStatus.ENDED, SessionStatus.ERROR]
        )
    
    async def cleanup_all_sessions(self):
        """Cleanup all sessions (for shutdown)"""
        logger.info("Cleaning up all sessions")
        session_ids = list(self.sessions.keys())
        
        for session_id in session_ids:
            await self.delete_session(session_id)
        
        logger.info("All sessions cleaned up")


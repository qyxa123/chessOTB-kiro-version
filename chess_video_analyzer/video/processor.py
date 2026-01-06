"""
Video processing module for handling video input and frame extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator
import logging

from ..core.data_models import VideoMetadata


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass


class UnsupportedFormatError(VideoProcessingError):
    """Raised when video format is not supported."""
    pass


class CorruptedFileError(VideoProcessingError):
    """Raised when video file is corrupted or unreadable."""
    pass


class VideoProcessor:
    """
    Handles video file input and frame extraction for chess game analysis.
    
    Supports MP4, AVI, and MOV formats with configurable frame extraction intervals.
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov'}
    
    def __init__(self):
        """Initialize the video processor."""
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_file_path: Optional[str] = None
        
    def load_video(self, file_path: str) -> VideoMetadata:
        """
        Load a video file and extract metadata.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            VideoMetadata: Metadata about the loaded video
            
        Raises:
            UnsupportedFormatError: If the video format is not supported
            CorruptedFileError: If the video file is corrupted or unreadable
            FileNotFoundError: If the video file does not exist
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            supported_list = ', '.join(self.SUPPORTED_FORMATS)
            raise UnsupportedFormatError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {supported_list}"
            )
        
        # Try to open the video file
        video_capture = cv2.VideoCapture(file_path)
        
        if not video_capture.isOpened():
            raise CorruptedFileError(
                f"Cannot open video file: {file_path}. "
                "The file may be corrupted or in an unsupported codec."
            )
        
        # Extract metadata
        try:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate metadata
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise CorruptedFileError(
                    f"Invalid video metadata detected. "
                    f"FPS: {fps}, Frames: {frame_count}, Resolution: {width}x{height}"
                )
            
            duration = frame_count / fps
            
            metadata = VideoMetadata(
                duration=duration,
                fps=fps,
                resolution=(width, height),
                format=path.suffix.lower()
            )
            
        except Exception as e:
            video_capture.release()
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"Error reading video metadata: {str(e)}")
        
        # Store the video capture and metadata
        if self._video_capture is not None:
            self._video_capture.release()
            
        self._video_capture = video_capture
        self._metadata = metadata
        self._current_file_path = file_path
        
        logging.info(f"Successfully loaded video: {file_path}")
        logging.info(f"Duration: {duration:.2f}s, FPS: {fps}, Resolution: {width}x{height}")
        
        return metadata
    
    def extract_frames(self, interval: float = 1.0) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from the loaded video at specified intervals.
        
        Args:
            interval: Time interval between extracted frames in seconds
            
        Yields:
            np.ndarray: Video frames as numpy arrays
            
        Raises:
            ValueError: If no video is loaded or interval is invalid
        """
        if self._video_capture is None or self._metadata is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        
        # Calculate frame step based on interval and FPS
        frame_step = max(1, int(interval * self._metadata.fps))
        
        # Reset video to beginning
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_number = 0
        frames_extracted = 0
        
        while True:
            # Set position to the next frame we want to extract
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = self._video_capture.read()
            if not ret:
                break
                
            yield frame
            frames_extracted += 1
            frame_number += frame_step
            
        logging.info(f"Extracted {frames_extracted} frames with interval {interval}s")
    
    def extract_frames_list(self, interval: float = 1.0) -> List[np.ndarray]:
        """
        Extract frames from the loaded video and return as a list.
        
        Args:
            interval: Time interval between extracted frames in seconds
            
        Returns:
            List[np.ndarray]: List of video frames as numpy arrays
        """
        return list(self.extract_frames(interval))
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific timestamp.
        
        Args:
            timestamp: Time in seconds to extract frame from
            
        Returns:
            np.ndarray or None: Frame at the specified timestamp, or None if invalid
        """
        if self._video_capture is None or self._metadata is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        if timestamp < 0 or timestamp > self._metadata.duration:
            return None
        
        # Calculate frame number
        frame_number = int(timestamp * self._metadata.fps)
        
        # Set position and read frame
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._video_capture.read()
        
        return frame if ret else None
    
    def get_metadata(self) -> Optional[VideoMetadata]:
        """
        Get metadata of the currently loaded video.
        
        Returns:
            VideoMetadata or None: Video metadata if a video is loaded
        """
        return self._metadata
    
    def is_loaded(self) -> bool:
        """
        Check if a video is currently loaded.
        
        Returns:
            bool: True if a video is loaded, False otherwise
        """
        return self._video_capture is not None and self._metadata is not None
    
    def close(self):
        """Release the video capture and clean up resources."""
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        self._metadata = None
        self._current_file_path = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()
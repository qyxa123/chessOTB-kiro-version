"""
Video processing module for handling video input and frame extraction.
"""

from .processor import VideoProcessor, VideoProcessingError, UnsupportedFormatError, CorruptedFileError

__all__ = ['VideoProcessor', 'VideoProcessingError', 'UnsupportedFormatError', 'CorruptedFileError']
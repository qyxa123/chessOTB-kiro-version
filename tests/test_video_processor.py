"""
Property-based tests for the VideoProcessor module.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import patch, MagicMock

from chess_video_analyzer.video import VideoProcessor, UnsupportedFormatError, CorruptedFileError
from chess_video_analyzer.core.data_models import VideoMetadata


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.processor.close()
    
    def create_test_video(self, filename: str, fps: float = 30.0, duration: float = 1.0, 
                         width: int = 640, height: int = 480) -> str:
        """
        Create a test video file for testing purposes.
        
        Args:
            filename: Name of the video file to create
            fps: Frames per second
            duration: Duration in seconds
            width: Video width
            height: Video height
            
        Returns:
            str: Path to the created video file
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, filename)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Generate frames
        frame_count = int(fps * duration)
        for i in range(frame_count):
            # Create a simple test frame (gradient)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255) // frame_count  # Red channel varies with frame
            frame[:, :, 1] = 128  # Green channel constant
            frame[:, :, 2] = 255 - ((i * 255) // frame_count)  # Blue channel inverse of red
            out.write(frame)
        
        out.release()
        return video_path
    
    @given(
        fps=st.floats(min_value=10.0, max_value=30.0),
        duration=st.floats(min_value=1.0, max_value=3.0),
        width=st.integers(min_value=320, max_value=640),
        height=st.integers(min_value=240, max_value=480)
    )
    @settings(max_examples=10, deadline=15000)  # Reduced examples for video creation
    def test_property_video_file_processing(self, fps, duration, width, height):
        """
        **Feature: chess-video-analyzer, Property 1: Video File Processing**
        
        For any valid video file in supported formats (MP4, AVI, MOV), 
        the Video_Processor should successfully load and begin processing 
        the file regardless of input method.
        
        **Validates: Requirements 1.1, 1.2, 1.4**
        """
        # Ensure dimensions are even (required by some codecs)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        # Test with different supported formats
        for extension in ['.mp4', '.avi', '.mov']:
            filename = f"test_video{extension}"
            
            try:
                # Create test video
                video_path = self.create_test_video(
                    filename, fps=fps, duration=duration, width=width, height=height
                )
                
                # Test loading the video
                metadata = self.processor.load_video(video_path)
                
                # Verify metadata is correct
                assert isinstance(metadata, VideoMetadata)
                assert metadata.fps > 0
                assert metadata.duration > 0
                assert metadata.resolution == (width, height)
                assert metadata.format == extension
                
                # Verify processor state
                assert self.processor.is_loaded()
                assert self.processor.get_metadata() == metadata
                
                # Test frame extraction works
                frames = list(self.processor.extract_frames(interval=duration/2))
                assert len(frames) >= 1
                assert all(isinstance(frame, np.ndarray) for frame in frames)
                assert all(frame.shape == (height, width, 3) for frame in frames)
                
            finally:
                # Clean up
                self.processor.close()
                if 'video_path' in locals():
                    try:
                        os.remove(video_path)
                        os.rmdir(os.path.dirname(video_path))
                    except (OSError, FileNotFoundError):
                        pass
    
    def test_unsupported_format_error(self):
        """Test that unsupported formats raise appropriate errors."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"not a video file")
            temp_path = f.name
        
        try:
            with pytest.raises(UnsupportedFormatError) as exc_info:
                self.processor.load_video(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Unsupported video format" in error_msg
            assert ".txt" in error_msg
            assert "Supported formats" in error_msg
            
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found_error(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_video("nonexistent_file.mp4")
    
    @patch('cv2.VideoCapture')
    def test_corrupted_file_error(self, mock_video_capture):
        """Test that corrupted files raise CorruptedFileError."""
        # Mock a video capture that fails to open
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture
        
        # Create a dummy file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(CorruptedFileError) as exc_info:
                self.processor.load_video(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Cannot open video file" in error_msg
            assert "corrupted" in error_msg.lower()
            
        finally:
            os.unlink(temp_path)
    
    @given(interval=st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=10, deadline=5000)
    def test_frame_extraction_intervals(self, interval):
        """Test frame extraction with different intervals."""
        # Create a test video
        video_path = self.create_test_video("test_interval.mp4", fps=30.0, duration=2.0)
        
        try:
            self.processor.load_video(video_path)
            frames = list(self.processor.extract_frames(interval=interval))
            
            # Should extract at least one frame
            assert len(frames) >= 1
            
            # All frames should be valid numpy arrays
            assert all(isinstance(frame, np.ndarray) for frame in frames)
            assert all(frame.shape == (480, 640, 3) for frame in frames)
            
        finally:
            self.processor.close()
            try:
                os.remove(video_path)
                os.rmdir(os.path.dirname(video_path))
            except (OSError, FileNotFoundError):
                pass
    
    def test_context_manager(self):
        """Test that VideoProcessor works as a context manager."""
        video_path = self.create_test_video("test_context.mp4")
        
        try:
            with VideoProcessor() as processor:
                metadata = processor.load_video(video_path)
                assert processor.is_loaded()
                assert isinstance(metadata, VideoMetadata)
            
            # After context exit, resources should be cleaned up
            # Note: We can't easily test this without accessing private members
            
        finally:
            try:
                os.remove(video_path)
                os.rmdir(os.path.dirname(video_path))
            except (OSError, FileNotFoundError):
                pass
    
    def test_multiple_video_loading(self):
        """Test loading multiple videos sequentially."""
        video1_path = self.create_test_video("test1.mp4", duration=1.0)
        video2_path = self.create_test_video("test2.mp4", duration=2.0)
        
        try:
            # Load first video
            metadata1 = self.processor.load_video(video1_path)
            assert metadata1.duration == pytest.approx(1.0, rel=0.1)
            
            # Load second video (should replace first)
            metadata2 = self.processor.load_video(video2_path)
            assert metadata2.duration == pytest.approx(2.0, rel=0.1)
            
            # Current metadata should be from second video
            assert self.processor.get_metadata() == metadata2
            
        finally:
            self.processor.close()
            for path in [video1_path, video2_path]:
                try:
                    os.remove(path)
                    os.rmdir(os.path.dirname(path))
                except (OSError, FileNotFoundError):
                    pass


class TestVideoProcessorErrorHandling:
    """Test suite for VideoProcessor error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.processor.close()
    
    @given(
        invalid_extension=st.sampled_from(['.txt', '.jpg', '.png', '.pdf', '.doc', '.xyz'])
    )
    def test_property_error_handling_invalid_inputs(self, invalid_extension):
        """
        **Feature: chess-video-analyzer, Property 2: Error Handling for Invalid Inputs**
        
        For any unsupported or corrupted video file, the Video_Processor should 
        return descriptive error messages and handle the error gracefully without crashing.
        
        **Validates: Requirements 1.3, 1.5**
        """
        # Create a file with invalid extension
        with tempfile.NamedTemporaryFile(suffix=invalid_extension, delete=False) as f:
            f.write(b"not a video file content")
            temp_path = f.name
        
        try:
            # Should raise UnsupportedFormatError with descriptive message
            with pytest.raises(UnsupportedFormatError) as exc_info:
                self.processor.load_video(temp_path)
            
            error_msg = str(exc_info.value)
            
            # Verify error message is descriptive
            assert "Unsupported video format" in error_msg
            assert invalid_extension in error_msg
            assert "Supported formats" in error_msg
            
            # Verify supported formats are listed
            for fmt in VideoProcessor.SUPPORTED_FORMATS:
                assert fmt in error_msg
            
            # Verify processor remains in clean state after error
            assert not self.processor.is_loaded()
            assert self.processor.get_metadata() is None
            
        finally:
            os.unlink(temp_path)
    
    @patch('cv2.VideoCapture')
    def test_corrupted_file_handling(self, mock_video_capture):
        """Test handling of corrupted video files."""
        # Test case 1: VideoCapture fails to open
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(CorruptedFileError) as exc_info:
                self.processor.load_video(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Cannot open video file" in error_msg
            assert "corrupted" in error_msg.lower()
            
        finally:
            os.unlink(temp_path)
    
    @patch('cv2.VideoCapture')
    def test_invalid_metadata_handling(self, mock_video_capture):
        """Test handling of videos with invalid metadata."""
        # Mock a video capture with invalid metadata
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0,  # Invalid FPS
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_video_capture.return_value = mock_capture
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(CorruptedFileError) as exc_info:
                self.processor.load_video(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Invalid video metadata" in error_msg
            
            # Verify capture is properly released on error
            mock_capture.release.assert_called_once()
            
        finally:
            os.unlink(temp_path)
    
    def test_no_video_loaded_errors(self):
        """Test proper error handling when no video is loaded."""
        # Should raise ValueError when trying to extract frames without loading video
        with pytest.raises(ValueError) as exc_info:
            list(self.processor.extract_frames())
        
        assert "No video loaded" in str(exc_info.value)
        
        # Should raise ValueError when trying to get frame at time without loading video
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_frame_at_time(1.0)
        
        assert "No video loaded" in str(exc_info.value)
    
    @given(invalid_interval=st.floats(max_value=0.0))
    def test_invalid_interval_error(self, invalid_interval):
        """Test error handling for invalid frame extraction intervals."""
        # Create and load a test video first
        video_path = None
        try:
            # Create a minimal test video
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "test.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            
            # Write a few frames
            for _ in range(10):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            # Load the video
            self.processor.load_video(video_path)
            
            # Test invalid interval
            with pytest.raises(ValueError) as exc_info:
                list(self.processor.extract_frames(interval=invalid_interval))
            
            assert "Interval must be positive" in str(exc_info.value)
            
        finally:
            if video_path:
                try:
                    os.remove(video_path)
                    os.rmdir(os.path.dirname(video_path))
                except (OSError, FileNotFoundError):
                    pass
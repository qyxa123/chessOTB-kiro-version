"""
Property-based tests for user interface responsiveness.

This module tests the responsiveness properties of the Chess Video Analyzer UI,
including progress indicators, user interaction handling, and interface updates.
"""

import pytest
import tkinter as tk
from tkinter import ttk
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings
import tempfile
import os

from chess_video_analyzer.ui.main_interface import MainInterface, ProgressCallback, ProcessingResults
from chess_video_analyzer.core.data_models import (
    VideoMetadata, GameState, Move, Position, PieceType, Color, PieceKind, BoardState, CastlingRights
)


class TestUIResponsiveness:
    """Test suite for UI responsiveness properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a root window for testing (but don't display it)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during testing
        
        # Create the main interface
        self.interface = MainInterface(self.root)
        
        # Mock the video processor to avoid actual video processing
        self.interface.video_processor = Mock()
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # Window already destroyed
    
    @given(
        progress_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=5,
            max_size=20
        ),
        status_messages=st.lists(
            st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_property_ui_responsiveness(self, progress_values, status_messages):
        """
        **Feature: chess-video-analyzer, Property 18: User Interface Responsiveness**
        
        For any processing operation, the system should display progress indicators 
        and allow user interaction for review and correction.
        
        **Validates: Requirements 8.2, 8.3, 8.4, 8.5**
        """
        # Ensure we have matching numbers of progress values and messages
        min_length = min(len(progress_values), len(status_messages))
        progress_values = progress_values[:min_length]
        status_messages = status_messages[:min_length]
        
        # Test progress indicator responsiveness
        progress_callback = ProgressCallback(self.interface._update_progress)
        
        # Track UI updates
        ui_updates = []
        original_update_progress_ui = self.interface._update_progress_ui
        
        def mock_update_progress_ui(message, progress):
            ui_updates.append((message, progress))
            original_update_progress_ui(message, progress)
        
        self.interface._update_progress_ui = mock_update_progress_ui
        
        try:
            # Simulate progress updates
            for i, (progress, message) in enumerate(zip(progress_values, status_messages)):
                progress_callback.update(message, progress)
                
                # Process pending UI events
                self.root.update_idletasks()
                self.root.update()
                
                # Verify progress bar is updated
                expected_progress = progress * 100
                actual_progress = self.interface.progress_var.get()
                assert abs(actual_progress - expected_progress) < 0.01, \
                    f"Progress bar not updated correctly: expected {expected_progress}, got {actual_progress}"
                
                # Verify status message is updated
                actual_status = self.interface.status_var.get()
                assert actual_status == message, \
                    f"Status message not updated correctly: expected '{message}', got '{actual_status}'"
            
            # Verify all updates were processed
            assert len(ui_updates) == len(progress_values), \
                f"Not all UI updates were processed: expected {len(progress_values)}, got {len(ui_updates)}"
            
            # Verify UI remains responsive (buttons should be accessible)
            # Note: In actual processing, some buttons would be disabled, but UI should still respond
            assert self.interface.browse_button.winfo_exists()
            assert self.interface.process_button.winfo_exists()
            
        finally:
            # Restore original method
            self.interface._update_progress_ui = original_update_progress_ui
    
    @given(
        video_metadata=st.builds(
            VideoMetadata,
            duration=st.floats(min_value=1.0, max_value=3600.0),
            fps=st.floats(min_value=10.0, max_value=60.0),
            resolution=st.tuples(
                st.integers(min_value=320, max_value=1920),
                st.integers(min_value=240, max_value=1080)
            ),
            format=st.sampled_from(['.mp4', '.avi', '.mov'])
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_video_info_display_responsiveness(self, video_metadata):
        """
        Test that video information is displayed responsively when a video is loaded.
        
        **Validates: Requirements 8.2, 8.3**
        """
        # Mock successful video loading
        self.interface.video_processor.load_video.return_value = video_metadata
        self.interface.video_processor.is_loaded.return_value = True
        self.interface.video_processor.get_metadata.return_value = video_metadata
        
        # Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(suffix=video_metadata.format, delete=False) as f:
            temp_path = f.name
        
        try:
            # Load the video file
            self.interface._load_video_file(temp_path)
            
            # Process UI updates
            self.root.update_idletasks()
            self.root.update()
            
            # Verify file path is displayed
            assert self.interface.file_path_var.get() == temp_path
            
            # Verify video info is displayed
            video_info = self.interface.video_info_var.get()
            assert str(video_metadata.duration) in video_info or f"{video_metadata.duration:.1f}" in video_info
            assert str(video_metadata.fps) in video_info or f"{video_metadata.fps:.1f}" in video_info
            assert f"{video_metadata.resolution[0]}x{video_metadata.resolution[1]}" in video_info
            assert video_metadata.format in video_info
            
            # Verify process button is enabled
            assert str(self.interface.process_button['state']) != 'disabled'
            
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, FileNotFoundError):
                pass
    
    def test_error_display_responsiveness(self):
        """
        Test that error messages are displayed responsively.
        
        **Validates: Requirements 8.5**
        """
        # Mock video loading error
        error_message = "Test error message"
        self.interface.video_processor.load_video.side_effect = Exception(error_message)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock messagebox to capture error display
            with patch('chess_video_analyzer.ui.main_interface.messagebox') as mock_messagebox:
                # Attempt to load the video (should trigger error)
                self.interface._load_video_file(temp_path)
                
                # Process UI updates
                self.root.update_idletasks()
                self.root.update()
                
                # Verify error message was displayed
                mock_messagebox.showerror.assert_called_once()
                call_args = mock_messagebox.showerror.call_args
                assert "Error" in call_args[0][0]  # Title should contain "Error"
                assert error_message in call_args[0][1]  # Message should contain our error
                
                # Verify UI state is reset after error
                assert self.interface.file_path_var.get() == ""
                assert self.interface.video_info_var.get() == ""
                assert str(self.interface.process_button['state']) == 'disabled'
        
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, FileNotFoundError):
                pass
    
    @given(
        moves_count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=5, deadline=3000)
    def test_results_display_responsiveness(self, moves_count):
        """
        Test that processing results are displayed responsively.
        
        **Validates: Requirements 8.4**
        """
        # Create mock game state with moves
        mock_moves = []
        for i in range(moves_count):
            move = Move(
                from_square=Position(i % 8, (i // 8) % 8),
                to_square=Position((i + 1) % 8, ((i + 1) // 8) % 8),
                piece=PieceType(
                    Color.WHITE if i % 2 == 0 else Color.BLACK,
                    PieceKind.PAWN
                )
            )
            mock_moves.append(move)
        
        mock_board_state = BoardState(squares={}, timestamp=0.0, confidence=0.95)
        mock_castling_rights = CastlingRights()
        mock_game_state = GameState(
            current_position=mock_board_state,
            move_history=mock_moves,
            castling_rights=mock_castling_rights
        )
        
        # Set up processing results
        self.interface.processing_results.game_state = mock_game_state
        self.interface.processing_results.pgn_content = "Mock PGN content"
        self.interface.processing_results.fen_sequence = ["fen1", "fen2", "fen3"]
        
        # Display results
        self.interface._display_results()
        
        # Process UI updates
        self.root.update_idletasks()
        self.root.update()
        
        # Verify moves are displayed
        moves_content = self.interface.moves_text.get(1.0, tk.END).strip()
        assert len(moves_content) > 0, "Moves should be displayed"
        
        # Verify PGN is displayed
        pgn_content = self.interface.pgn_text.get(1.0, tk.END).strip()
        assert "Mock PGN content" in pgn_content, "PGN content should be displayed"
        
        # Verify FEN is displayed
        fen_content = self.interface.fen_text.get(1.0, tk.END).strip()
        assert "fen1" in fen_content, "FEN content should be displayed"
        
        # Verify export buttons are enabled
        assert str(self.interface.export_pgn_button['state']) != 'disabled'
        assert str(self.interface.export_fen_button['state']) != 'disabled'
    
    def test_processing_state_management(self):
        """
        Test that UI properly manages processing state transitions.
        
        **Validates: Requirements 8.2, 8.3**
        """
        # Set up a mock video file
        mock_metadata = VideoMetadata(
            duration=10.0,
            fps=30.0,
            resolution=(640, 480),
            format='.mp4'
        )
        
        self.interface.video_processor.load_video.return_value = mock_metadata
        self.interface.video_processor.is_loaded.return_value = True
        self.interface.video_processor.get_metadata.return_value = mock_metadata
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            # Load video
            self.interface._load_video_file(temp_path)
            self.root.update_idletasks()
            
            # Verify initial state
            assert str(self.interface.process_button['state']) != 'disabled'
            assert str(self.interface.cancel_button['state']) == 'disabled'
            assert not self.interface.is_processing
            
            # Mock the processing thread to avoid actual processing
            with patch('threading.Thread') as mock_thread:
                # Start processing
                self.interface._start_processing()
                self.root.update_idletasks()
                
                # Verify processing state
                assert self.interface.is_processing
                assert str(self.interface.process_button['state']) == 'disabled'
                assert str(self.interface.cancel_button['state']) != 'disabled'
                assert str(self.interface.browse_button['state']) == 'disabled'
                
                # Verify thread was started
                mock_thread.assert_called_once()
                
                # Simulate processing completion
                self.interface._processing_complete()
                self.root.update_idletasks()
                
                # Verify final state
                assert not self.interface.is_processing
                assert str(self.interface.process_button['state']) != 'disabled'
                assert str(self.interface.cancel_button['state']) == 'disabled'
                assert str(self.interface.browse_button['state']) != 'disabled'
        
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, FileNotFoundError):
                pass
    
    def test_cancel_processing_responsiveness(self):
        """
        Test that processing can be cancelled responsively.
        
        **Validates: Requirements 8.3**
        """
        # Set processing state
        self.interface.is_processing = True
        self.interface.process_button.config(state='disabled')
        self.interface.cancel_button.config(state='normal')
        self.interface.browse_button.config(state='disabled')
        
        # Cancel processing
        self.interface._cancel_processing()
        self.root.update_idletasks()
        
        # Verify state is reset
        assert not self.interface.is_processing
        assert str(self.interface.process_button['state']) != 'disabled'
        assert str(self.interface.cancel_button['state']) == 'disabled'
        assert str(self.interface.browse_button['state']) != 'disabled'
        
        # Verify status is updated
        assert "cancelled" in self.interface.status_var.get().lower()
        assert self.interface.progress_var.get() == 0
        assert self.interface.time_var.get() == ""
    
    @given(
        export_type=st.sampled_from(['pgn', 'fen'])
    )
    @settings(max_examples=5, deadline=2000)
    def test_export_functionality_responsiveness(self, export_type):
        """
        Test that export functionality is responsive.
        
        **Validates: Requirements 8.4, 8.5**
        """
        # Set up mock results
        if export_type == 'pgn':
            self.interface.processing_results.pgn_content = "Mock PGN content"
        else:
            self.interface.processing_results.fen_sequence = ["fen1", "fen2"]
        
        # Mock file dialog and file operations
        with patch('chess_video_analyzer.ui.main_interface.filedialog') as mock_filedialog, \
             patch('builtins.open', create=True) as mock_open, \
             patch('chess_video_analyzer.ui.main_interface.messagebox') as mock_messagebox:
            
            # Mock file dialog to return a filename
            mock_filedialog.asksaveasfilename.return_value = f"test.{export_type}"
            
            # Mock file writing
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Test export
            if export_type == 'pgn':
                self.interface._export_pgn()
            else:
                self.interface._export_fen()
            
            # Process UI updates
            self.root.update_idletasks()
            
            # Verify file dialog was called
            mock_filedialog.asksaveasfilename.assert_called_once()
            
            # Verify file was written
            mock_open.assert_called_once()
            mock_file.write.assert_called()
            
            # Verify success message was shown
            mock_messagebox.showinfo.assert_called_once()
            success_call = mock_messagebox.showinfo.call_args
            assert "successful" in success_call[0][0].lower() or "exported" in success_call[0][1].lower()
    
    def test_ui_thread_safety(self):
        """
        Test that UI updates from background threads are handled safely.
        
        **Validates: Requirements 8.2, 8.3**
        """
        # Track UI updates
        ui_updates = []
        original_after = self.root.after
        
        def mock_after(delay, func, *args):
            ui_updates.append((delay, func, args))
            return original_after(delay, func, *args)
        
        self.root.after = mock_after
        
        try:
            # Create progress callback
            progress_callback = ProgressCallback(self.interface._update_progress)
            
            # Simulate updates from background thread
            def background_updates():
                for i in range(5):
                    progress_callback.update(f"Step {i}", i / 4.0)
                    time.sleep(0.01)  # Small delay to simulate work
            
            # Run background updates
            thread = threading.Thread(target=background_updates, daemon=True)
            thread.start()
            thread.join(timeout=1.0)  # Wait for completion with timeout
            
            # Process any pending UI updates
            for _ in range(10):  # Process multiple update cycles
                self.root.update_idletasks()
                self.root.update()
                time.sleep(0.01)
            
            # Verify UI updates were scheduled properly
            assert len(ui_updates) >= 5, f"Expected at least 5 UI updates, got {len(ui_updates)}"
            
            # Verify all updates used delay=0 (immediate scheduling)
            for delay, func, args in ui_updates:
                assert delay == 0, f"UI update should be immediate, got delay={delay}"
        
        finally:
            # Restore original method
            self.root.after = original_after


class TestProgressCallback:
    """Test suite for ProgressCallback functionality."""
    
    def test_progress_callback_creation(self):
        """Test that ProgressCallback can be created and used."""
        updates = []
        
        def mock_callback(message, progress):
            updates.append((message, progress))
        
        callback = ProgressCallback(mock_callback)
        
        # Test callback usage
        callback.update("Test message", 0.5)
        
        assert len(updates) == 1
        assert updates[0] == ("Test message", 0.5)
    
    @given(
        messages=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
        progress_values=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10)
    )
    def test_progress_callback_multiple_updates(self, messages, progress_values):
        """Test ProgressCallback with multiple updates."""
        updates = []
        
        def mock_callback(message, progress):
            updates.append((message, progress))
        
        callback = ProgressCallback(mock_callback)
        
        # Ensure we have matching numbers of messages and progress values
        min_length = min(len(messages), len(progress_values))
        
        for i in range(min_length):
            callback.update(messages[i], progress_values[i])
        
        assert len(updates) == min_length
        for i in range(min_length):
            assert updates[i] == (messages[i], progress_values[i])


class TestProcessingResults:
    """Test suite for ProcessingResults container."""
    
    def test_processing_results_initialization(self):
        """Test that ProcessingResults initializes with correct defaults."""
        results = ProcessingResults()
        
        assert results.game_state is None
        assert results.video_metadata is None
        assert results.game_metadata is None
        assert results.pgn_content is None
        assert results.fen_sequence is None
        assert results.processing_time is None
        assert results.errors == []
        assert results.warnings == []
    
    def test_processing_results_data_storage(self):
        """Test that ProcessingResults can store various data types."""
        results = ProcessingResults()
        
        # Test storing different types of data
        mock_game_state = Mock()
        mock_video_metadata = Mock()
        mock_game_metadata = Mock()
        
        results.game_state = mock_game_state
        results.video_metadata = mock_video_metadata
        results.game_metadata = mock_game_metadata
        results.pgn_content = "Test PGN"
        results.fen_sequence = ["fen1", "fen2"]
        results.processing_time = 10.5
        results.errors = ["error1", "error2"]
        results.warnings = ["warning1"]
        
        # Verify all data is stored correctly
        assert results.game_state == mock_game_state
        assert results.video_metadata == mock_video_metadata
        assert results.game_metadata == mock_game_metadata
        assert results.pgn_content == "Test PGN"
        assert results.fen_sequence == ["fen1", "fen2"]
        assert results.processing_time == 10.5
        assert results.errors == ["error1", "error2"]
        assert results.warnings == ["warning1"]
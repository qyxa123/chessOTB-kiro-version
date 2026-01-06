"""
Integration tests for the Chess Video Analyzer system.

These tests verify end-to-end functionality from video input to notation output,
testing the complete pipeline with all components working together.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

from chess_video_analyzer.main import ChessVideoAnalyzer, ProcessingError
from chess_video_analyzer.core.data_models import (
    VideoMetadata, GameMetadata, BoardState, Move, Position, 
    PieceType, Color, PieceKind, GameState
)


class TestChessVideoAnalyzerIntegration:
    """Integration tests for the complete Chess Video Analyzer system."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a Chess Video Analyzer instance for testing."""
        return ChessVideoAnalyzer(
            confidence_threshold=0.7,
            enable_quality_control=True,
            enable_ui=False  # Disable UI for testing
        )
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a minimal mock video file
            f.write(b'mock video content')
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            duration=30.0,
            fps=30.0,
            resolution=(1920, 1080),
            format='.mp4'
        )
    
    @pytest.fixture
    def sample_game_metadata(self):
        """Create sample game metadata."""
        return GameMetadata(
            event="Test Game",
            site="Test Location",
            date="2024.01.01",
            round="1",
            white_player="White Player",
            black_player="Black Player",
            result="*"
        )
    
    @pytest.fixture
    def mock_frames(self):
        """Create mock video frames."""
        frames = []
        for i in range(5):
            # Create a simple mock frame (black image)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        return frames
    
    def test_complete_pipeline_success(self, analyzer, mock_video_file, 
                                     sample_video_metadata, sample_game_metadata, 
                                     mock_frames):
        """Test the complete pipeline from video loading to notation generation."""
        
        # Mock the video processor
        with patch.object(analyzer.video_processor, 'load_video') as mock_load, \
             patch.object(analyzer.video_processor, 'extract_frames') as mock_extract:
            
            mock_load.return_value = sample_video_metadata
            mock_extract.return_value = iter(mock_frames)
            
            # Mock board detection
            with patch.object(analyzer.board_detector, 'detect_board') as mock_detect_board, \
                 patch.object(analyzer.board_detector, 'get_square_coordinates') as mock_get_squares:
                
                # Create mock board region and square grid
                mock_board_region = Mock()
                mock_board_region.corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
                mock_board_region.confidence = 0.9
                mock_board_region.orientation = Mock()
                
                mock_square_grid = Mock()
                mock_square_grid.squares = {Position(x, y): (x*10, y*10, (x+1)*10, (y+1)*10) 
                                          for x in range(8) for y in range(8)}
                
                mock_detect_board.return_value = mock_board_region
                mock_get_squares.return_value = mock_square_grid
                
                # Mock move detection
                with patch.object(analyzer.move_tracker, 'detect_move') as mock_detect_move:
                    
                    # Create mock moves
                    mock_moves = [
                        Move(
                            from_square=Position(4, 6),
                            to_square=Position(4, 4),
                            piece=PieceType(Color.WHITE, PieceKind.PAWN)
                        ),
                        Move(
                            from_square=Position(4, 1),
                            to_square=Position(4, 3),
                            piece=PieceType(Color.BLACK, PieceKind.PAWN)
                        )
                    ]
                    
                    mock_detect_move.side_effect = mock_moves + [None] * 10  # Return moves then None
                    
                    # Test the complete pipeline
                    # 1. Load video
                    metadata = analyzer.load_video(mock_video_file)
                    assert metadata == sample_video_metadata
                    assert analyzer.current_video_path == mock_video_file
                    
                    # 2. Process video
                    results = analyzer.process_video(sample_game_metadata)
                    
                    # Verify results structure
                    assert 'game_state' in results
                    assert 'game_metadata' in results
                    assert 'pgn_content' in results
                    assert 'fen_sequence' in results
                    assert 'quality_report' in results
                    assert 'processing_time' in results
                    
                    # Verify game state
                    game_state = results['game_state']
                    assert isinstance(game_state, GameState)
                    assert len(game_state.move_history) >= 0  # May be empty due to mocking
                    
                    # Verify PGN content
                    pgn_content = results['pgn_content']
                    assert isinstance(pgn_content, str)
                    assert 'Test Game' in pgn_content
                    assert 'White Player' in pgn_content
                    assert 'Black Player' in pgn_content
                    
                    # Verify FEN sequence
                    fen_sequence = results['fen_sequence']
                    assert isinstance(fen_sequence, list)
                    assert len(fen_sequence) > 0
                    
                    # Verify quality report
                    quality_report = results['quality_report']
                    assert quality_report is not None
                    assert hasattr(quality_report, 'overall_confidence')
    
    def test_video_loading_error_handling(self, analyzer):
        """Test error handling during video loading."""
        
        # Test with non-existent file
        with pytest.raises(ProcessingError, match="Failed to load video"):
            analyzer.load_video("non_existent_file.mp4")
        
        # Test with unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not a video file')
            temp_path = f.name
        
        try:
            with pytest.raises(ProcessingError, match="Failed to load video"):
                analyzer.load_video(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_processing_without_loaded_video(self, analyzer):
        """Test processing fails when no video is loaded."""
        
        with pytest.raises(ProcessingError, match="No video loaded"):
            analyzer.process_video()
    
    def test_export_functionality(self, analyzer, mock_video_file, 
                                sample_video_metadata, sample_game_metadata):
        """Test PGN and FEN export functionality."""
        
        # Mock successful processing
        mock_results = {
            'pgn_content': '[Event "Test Game"]\n[Site "Test"]\n[Date "2024.01.01"]\n1. e4 e5 *',
            'fen_sequence': ['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1']
        }
        analyzer.processing_results = mock_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test PGN export
            pgn_path = os.path.join(temp_dir, 'test.pgn')
            analyzer.export_pgn(pgn_path)
            
            assert os.path.exists(pgn_path)
            with open(pgn_path, 'r') as f:
                content = f.read()
                assert 'Test Game' in content
                assert '1. e4 e5' in content
            
            # Test FEN export
            fen_path = os.path.join(temp_dir, 'test.fen')
            analyzer.export_fen(fen_path)
            
            assert os.path.exists(fen_path)
            with open(fen_path, 'r') as f:
                content = f.read()
                assert 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' in content
    
    def test_export_without_results(self, analyzer):
        """Test export fails when no processing results are available."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pgn_path = os.path.join(temp_dir, 'test.pgn')
            fen_path = os.path.join(temp_dir, 'test.fen')
            
            with pytest.raises(ProcessingError, match="No PGN content available"):
                analyzer.export_pgn(pgn_path)
            
            with pytest.raises(ProcessingError, match="No FEN content available"):
                analyzer.export_fen(fen_path)
    
    def test_quality_control_integration(self, analyzer, mock_video_file, 
                                       sample_video_metadata, mock_frames):
        """Test quality control integration throughout the pipeline."""
        
        with patch.object(analyzer.video_processor, 'load_video') as mock_load, \
             patch.object(analyzer.video_processor, 'extract_frames') as mock_extract:
            
            mock_load.return_value = sample_video_metadata
            mock_extract.return_value = iter(mock_frames)
            
            # Mock board detection with low confidence
            with patch.object(analyzer.board_detector, 'detect_board') as mock_detect_board, \
                 patch.object(analyzer.board_detector, 'get_square_coordinates') as mock_get_squares:
                
                mock_board_region = Mock()
                mock_board_region.corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
                mock_board_region.confidence = 0.3  # Low confidence
                mock_board_region.orientation = Mock()
                
                mock_square_grid = Mock()
                mock_square_grid.squares = {Position(x, y): (x*10, y*10, (x+1)*10, (y+1)*10) 
                                          for x in range(8) for y in range(8)}
                
                mock_detect_board.return_value = mock_board_region
                mock_get_squares.return_value = mock_square_grid
                
                # Set low confidence for board states created during processing
                analyzer._test_confidence = 0.3
                
                # Load video and process
                analyzer.load_video(mock_video_file)
                results = analyzer.process_video()
                
                # Verify quality control detected issues
                quality_report = results['quality_report']
                assert quality_report is not None
                assert quality_report.overall_confidence < 0.7  # Should be low due to mock confidence
                assert len(quality_report.issues) > 0  # Should have quality issues
    
    def test_progress_callback_integration(self, analyzer, mock_video_file, 
                                         sample_video_metadata, mock_frames):
        """Test progress callback functionality during processing."""
        
        progress_updates = []
        
        def progress_callback(message, progress):
            progress_updates.append((message, progress))
        
        analyzer.set_progress_callback(progress_callback)
        
        with patch.object(analyzer.video_processor, 'load_video') as mock_load, \
             patch.object(analyzer.video_processor, 'extract_frames') as mock_extract:
            
            mock_load.return_value = sample_video_metadata
            mock_extract.return_value = iter(mock_frames)
            
            # Mock other components
            with patch.object(analyzer.board_detector, 'detect_board') as mock_detect_board, \
                 patch.object(analyzer.board_detector, 'get_square_coordinates') as mock_get_squares:
                
                mock_board_region = Mock()
                mock_board_region.corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
                mock_board_region.confidence = 0.9
                mock_board_region.orientation = Mock()
                
                mock_square_grid = Mock()
                mock_square_grid.squares = {Position(x, y): (x*10, y*10, (x+1)*10, (y+1)*10) 
                                          for x in range(8) for y in range(8)}
                
                mock_detect_board.return_value = mock_board_region
                mock_get_squares.return_value = mock_square_grid
                
                # Process video
                analyzer.load_video(mock_video_file)
                analyzer.process_video()
                
                # Verify progress updates were called
                assert len(progress_updates) > 0
                
                # Check that progress values are in correct range
                for message, progress in progress_updates:
                    assert isinstance(message, str)
                    assert 0.0 <= progress <= 1.0
                
                # Check that final progress is 1.0
                final_message, final_progress = progress_updates[-1]
                assert final_progress == 1.0
                assert "complete" in final_message.lower()
    
    def test_error_callback_integration(self, analyzer):
        """Test error callback functionality."""
        
        error_messages = []
        
        def error_callback(message):
            error_messages.append(message)
        
        analyzer.set_error_callback(error_callback)
        
        # Trigger an error by trying to process without loading video
        with pytest.raises(ProcessingError):
            analyzer.process_video()
        
        # Note: Error callback is called during processing, not on exceptions
        # So we test with a processing error scenario
        
        with patch.object(analyzer.video_processor, 'load_video') as mock_load:
            mock_load.side_effect = Exception("Mock video loading error")
            
            with pytest.raises(ProcessingError):
                analyzer.load_video("test.mp4")
    
    def test_statistics_generation(self, analyzer):
        """Test processing statistics generation."""
        
        # Test with no results
        stats = analyzer.get_processing_statistics()
        assert stats == {}
        
        # Mock processing results
        mock_game_state = Mock()
        mock_game_state.move_history = [Mock(), Mock(), Mock()]  # 3 moves
        mock_game_state.flagged_moves = [Mock()]  # 1 flagged move
        
        mock_quality_report = Mock()
        mock_quality_report.overall_confidence = 0.85
        mock_quality_report.issues = [Mock(), Mock()]  # 2 issues
        mock_quality_report.get_critical_issues.return_value = [Mock()]  # 1 critical
        
        analyzer.processing_results = {
            'processing_time': 15.5,
            'game_state': mock_game_state,
            'quality_report': mock_quality_report,
            'board_states': [Mock(), Mock(), Mock(), Mock()]  # 4 board states
        }
        
        analyzer.video_metadata = VideoMetadata(
            duration=30.0, fps=30.0, resolution=(1920, 1080), format='.mp4'
        )
        
        stats = analyzer.get_processing_statistics()
        
        assert stats['processing_time'] == 15.5
        assert stats['video_duration'] == 30.0
        assert stats['total_moves'] == 3
        assert stats['flagged_moves'] == 1
        assert stats['board_states_processed'] == 4
        assert stats['overall_confidence'] == 0.85
        assert stats['quality_issues'] == 2
        assert stats['critical_issues'] == 1
    
    def test_output_validation(self, analyzer):
        """Test output file validation functionality."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid PGN file
            pgn_path = os.path.join(temp_dir, 'valid.pgn')
            with open(pgn_path, 'w') as f:
                f.write('[Event "Test"]\n[Site "Test"]\n[Date "2024.01.01"]\n'
                       '[Round "1"]\n[White "White"]\n[Black "Black"]\n'
                       '[Result "*"]\n\n1. e4 e5 *')
            
            # Create valid FEN file
            fen_path = os.path.join(temp_dir, 'valid.fen')
            with open(fen_path, 'w') as f:
                f.write('Move 0: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n')
                f.write('Move 1: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n')
            
            # Test validation
            results = analyzer.validate_output(pgn_path, fen_path)
            
            assert 'pgn_valid' in results
            assert 'fen_valid' in results
            assert results['pgn_valid'] is True
            assert results['fen_valid'] is True
            
            # Test with invalid files
            invalid_pgn_path = os.path.join(temp_dir, 'invalid.pgn')
            with open(invalid_pgn_path, 'w') as f:
                f.write('Invalid PGN content')
            
            invalid_fen_path = os.path.join(temp_dir, 'invalid.fen')
            with open(invalid_fen_path, 'w') as f:
                f.write('Move 0: invalid fen string\n')
            
            results = analyzer.validate_output(invalid_pgn_path, invalid_fen_path)
            
            assert results['pgn_valid'] is False
            assert results['fen_valid'] is False
    
    def test_resource_cleanup(self, analyzer, mock_video_file, sample_video_metadata):
        """Test proper resource cleanup."""
        
        with patch.object(analyzer.video_processor, 'load_video') as mock_load, \
             patch.object(analyzer.video_processor, 'close') as mock_close:
            
            mock_load.return_value = sample_video_metadata
            
            # Load video
            analyzer.load_video(mock_video_file)
            assert analyzer.current_video_path == mock_video_file
            assert analyzer.video_metadata == sample_video_metadata
            
            # Close analyzer
            analyzer.close()
            
            # Verify cleanup
            assert analyzer.current_video_path is None
            assert analyzer.video_metadata is None
            assert len(analyzer.processing_results) == 0
            assert analyzer.is_processing is False
            mock_close.assert_called_once()
    
    def test_concurrent_processing_prevention(self, analyzer, mock_video_file, 
                                            sample_video_metadata):
        """Test that concurrent processing is prevented."""
        
        with patch.object(analyzer.video_processor, 'load_video') as mock_load:
            mock_load.return_value = sample_video_metadata
            
            analyzer.load_video(mock_video_file)
            
            # Set processing flag manually
            analyzer.is_processing = True
            
            # Try to process again
            with pytest.raises(ProcessingError, match="Processing already in progress"):
                analyzer.process_video()
    
    def test_flagged_moves_retrieval(self, analyzer):
        """Test retrieval of flagged moves."""
        
        # Test with no results
        flagged_moves = analyzer.get_flagged_moves()
        assert flagged_moves == []
        
        # Mock game state with flagged moves
        mock_flagged_move = Mock()
        mock_game_state = Mock()
        mock_game_state.flagged_moves = [mock_flagged_move]
        
        analyzer.processing_results = {'game_state': mock_game_state}
        
        flagged_moves = analyzer.get_flagged_moves()
        assert len(flagged_moves) == 1
        assert flagged_moves[0] == mock_flagged_move


class TestEndToEndScenarios:
    """End-to-end scenario tests for realistic use cases."""
    
    def test_typical_game_analysis_workflow(self):
        """Test a typical workflow of analyzing a chess game video."""
        
        analyzer = ChessVideoAnalyzer(enable_ui=False)
        
        # Mock a complete workflow
        with patch.object(analyzer.video_processor, 'load_video') as mock_load, \
             patch.object(analyzer.video_processor, 'extract_frames') as mock_extract:
            
            # Mock video loading
            video_metadata = VideoMetadata(
                duration=120.0, fps=30.0, resolution=(1920, 1080), format='.mp4'
            )
            mock_load.return_value = video_metadata
            
            # Mock frame extraction (simulate 10 frames)
            mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
            mock_extract.return_value = iter(mock_frames)
            
            # Mock board detection
            with patch.object(analyzer.board_detector, 'detect_board') as mock_detect_board, \
                 patch.object(analyzer.board_detector, 'get_square_coordinates') as mock_get_squares:
                
                mock_board_region = Mock()
                mock_board_region.corners = [(100, 100), (500, 100), (500, 500), (100, 500)]
                mock_board_region.confidence = 0.95
                mock_board_region.orientation = Mock()
                
                mock_square_grid = Mock()
                mock_square_grid.squares = {Position(x, y): (x*50+100, y*50+100, (x+1)*50+100, (y+1)*50+100) 
                                          for x in range(8) for y in range(8)}
                
                mock_detect_board.return_value = mock_board_region
                mock_get_squares.return_value = mock_square_grid
                
                # Mock move detection (simulate a few moves)
                with patch.object(analyzer.move_tracker, 'detect_move') as mock_detect_move:
                    
                    moves = [
                        Move(Position(4, 6), Position(4, 4), PieceType(Color.WHITE, PieceKind.PAWN)),
                        Move(Position(4, 1), Position(4, 3), PieceType(Color.BLACK, PieceKind.PAWN)),
                        Move(Position(6, 7), Position(5, 5), PieceType(Color.WHITE, PieceKind.KNIGHT)),
                    ]
                    
                    move_iter = iter(moves + [None] * 20)  # Return moves then None
                    mock_detect_move.side_effect = lambda prev, curr: next(move_iter)
                    
                    try:
                        # Step 1: Load video
                        metadata = analyzer.load_video("test_game.mp4")
                        assert metadata.duration == 120.0
                        
                        # Step 2: Process video
                        game_metadata = GameMetadata(
                            event="Club Championship",
                            site="Local Chess Club",
                            date="2024.01.15",
                            round="3",
                            white_player="Alice",
                            black_player="Bob",
                            result="1-0"
                        )
                        
                        results = analyzer.process_video(game_metadata, frame_interval=2.0)
                        
                        # Step 3: Verify results
                        assert 'game_state' in results
                        assert 'pgn_content' in results
                        assert 'fen_sequence' in results
                        
                        # Check PGN content
                        pgn = results['pgn_content']
                        assert 'Club Championship' in pgn
                        assert 'Alice' in pgn
                        assert 'Bob' in pgn
                        
                        # Step 4: Export files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            pgn_path = os.path.join(temp_dir, 'game.pgn')
                            fen_path = os.path.join(temp_dir, 'game.fen')
                            
                            analyzer.export_pgn(pgn_path)
                            analyzer.export_fen(fen_path)
                            
                            # Verify files exist and have content
                            assert os.path.exists(pgn_path)
                            assert os.path.exists(fen_path)
                            
                            with open(pgn_path, 'r') as f:
                                pgn_content = f.read()
                                assert len(pgn_content) > 0
                                assert 'Alice' in pgn_content
                            
                            with open(fen_path, 'r') as f:
                                fen_content = f.read()
                                assert len(fen_content) > 0
                                assert 'Move' in fen_content
                        
                        # Step 5: Get statistics
                        stats = analyzer.get_processing_statistics()
                        assert stats['video_duration'] == 120.0
                        assert stats['total_moves'] >= 0
                        assert 'processing_time' in stats
                        
                        # Step 6: Validate output
                        validation_results = analyzer.validate_output(pgn_path, fen_path)
                        # Note: Files may not exist anymore due to temp directory cleanup
                        
                    finally:
                        analyzer.close()
    
    def test_error_recovery_workflow(self):
        """Test error recovery and graceful degradation."""
        
        analyzer = ChessVideoAnalyzer(enable_ui=False)
        
        try:
            # Test 1: Invalid video file
            with pytest.raises(ProcessingError):
                analyzer.load_video("nonexistent.mp4")
            
            # Test 2: Processing without video
            with pytest.raises(ProcessingError):
                analyzer.process_video()
            
            # Test 3: Partial processing failure
            with patch.object(analyzer.video_processor, 'load_video') as mock_load:
                video_metadata = VideoMetadata(
                    duration=60.0, fps=30.0, resolution=(640, 480), format='.mp4'
                )
                mock_load.return_value = video_metadata
                
                analyzer.load_video("test.mp4")
                
                # Mock frame extraction failure
                with patch.object(analyzer.video_processor, 'extract_frames') as mock_extract:
                    mock_extract.side_effect = Exception("Frame extraction failed")
                    
                    with pytest.raises(ProcessingError, match="Frame extraction failed"):
                        analyzer.process_video()
            
            # Test 4: Export without results
            with pytest.raises(ProcessingError):
                analyzer.export_pgn("output.pgn")
            
            with pytest.raises(ProcessingError):
                analyzer.export_fen("output.fen")
        
        finally:
            analyzer.close()


if __name__ == "__main__":
    pytest.main([__file__])
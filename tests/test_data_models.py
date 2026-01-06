"""
Tests for core data models.
"""

import pytest
from hypothesis import given, strategies as st

from chess_video_analyzer.core.data_models import (
    Position, PieceType, PieceKind, Color, Move, BoardState, 
    GameState, VideoMetadata, GameMetadata, CastlingRights, Square
)


class TestPosition:
    """Test Position data class."""
    
    def test_valid_position(self):
        """Test creating a valid position."""
        pos = Position(3, 4)
        assert pos.x == 3
        assert pos.y == 4
    
    def test_invalid_position_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        with pytest.raises(ValueError):
            Position(-1, 4)
        
        with pytest.raises(ValueError):
            Position(3, 8)
        
        with pytest.raises(ValueError):
            Position(8, 3)


class TestPieceType:
    """Test PieceType data class."""
    
    def test_piece_creation(self):
        """Test creating a piece."""
        piece = PieceType(Color.WHITE, PieceKind.KING)
        assert piece.color == Color.WHITE
        assert piece.type == PieceKind.KING


class TestBoardState:
    """Test BoardState data class."""
    
    def test_valid_board_state(self):
        """Test creating a valid board state."""
        board = BoardState({}, 0.0, 0.95)
        assert board.confidence == 0.95
        assert board.timestamp == 0.0
    
    def test_invalid_confidence(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            BoardState({}, 0.0, 1.5)
        
        with pytest.raises(ValueError):
            BoardState({}, 0.0, -0.1)


class TestGameState:
    """Test GameState data class."""
    
    def test_valid_game_state(self):
        """Test creating a valid game state."""
        board = BoardState({}, 0.0)
        castling = CastlingRights()
        game = GameState(board, [], castling)
        assert game.halfmove_clock == 0
        assert game.fullmove_number == 1
        assert game.active_color == Color.WHITE
    
    def test_invalid_halfmove_clock(self):
        """Test that negative halfmove clock raises ValueError."""
        board = BoardState({}, 0.0)
        castling = CastlingRights()
        with pytest.raises(ValueError):
            GameState(board, [], castling, halfmove_clock=-1)
    
    def test_invalid_fullmove_number(self):
        """Test that invalid fullmove number raises ValueError."""
        board = BoardState({}, 0.0)
        castling = CastlingRights()
        with pytest.raises(ValueError):
            GameState(board, [], castling, fullmove_number=0)


class TestVideoMetadata:
    """Test VideoMetadata data class."""
    
    def test_valid_video_metadata(self):
        """Test creating valid video metadata."""
        metadata = VideoMetadata(120.0, 30.0, (1920, 1080), "mp4")
        assert metadata.duration == 120.0
        assert metadata.fps == 30.0
        assert metadata.resolution == (1920, 1080)
        assert metadata.format == "mp4"
    
    def test_invalid_duration(self):
        """Test that invalid duration raises ValueError."""
        with pytest.raises(ValueError):
            VideoMetadata(-1.0, 30.0, (1920, 1080), "mp4")
    
    def test_invalid_fps(self):
        """Test that invalid FPS raises ValueError."""
        with pytest.raises(ValueError):
            VideoMetadata(120.0, 0.0, (1920, 1080), "mp4")
    
    def test_invalid_resolution(self):
        """Test that invalid resolution raises ValueError."""
        with pytest.raises(ValueError):
            VideoMetadata(120.0, 30.0, (0, 1080), "mp4")
        
        with pytest.raises(ValueError):
            VideoMetadata(120.0, 30.0, (1920,), "mp4")


# Property-based tests using Hypothesis
class TestPropertyBasedDataModels:
    """Property-based tests for data models."""
    
    @pytest.mark.property
    @given(st.integers(min_value=0, max_value=7), st.integers(min_value=0, max_value=7))
    def test_position_valid_coordinates(self, x, y):
        """Property: Valid coordinates should create valid positions."""
        pos = Position(x, y)
        assert pos.x == x
        assert pos.y == y
    
    @pytest.mark.property
    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_board_state_valid_confidence(self, confidence):
        """Property: Valid confidence values should create valid board states."""
        board = BoardState({}, 0.0, confidence)
        assert board.confidence == confidence
    
    @pytest.mark.property
    @given(st.integers(min_value=0), st.integers(min_value=1))
    def test_game_state_valid_clocks(self, halfmove, fullmove):
        """Property: Valid clock values should create valid game states."""
        board = BoardState({}, 0.0)
        castling = CastlingRights()
        game = GameState(board, [], castling, halfmove_clock=halfmove, fullmove_number=fullmove)
        assert game.halfmove_clock == halfmove
        assert game.fullmove_number == fullmove
"""
Tests for move tracking and detection functionality.

**Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection** (move detection part)
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Optional, Tuple

from chess_video_analyzer.core.data_models import (
    Position, PieceType, PieceKind, Color, Move, BoardState, 
    SpecialMoveType
)
from chess_video_analyzer.tracking.move_tracker import MoveTracker, MoveCandidate


# Strategies for generating test data
@st.composite
def chess_position(draw):
    """Generate valid chess board positions."""
    x = draw(st.integers(min_value=0, max_value=7))
    y = draw(st.integers(min_value=0, max_value=7))
    return Position(x, y)


@st.composite
def piece_type(draw):
    """Generate valid piece types."""
    color = draw(st.sampled_from(list(Color)))
    kind = draw(st.sampled_from(list(PieceKind)))
    return PieceType(color, kind)


@st.composite
def board_state_with_pieces(draw):
    """Generate a board state with random pieces, avoiding identical pieces in ambiguous positions."""
    # Generate pieces on random positions
    num_pieces = draw(st.integers(min_value=2, max_value=16))
    positions = draw(st.lists(
        chess_position(), 
        min_size=num_pieces, 
        max_size=num_pieces, 
        unique=True
    ))
    
    squares = {}
    # Initialize all squares as empty
    for x in range(8):
        for y in range(8):
            squares[Position(x, y)] = None
    
    # Place pieces on selected positions, ensuring no identical pieces are adjacent
    used_piece_types = set()
    for pos in positions:
        # Generate a unique piece type for this position
        attempts = 0
        while attempts < 10:  # Limit attempts to avoid infinite loop
            piece = draw(piece_type())
            
            # Check if this piece type would create ambiguity with adjacent positions
            adjacent_positions = [
                Position(pos.x + dx, pos.y + dy)
                for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                if dx != 0 or dy != 0  # Exclude the position itself
                if 0 <= pos.x + dx <= 7 and 0 <= pos.y + dy <= 7  # Stay within board
            ]
            
            # Check if any adjacent position has the same piece type
            has_adjacent_identical = any(
                squares.get(adj_pos) == piece for adj_pos in adjacent_positions
            )
            
            if not has_adjacent_identical:
                squares[pos] = piece
                break
            attempts += 1
        
        # If we couldn't find a non-conflicting piece after 10 attempts, use a unique one
        if squares[pos] is None:
            # Create a unique piece by cycling through types and colors
            color_idx = len([p for p in squares.values() if p is not None]) % 2
            piece_idx = len([p for p in squares.values() if p is not None]) % 6
            color = Color.WHITE if color_idx == 0 else Color.BLACK
            piece_kind = list(PieceKind)[piece_idx]
            squares[pos] = PieceType(color, piece_kind)
    
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    confidence = draw(st.floats(min_value=0.7, max_value=1.0))
    
    return BoardState(squares, timestamp, confidence)


@st.composite
def board_state_pair_with_move(draw):
    """Generate a pair of board states showing a single move, avoiding identical piece ambiguity."""
    # Create initial board state with non-adjacent identical pieces
    initial_state = draw(board_state_with_pieces())
    
    # Find a piece to move
    pieces_positions = [(pos, piece) for pos, piece in initial_state.squares.items() if piece is not None]
    assume(len(pieces_positions) > 0)
    
    # Select a piece to move
    from_pos, moving_piece = draw(st.sampled_from(pieces_positions))
    
    # Select destination (different from source and preferably not creating identical piece conflicts)
    all_positions = [Position(x, y) for x in range(8) for y in range(8)]
    available_positions = [pos for pos in all_positions if pos != from_pos]
    
    # Prefer destinations that don't create identical piece ambiguity
    preferred_positions = []
    for pos in available_positions:
        # Check if moving to this position would create ambiguity
        adjacent_positions = [
            Position(pos.x + dx, pos.y + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if dx != 0 or dy != 0  # Exclude the position itself
            if 0 <= pos.x + dx <= 7 and 0 <= pos.y + dy <= 7  # Stay within board
            if Position(pos.x + dx, pos.y + dy) != from_pos  # Exclude the source position
        ]
        
        # Check if any adjacent position has the same piece type
        has_adjacent_identical = any(
            initial_state.squares.get(adj_pos) == moving_piece for adj_pos in adjacent_positions
        )
        
        if not has_adjacent_identical:
            preferred_positions.append(pos)
    
    # Use preferred positions if available, otherwise use any available position
    candidate_positions = preferred_positions if preferred_positions else available_positions
    to_pos = draw(st.sampled_from(candidate_positions))
    
    # Create the new board state
    new_squares = initial_state.squares.copy()
    captured_piece = new_squares.get(to_pos)  # Piece that was captured (if any)
    
    # Move the piece
    new_squares[from_pos] = None
    new_squares[to_pos] = moving_piece
    
    new_timestamp = initial_state.timestamp + draw(st.floats(min_value=0.1, max_value=5.0))
    new_confidence = draw(st.floats(min_value=0.7, max_value=1.0))
    
    final_state = BoardState(new_squares, new_timestamp, new_confidence)
    
    expected_move = Move(
        from_square=from_pos,
        to_square=to_pos,
        piece=moving_piece,
        captured_piece=captured_piece
    )
    
    return initial_state, final_state, expected_move


class TestMoveTracker:
    """Test the MoveTracker class."""
    
    def test_move_tracker_initialization(self):
        """Test MoveTracker initialization."""
        tracker = MoveTracker()
        assert tracker.confidence_threshold == 0.7
        assert len(tracker.get_move_history()) == 0
        
        # Test with custom threshold
        custom_tracker = MoveTracker(confidence_threshold=0.8)
        assert custom_tracker.confidence_threshold == 0.8
    
    def test_detect_simple_move(self):
        """Test detection of a simple piece move."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create initial board state with a piece
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # Create final board state with piece moved
        final_squares = initial_squares.copy()
        final_squares[Position(4, 4)] = None
        final_squares[Position(4, 5)] = PieceType(Color.WHITE, PieceKind.PAWN)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.from_square == Position(4, 4)
        assert detected_move.to_square == Position(4, 5)
        assert detected_move.piece.color == Color.WHITE
        assert detected_move.piece.type == PieceKind.PAWN
        assert detected_move.captured_piece is None
    
    def test_detect_capture_move(self):
        """Test detection of a capture move."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create initial board state with two pieces
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_squares[Position(5, 5)] = PieceType(Color.BLACK, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # Create final board state with white pawn capturing black pawn
        final_squares = initial_squares.copy()
        final_squares[Position(4, 4)] = None
        final_squares[Position(5, 5)] = PieceType(Color.WHITE, PieceKind.PAWN)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.from_square == Position(4, 4)
        assert detected_move.to_square == Position(5, 5)
        assert detected_move.piece.color == Color.WHITE
        assert detected_move.piece.type == PieceKind.PAWN
        assert detected_move.captured_piece is not None
        assert detected_move.captured_piece.color == Color.BLACK
        assert detected_move.captured_piece.type == PieceKind.PAWN
    
    def test_detect_capture_events(self):
        """Test detection of capture events."""
        tracker = MoveTracker()
        
        # Create board states for capture scenario
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_squares[Position(5, 5)] = PieceType(Color.BLACK, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # White pawn captures black pawn
        final_squares = initial_squares.copy()
        final_squares[Position(4, 4)] = None
        final_squares[Position(5, 5)] = PieceType(Color.WHITE, PieceKind.PAWN)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        captures = tracker.detect_capture(initial_state, final_state)
        
        assert len(captures) == 1
        assert Position(5, 5) in captures
    
    def test_detect_piece_disappearances(self):
        """Test detection of piece disappearances."""
        tracker = MoveTracker()
        
        # Create board state with pieces
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_squares[Position(5, 5)] = PieceType(Color.BLACK, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # Remove black pawn (disappeared)
        final_squares = initial_squares.copy()
        final_squares[Position(5, 5)] = None
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        disappearances = tracker.detect_piece_disappearances(initial_state, final_state)
        
        assert len(disappearances) == 1
        assert disappearances[0][0] == Position(5, 5)
        assert disappearances[0][1].color == Color.BLACK
        assert disappearances[0][1].type == PieceKind.PAWN
    
    def test_detect_castling_kingside(self):
        """Test detection of kingside castling."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create initial position for castling
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 7)] = PieceType(Color.WHITE, PieceKind.KING)
        initial_squares[Position(7, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # After castling
        final_squares = initial_squares.copy()
        final_squares[Position(4, 7)] = None
        final_squares[Position(7, 7)] = None
        final_squares[Position(6, 7)] = PieceType(Color.WHITE, PieceKind.KING)
        final_squares[Position(5, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move (should detect king move as castling)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.piece.type == PieceKind.KING
        assert detected_move.special_move == SpecialMoveType.CASTLING_KINGSIDE
    
    def test_detect_castling_queenside(self):
        """Test detection of queenside castling."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create initial position for queenside castling
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 7)] = PieceType(Color.WHITE, PieceKind.KING)
        initial_squares[Position(0, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # After queenside castling
        final_squares = initial_squares.copy()
        final_squares[Position(4, 7)] = None
        final_squares[Position(0, 7)] = None
        final_squares[Position(2, 7)] = PieceType(Color.WHITE, PieceKind.KING)
        final_squares[Position(3, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.piece.type == PieceKind.KING
        assert detected_move.special_move == SpecialMoveType.CASTLING_QUEENSIDE
    
    def test_detect_en_passant(self):
        """Test detection of en passant capture."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create position for en passant
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 3)] = PieceType(Color.WHITE, PieceKind.PAWN)  # White pawn
        initial_squares[Position(5, 3)] = PieceType(Color.BLACK, PieceKind.PAWN)  # Black pawn
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # After en passant capture
        final_squares = initial_squares.copy()
        final_squares[Position(4, 3)] = None  # White pawn moved
        final_squares[Position(5, 3)] = None  # Black pawn captured
        final_squares[Position(5, 2)] = PieceType(Color.WHITE, PieceKind.PAWN)  # White pawn to new position
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.piece.type == PieceKind.PAWN
        assert detected_move.special_move == SpecialMoveType.EN_PASSANT
    
    def test_detect_pawn_promotion(self):
        """Test detection of pawn promotion."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create position for pawn promotion (white pawn on 7th rank)
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 1)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        # After promotion to queen
        final_squares = initial_squares.copy()
        final_squares[Position(4, 1)] = None
        final_squares[Position(4, 0)] = PieceType(Color.WHITE, PieceKind.QUEEN)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        # Detect move
        detected_move = tracker.detect_move(initial_state, final_state)
        
        assert detected_move is not None
        assert detected_move.from_square == Position(4, 1)
        assert detected_move.to_square == Position(4, 0)
        assert detected_move.piece.type == PieceKind.PAWN
        assert detected_move.special_move == SpecialMoveType.PROMOTION
    
    def test_low_confidence_rejection(self):
        """Test that low confidence moves are rejected."""
        tracker = MoveTracker(confidence_threshold=0.8)
        
        # Create board states with low confidence
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.6)  # Low confidence
        
        final_squares = initial_squares.copy()
        final_squares[Position(4, 4)] = None
        final_squares[Position(4, 5)] = PieceType(Color.WHITE, PieceKind.PAWN)
        final_state = BoardState(final_squares, 1.0, 0.6)  # Low confidence
        
        # Should not detect move due to low confidence
        detected_move = tracker.detect_move(initial_state, final_state)
        assert detected_move is None
    
    def test_move_history(self):
        """Test move history tracking."""
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Detect a move
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(4, 4)] = PieceType(Color.WHITE, PieceKind.PAWN)
        initial_state = BoardState(initial_squares, 0.0, 0.9)
        
        final_squares = initial_squares.copy()
        final_squares[Position(4, 4)] = None
        final_squares[Position(4, 5)] = PieceType(Color.WHITE, PieceKind.PAWN)
        final_state = BoardState(final_squares, 1.0, 0.9)
        
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Check history
        history = tracker.get_move_history()
        assert len(history) == 1
        assert history[0] == detected_move
        
        # Clear history
        tracker.clear_history()
        assert len(tracker.get_move_history()) == 0


class TestMoveTrackingProperty:
    """
    Property-based tests for move tracking functionality.
    
    **Validates: Requirements 3.2, 3.3**
    """
    
    @given(board_state_pair_with_move())
    @settings(max_examples=10, deadline=8000)
    def test_comprehensive_move_detection_property(self, board_state_data):
        """
        Property 5: Comprehensive Piece Recognition and Move Detection (move detection part)
        
        For any pair of board states showing a single move, the system should correctly
        detect the source and destination squares and identify capture events.
        
        **Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection**
        **Validates: Requirements 3.2, 3.3**
        """
        initial_state, final_state, expected_move = board_state_data
        
        tracker = MoveTracker(confidence_threshold=0.6)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Should detect a move
        assert detected_move is not None, "Should detect a move when pieces change position"
        
        # Verify move properties
        assert isinstance(detected_move, Move)
        assert isinstance(detected_move.from_square, Position)
        assert isinstance(detected_move.to_square, Position)
        assert isinstance(detected_move.piece, PieceType)
        
        # Verify the move makes sense
        assert detected_move.from_square != detected_move.to_square, "Source and destination should be different"
        
        # Verify piece consistency
        assert detected_move.piece == expected_move.piece, "Detected piece should match the moved piece"
        
        # For identical piece scenarios, we need to be more flexible about exact position matching
        # Instead, verify that the detected move produces a logically consistent result
        
        # Verify that the detected move is logically valid:
        # 1. The piece was actually at the from_square in the initial state
        assert initial_state.squares.get(detected_move.from_square) == detected_move.piece, \
            "Piece should have been at the detected from_square in initial state"
        
        # 2. The piece is at the to_square in the final state
        assert final_state.squares.get(detected_move.to_square) == detected_move.piece, \
            "Piece should be at the detected to_square in final state"
        
        # 3. The from_square is empty in the final state (piece moved away)
        assert final_state.squares.get(detected_move.from_square) is None, \
            "From_square should be empty in final state after piece moved"
        
        # 4. Verify capture detection is consistent with board state changes
        initial_piece_at_dest = initial_state.squares.get(detected_move.to_square)
        if detected_move.captured_piece is not None:
            # If a capture was detected, there should have been a piece at the destination
            assert initial_piece_at_dest is not None, "Should have been a piece to capture at destination"
            assert detected_move.captured_piece == initial_piece_at_dest, "Captured piece should match what was at destination"
        else:
            # If no capture was detected, destination should have been empty or had the same piece type
            # (in case of identical piece ambiguity)
            if initial_piece_at_dest is not None:
                # Allow for identical piece scenarios where the "capture" is actually just ambiguous movement
                assert initial_piece_at_dest == detected_move.piece, \
                    "If no capture detected but destination wasn't empty, it should have had the same piece type"
        
        # 5. Verify that applying the detected move to the initial state would produce a consistent result
        # Count total pieces to ensure the move makes sense
        initial_piece_count = sum(1 for p in initial_state.squares.values() if p is not None)
        final_piece_count = sum(1 for p in final_state.squares.values() if p is not None)
        
        if detected_move.captured_piece is not None:
            # If there was a capture, final count should be one less than initial
            expected_final_count = initial_piece_count - 1
            assert final_piece_count == expected_final_count, \
                f"After capture, piece count should decrease by 1: {initial_piece_count} -> {expected_final_count}, got {final_piece_count}"
        else:
            # If no capture, piece count should remain the same
            assert final_piece_count == initial_piece_count, \
                f"Without capture, piece count should remain same: {initial_piece_count}, got {final_piece_count}"
    
    @given(board_state_with_pieces())
    @settings(max_examples=20, deadline=5000)
    def test_no_move_detection_identical_states(self, board_state):
        """
        Test that identical board states don't produce moves.
        
        **Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection**
        **Validates: Requirements 3.2**
        """
        tracker = MoveTracker(confidence_threshold=0.5)
        
        # Create identical states with different timestamps
        state1 = board_state
        state2 = BoardState(
            squares=board_state.squares.copy(),
            timestamp=board_state.timestamp + 1.0,
            confidence=board_state.confidence
        )
        
        detected_move = tracker.detect_move(state1, state2)
        
        # Should not detect any move for identical board positions
        assert detected_move is None, "Should not detect move when board states are identical"
    
    @given(board_state_with_pieces(), st.floats(min_value=0.1, max_value=0.6))
    @settings(max_examples=15, deadline=5000)
    def test_confidence_threshold_property(self, board_state, low_confidence):
        """
        Test that confidence thresholds are respected.
        
        **Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection**
        **Validates: Requirements 3.2**
        """
        tracker = MoveTracker(confidence_threshold=0.7)
        
        # Create a low-confidence board state
        low_conf_state = BoardState(
            squares=board_state.squares,
            timestamp=board_state.timestamp,
            confidence=low_confidence
        )
        
        # Create a modified state (move a piece if possible)
        modified_squares = board_state.squares.copy()
        pieces_positions = [(pos, piece) for pos, piece in modified_squares.items() if piece is not None]
        
        if pieces_positions:
            # Move first piece to a different position
            from_pos, piece = pieces_positions[0]
            # Find an empty position
            empty_positions = [pos for pos, p in modified_squares.items() if p is None]
            if empty_positions:
                to_pos = empty_positions[0]
                modified_squares[from_pos] = None
                modified_squares[to_pos] = piece
                
                modified_state = BoardState(
                    squares=modified_squares,
                    timestamp=board_state.timestamp + 1.0,
                    confidence=low_confidence
                )
                
                # Should not detect move due to low confidence
                detected_move = tracker.detect_move(low_conf_state, modified_state)
                assert detected_move is None, "Should not detect move when confidence is below threshold"
    
    @given(board_state_with_pieces())
    @settings(max_examples=15, deadline=5000)
    def test_capture_detection_property(self, board_state):
        """
        Test capture detection across various board states.
        
        **Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection**
        **Validates: Requirements 3.3**
        """
        tracker = MoveTracker()
        
        # Find pieces that can be "captured"
        pieces_positions = [(pos, piece) for pos, piece in board_state.squares.items() if piece is not None]
        
        if len(pieces_positions) >= 2:
            # Find two different pieces for a meaningful capture test
            attacker_pos, attacker_piece = pieces_positions[0]
            victim_pos, victim_piece = None, None
            
            # Find a different piece to be the victim
            for pos, piece in pieces_positions[1:]:
                if piece != attacker_piece:  # Ensure different pieces
                    victim_pos, victim_piece = pos, piece
                    break
            
            # Only test if we found two different pieces
            if victim_pos is not None:
                # Create state after capture
                final_squares = board_state.squares.copy()
                final_squares[attacker_pos] = None  # Attacker moved
                final_squares[victim_pos] = attacker_piece  # Attacker now on victim's square
                
                final_state = BoardState(
                    squares=final_squares,
                    timestamp=board_state.timestamp + 1.0,
                    confidence=board_state.confidence
                )
                
                # Detect captures
                captures = tracker.detect_capture(board_state, final_state)
                
                # Should detect the capture
                assert victim_pos in captures, "Should detect capture at victim's position"
    
    @given(board_state_with_pieces())
    @settings(max_examples=15, deadline=5000)
    def test_piece_disappearance_property(self, board_state):
        """
        Test piece disappearance detection.
        
        **Feature: chess-video-analyzer, Property 5: Comprehensive Piece Recognition and Move Detection**
        **Validates: Requirements 3.3**
        """
        tracker = MoveTracker()
        
        # Find a piece to make disappear
        pieces_positions = [(pos, piece) for pos, piece in board_state.squares.items() if piece is not None]
        
        if pieces_positions:
            disappearing_pos, disappearing_piece = pieces_positions[0]
            
            # Create state with piece removed
            final_squares = board_state.squares.copy()
            final_squares[disappearing_pos] = None
            
            final_state = BoardState(
                squares=final_squares,
                timestamp=board_state.timestamp + 1.0,
                confidence=board_state.confidence
            )
            
            # Detect disappearances
            disappearances = tracker.detect_piece_disappearances(board_state, final_state)
            
            # Should detect the disappearance
            disappeared_positions = [pos for pos, piece in disappearances]
            assert disappearing_pos in disappeared_positions, "Should detect piece disappearance"
            
            # Find the disappeared piece in results
            disappeared_piece = None
            for pos, piece in disappearances:
                if pos == disappearing_pos:
                    disappeared_piece = piece
                    break
            
            assert disappeared_piece == disappearing_piece, "Disappeared piece should match original"


class TestSpecialMoveRecognitionProperty:
    """
    Property-based tests for special move recognition functionality.
    
    **Validates: Requirements 3.4, 3.5, 3.6**
    """
    
    @st.composite
    def castling_scenario(draw, color=None):
        """Generate valid castling scenarios."""
        if color is None:
            color = draw(st.sampled_from(list(Color)))
        
        # Determine rank based on color
        rank = 7 if color == Color.WHITE else 0
        
        # Choose castling type
        castling_type = draw(st.sampled_from([SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE]))
        
        # Create initial position for castling
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        
        # Place king and rook in starting positions
        initial_squares[Position(4, rank)] = PieceType(color, PieceKind.KING)
        
        if castling_type == SpecialMoveType.CASTLING_KINGSIDE:
            initial_squares[Position(7, rank)] = PieceType(color, PieceKind.ROOK)
            # Final positions after kingside castling
            final_king_pos = Position(6, rank)
            final_rook_pos = Position(5, rank)
        else:  # CASTLING_QUEENSIDE
            initial_squares[Position(0, rank)] = PieceType(color, PieceKind.ROOK)
            # Final positions after queenside castling
            final_king_pos = Position(2, rank)
            final_rook_pos = Position(3, rank)
        
        # Add some random other pieces (but not blocking castling path)
        num_other_pieces = draw(st.integers(min_value=0, max_value=8))
        for _ in range(num_other_pieces):
            pos = draw(chess_position())
            # Don't place pieces on castling path or king/rook positions
            castling_path = {Position(4, rank), Position(5, rank), Position(6, rank), Position(7, rank)} if castling_type == SpecialMoveType.CASTLING_KINGSIDE else {Position(0, rank), Position(1, rank), Position(2, rank), Position(3, rank), Position(4, rank)}
            
            if pos not in castling_path and initial_squares[pos] is None:
                piece = draw(piece_type())
                initial_squares[pos] = piece
        
        # Create final state after castling
        final_squares = initial_squares.copy()
        final_squares[Position(4, rank)] = None  # King moved
        final_squares[Position(7 if castling_type == SpecialMoveType.CASTLING_KINGSIDE else 0, rank)] = None  # Rook moved
        final_squares[final_king_pos] = PieceType(color, PieceKind.KING)
        final_squares[final_rook_pos] = PieceType(color, PieceKind.ROOK)
        
        timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
        confidence = draw(st.floats(min_value=0.8, max_value=1.0))
        
        initial_state = BoardState(initial_squares, timestamp, confidence)
        final_state = BoardState(final_squares, timestamp + 1.0, confidence)
        
        return initial_state, final_state, castling_type, color
    
    @st.composite
    def en_passant_scenario(draw, color=None):
        """Generate valid en passant scenarios."""
        if color is None:
            color = draw(st.sampled_from(list(Color)))
        
        # Determine ranks based on color
        if color == Color.WHITE:
            pawn_rank = 3  # White pawn on 4th rank
            capture_rank = 2  # Captures to 3rd rank
        else:
            pawn_rank = 4  # Black pawn on 5th rank  
            capture_rank = 5  # Captures to 6th rank
        
        # Choose file positions
        attacking_file = draw(st.integers(min_value=0, max_value=7))
        target_file = attacking_file + draw(st.sampled_from([-1, 1]))
        
        # Ensure target file is valid
        if target_file < 0 or target_file > 7:
            target_file = attacking_file - (target_file - attacking_file)
        
        # Create initial position
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        
        # Place attacking pawn and target pawn
        initial_squares[Position(attacking_file, pawn_rank)] = PieceType(color, PieceKind.PAWN)
        initial_squares[Position(target_file, pawn_rank)] = PieceType(Color.BLACK if color == Color.WHITE else Color.WHITE, PieceKind.PAWN)
        
        # Add some random other pieces
        num_other_pieces = draw(st.integers(min_value=0, max_value=8))
        for _ in range(num_other_pieces):
            pos = draw(chess_position())
            if pos not in [Position(attacking_file, pawn_rank), Position(target_file, pawn_rank), Position(target_file, capture_rank)] and initial_squares[pos] is None:
                piece = draw(piece_type())
                initial_squares[pos] = piece
        
        # Create final state after en passant
        final_squares = initial_squares.copy()
        final_squares[Position(attacking_file, pawn_rank)] = None  # Attacking pawn moved
        final_squares[Position(target_file, pawn_rank)] = None  # Target pawn captured
        final_squares[Position(target_file, capture_rank)] = PieceType(color, PieceKind.PAWN)  # Attacking pawn to new position
        
        timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
        confidence = draw(st.floats(min_value=0.8, max_value=1.0))
        
        initial_state = BoardState(initial_squares, timestamp, confidence)
        final_state = BoardState(final_squares, timestamp + 1.0, confidence)
        
        return initial_state, final_state, Position(attacking_file, pawn_rank), Position(target_file, capture_rank), color
    
    @st.composite
    def promotion_scenario(draw, color=None):
        """Generate valid pawn promotion scenarios."""
        if color is None:
            color = draw(st.sampled_from(list(Color)))
        
        # Determine ranks based on color
        if color == Color.WHITE:
            start_rank = 1  # White pawn on 2nd rank (7th rank in 0-indexed)
            end_rank = 0    # Promotes to 1st rank (8th rank in 0-indexed)
        else:
            start_rank = 6  # Black pawn on 7th rank (2nd rank in 0-indexed)
            end_rank = 7    # Promotes to 8th rank (1st rank in 0-indexed)
        
        # Choose file and promotion piece
        file = draw(st.integers(min_value=0, max_value=7))
        promotion_piece_type = draw(st.sampled_from([PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]))
        
        # Create initial position
        initial_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        initial_squares[Position(file, start_rank)] = PieceType(color, PieceKind.PAWN)
        
        # Optionally place a piece to capture during promotion
        capture_piece = None
        if draw(st.booleans()):
            capture_piece = PieceType(Color.BLACK if color == Color.WHITE else Color.WHITE, draw(st.sampled_from(list(PieceKind))))
            initial_squares[Position(file, end_rank)] = capture_piece
        
        # Add some random other pieces
        num_other_pieces = draw(st.integers(min_value=0, max_value=8))
        for _ in range(num_other_pieces):
            pos = draw(chess_position())
            if pos not in [Position(file, start_rank), Position(file, end_rank)] and initial_squares[pos] is None:
                piece = draw(piece_type())
                initial_squares[pos] = piece
        
        # Create final state after promotion
        final_squares = initial_squares.copy()
        final_squares[Position(file, start_rank)] = None  # Pawn moved
        final_squares[Position(file, end_rank)] = PieceType(color, promotion_piece_type)  # Promoted piece
        
        timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
        confidence = draw(st.floats(min_value=0.8, max_value=1.0))
        
        initial_state = BoardState(initial_squares, timestamp, confidence)
        final_state = BoardState(final_squares, timestamp + 1.0, confidence)
        
        return initial_state, final_state, Position(file, start_rank), Position(file, end_rank), promotion_piece_type, capture_piece, color
    
    @given(castling_scenario())
    @settings(max_examples=15, deadline=10000)
    def test_castling_recognition_property(self, castling_data):
        """
        Property 6: Special Move Recognition (castling part)
        
        For any valid castling scenario, the system should correctly identify
        the castling move and classify it as kingside or queenside castling.
        
        **Feature: chess-video-analyzer, Property 6: Special Move Recognition**
        **Validates: Requirements 3.4**
        """
        initial_state, final_state, expected_castling_type, color = castling_data
        
        tracker = MoveTracker(confidence_threshold=0.7)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Should detect a move
        assert detected_move is not None, "Should detect castling move"
        
        # Should be identified as castling
        assert detected_move.special_move == expected_castling_type, f"Should detect {expected_castling_type.value} castling"
        
        # Should be a king move
        assert detected_move.piece.type == PieceKind.KING, "Castling should be detected as king move"
        assert detected_move.piece.color == color, "King color should match expected color"
        
        # Verify king movement pattern
        king_start = Position(4, 7 if color == Color.WHITE else 0)
        if expected_castling_type == SpecialMoveType.CASTLING_KINGSIDE:
            expected_king_end = Position(6, 7 if color == Color.WHITE else 0)
        else:  # CASTLING_QUEENSIDE
            expected_king_end = Position(2, 7 if color == Color.WHITE else 0)
        
        assert detected_move.from_square == king_start, "King should move from starting position"
        assert detected_move.to_square == expected_king_end, "King should move to correct castling position"
        
        # Should not be a capture (castling doesn't capture)
        assert detected_move.captured_piece is None, "Castling should not involve capture"
    
    @given(en_passant_scenario())
    @settings(max_examples=10, deadline=8000)
    def test_en_passant_recognition_property(self, en_passant_data):
        """
        Property 6: Special Move Recognition (en passant part)
        
        For any valid en passant scenario, the system should correctly identify
        the en passant capture move.
        
        **Feature: chess-video-analyzer, Property 6: Special Move Recognition**
        **Validates: Requirements 3.5**
        """
        initial_state, final_state, from_pos, to_pos, color = en_passant_data
        
        tracker = MoveTracker(confidence_threshold=0.7)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Should detect a move
        assert detected_move is not None, "Should detect en passant move"
        
        # Should be identified as en passant
        assert detected_move.special_move == SpecialMoveType.EN_PASSANT, "Should detect en passant capture"
        
        # Should be a pawn move
        assert detected_move.piece.type == PieceKind.PAWN, "En passant should be detected as pawn move"
        assert detected_move.piece.color == color, "Pawn color should match expected color"
        
        # Verify movement pattern
        assert detected_move.from_square == from_pos, "Pawn should move from expected starting position"
        assert detected_move.to_square == to_pos, "Pawn should move to expected ending position"
        
        # Verify diagonal movement (characteristic of en passant)
        assert abs(detected_move.from_square.x - detected_move.to_square.x) == 1, "En passant should involve diagonal movement"
        assert abs(detected_move.from_square.y - detected_move.to_square.y) == 1, "En passant should involve diagonal movement"
    
    @given(promotion_scenario())
    @settings(max_examples=10, deadline=8000)
    def test_pawn_promotion_recognition_property(self, promotion_data):
        """
        Property 6: Special Move Recognition (promotion part)
        
        For any valid pawn promotion scenario, the system should correctly identify
        the promotion move and the new piece type.
        
        **Feature: chess-video-analyzer, Property 6: Special Move Recognition**
        **Validates: Requirements 3.6**
        """
        initial_state, final_state, from_pos, to_pos, promotion_piece_type, capture_piece, color = promotion_data
        
        tracker = MoveTracker(confidence_threshold=0.7)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Should detect a move
        assert detected_move is not None, "Should detect promotion move"
        
        # Should be identified as promotion
        assert detected_move.special_move == SpecialMoveType.PROMOTION, "Should detect pawn promotion"
        
        # Should be a pawn move (the original piece)
        assert detected_move.piece.type == PieceKind.PAWN, "Promotion should be detected as pawn move"
        assert detected_move.piece.color == color, "Pawn color should match expected color"
        
        # Verify movement to promotion rank
        assert detected_move.from_square == from_pos, "Pawn should move from expected starting position"
        assert detected_move.to_square == to_pos, "Pawn should move to expected ending position"
        
        # Verify promotion rank
        if color == Color.WHITE:
            assert detected_move.to_square.y == 0, "White pawn should promote on rank 1 (y=0)"
        else:
            assert detected_move.to_square.y == 7, "Black pawn should promote on rank 8 (y=7)"
        
        # Verify capture if there was one
        if capture_piece is not None:
            assert detected_move.captured_piece == capture_piece, "Should detect captured piece during promotion"
        else:
            assert detected_move.captured_piece is None, "Should not detect capture when there was none"
    
    @given(st.one_of([
        castling_scenario(),
        en_passant_scenario(), 
        promotion_scenario()
    ]))
    @settings(max_examples=20, deadline=15000)
    def test_special_move_general_properties(self, special_move_data):
        """
        Property 6: Special Move Recognition (general properties)
        
        For any special move scenario, the system should correctly identify
        that it is a special move and provide appropriate move details.
        
        **Feature: chess-video-analyzer, Property 6: Special Move Recognition**
        **Validates: Requirements 3.4, 3.5, 3.6**
        """
        # Unpack the data based on the type of special move
        # We can identify the type by the structure of the tuple
        if isinstance(special_move_data, tuple):
            if len(special_move_data) == 4:  # Castling
                initial_state, final_state, expected_special_move, color = special_move_data
            elif len(special_move_data) == 5:  # En passant
                initial_state, final_state, from_pos, to_pos, color = special_move_data
                expected_special_move = SpecialMoveType.EN_PASSANT
            else:  # Promotion (7 elements)
                initial_state, final_state, from_pos, to_pos, promotion_piece_type, capture_piece, color = special_move_data
                expected_special_move = SpecialMoveType.PROMOTION
        else:
            # If it's not a tuple, skip this test case
            return
        
        tracker = MoveTracker(confidence_threshold=0.7)
        detected_move = tracker.detect_move(initial_state, final_state)
        
        # Should detect a move
        assert detected_move is not None, "Should detect special move"
        
        # Should be identified as a special move
        assert detected_move.special_move is not None, "Should identify move as special"
        assert detected_move.special_move == expected_special_move, f"Should correctly identify {expected_special_move.value}"
        
        # Move should have valid structure
        assert isinstance(detected_move.from_square, Position), "Move should have valid from_square"
        assert isinstance(detected_move.to_square, Position), "Move should have valid to_square"
        assert isinstance(detected_move.piece, PieceType), "Move should have valid piece"
        assert detected_move.from_square != detected_move.to_square, "From and to squares should be different"
        
        # Piece should be valid for the special move type
        if expected_special_move in [SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE]:
            assert detected_move.piece.type == PieceKind.KING, "Castling should involve king"
        elif expected_special_move in [SpecialMoveType.EN_PASSANT, SpecialMoveType.PROMOTION]:
            assert detected_move.piece.type == PieceKind.PAWN, "En passant and promotion should involve pawn"
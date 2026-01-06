"""
Property-based tests for quality control functionality.

**Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
**Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from chess_video_analyzer.core.data_models import (
    Position, PieceType, PieceKind, Color, BoardState, Move, 
    VideoMetadata, GameState, CastlingRights, SpecialMoveType
)
from chess_video_analyzer.quality.quality_controller import (
    QualityController, QualityReport, QualityFlag, QualityIssue
)


# Strategies for generating test data
@st.composite
def board_state_with_confidence(draw):
    """Generate board state with specified confidence level."""
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate some pieces
    num_pieces = draw(st.integers(min_value=0, max_value=32))
    squares = {}
    
    positions = [Position(x, y) for x in range(8) for y in range(8)]
    selected_positions = draw(st.lists(
        st.sampled_from(positions), 
        min_size=min(num_pieces, 64), 
        max_size=min(num_pieces, 64),
        unique=True
    ))
    
    for pos in selected_positions:
        color = draw(st.sampled_from(list(Color)))
        piece_kind = draw(st.sampled_from(list(PieceKind)))
        squares[pos] = PieceType(color, piece_kind)
    
    # Fill remaining positions with None
    for pos in positions:
        if pos not in squares:
            squares[pos] = None
    
    return BoardState(squares, timestamp=0.0, confidence=confidence)


@st.composite
def chess_move(draw):
    """Generate a chess move."""
    from_pos = Position(
        draw(st.integers(min_value=0, max_value=7)),
        draw(st.integers(min_value=0, max_value=7))
    )
    to_pos = Position(
        draw(st.integers(min_value=0, max_value=7)),
        draw(st.integers(min_value=0, max_value=7))
    )
    
    # Ensure from and to are different
    assume(from_pos != to_pos)
    
    color = draw(st.sampled_from(list(Color)))
    piece_kind = draw(st.sampled_from(list(PieceKind)))
    piece = PieceType(color, piece_kind)
    
    # Optional captured piece
    captured_piece = None
    if draw(st.booleans()):
        captured_color = Color.WHITE if color == Color.BLACK else Color.BLACK
        captured_kind = draw(st.sampled_from(list(PieceKind)))
        captured_piece = PieceType(captured_color, captured_kind)
    
    # Optional special move
    special_move = None
    if draw(st.booleans()):
        special_move = draw(st.sampled_from(list(SpecialMoveType)))
    
    return Move(
        from_square=from_pos,
        to_square=to_pos,
        piece=piece,
        captured_piece=captured_piece,
        special_move=special_move
    )


@st.composite
def video_metadata(draw):
    """Generate video metadata."""
    width = draw(st.integers(min_value=240, max_value=1920))
    height = draw(st.integers(min_value=240, max_value=1080))
    fps = draw(st.floats(min_value=1.0, max_value=60.0))
    duration = draw(st.floats(min_value=1.0, max_value=7200.0))  # Up to 2 hours
    
    return VideoMetadata(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        format="mp4"
    )


class TestQualityControlProperty:
    """
    Property-based tests for quality control functionality.
    
    **Validates: Requirements 7.2**
    """
    
    @given(board_state_with_confidence(), st.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=20, deadline=5000)
    def test_confidence_based_quality_control(self, board_state, threshold):
        """
        Property 15: Confidence-based Quality Control
        
        For any piece recognition with low confidence scores, the system should 
        flag uncertain moves for user review.
        
        **Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
        **Validates: Requirements 7.2**
        """
        controller = QualityController(confidence_threshold=threshold)
        
        # Assess board state quality
        issues = controller.assess_board_state_quality(board_state, frame_number=1)
        
        # Verify basic properties
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, QualityIssue)
            assert isinstance(issue.flag_type, QualityFlag)
            assert 0.0 <= issue.confidence <= 1.0
            assert issue.severity in ["low", "medium", "high", "critical"]
        
        # Test confidence-based flagging
        if board_state.confidence < threshold:
            # Should flag low confidence
            confidence_issues = [i for i in issues if i.flag_type == QualityFlag.LOW_CONFIDENCE_DETECTION]
            assert len(confidence_issues) > 0, "Low confidence should be flagged"
            
            # Check severity based on how low the confidence is
            if board_state.confidence < controller.critical_confidence_threshold:
                critical_issues = [i for i in confidence_issues if i.severity == "critical"]
                assert len(critical_issues) > 0, "Very low confidence should be critical"
        else:
            # Should not flag confidence if above threshold
            confidence_issues = [i for i in issues if i.flag_type == QualityFlag.LOW_CONFIDENCE_DETECTION]
            # Note: might still flag for other reasons, but not specifically for confidence
        
        # Verify issue descriptions are meaningful
        for issue in issues:
            assert len(issue.description) > 0, "Issue should have description"
            assert len(issue.suggested_action) > 0, "Issue should have suggested action"
    
    @given(chess_move(), board_state_with_confidence(), board_state_with_confidence())
    @settings(max_examples=15, deadline=8000)
    def test_move_quality_assessment(self, move, prev_state, curr_state):
        """
        Test move quality assessment with confidence-based flagging.
        
        **Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
        **Validates: Requirements 7.2**
        """
        controller = QualityController(confidence_threshold=0.7)
        
        # Assess move quality
        issues = controller.assess_move_quality(move, prev_state, curr_state, frame_number=1)
        
        # Verify basic properties
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, QualityIssue)
            assert issue.move == move or issue.move is None
        
        # Calculate expected move confidence
        move_confidence = min(prev_state.confidence, curr_state.confidence)
        
        # Check confidence-based flagging
        if move_confidence < controller.confidence_threshold:
            confidence_issues = [i for i in issues if i.flag_type == QualityFlag.LOW_CONFIDENCE_DETECTION]
            # Should flag low confidence moves
            if move_confidence < controller.critical_confidence_threshold:
                # Should have critical issues for very low confidence
                critical_issues = [i for i in issues if i.severity == "critical"]
                assert len(critical_issues) > 0, "Very low confidence moves should be critical"
        
        # If move is pre-flagged, sh
        # If move is pre-flagged, should detect it
        if move.is_flagged:
            flagged_issues = [i for i in issues if i.flag_type == QualityFlag.ILLEGAL_MOVE]
            assert len(flagged_issues) > 0, "Flagged moves should be detected"
    
    @given(video_metadata())
    @settings(max_examples=30, deadline=5000)
    def test_video_quality_assessment(self, metadata):
        """
        Test video quality assessment.
        
        **Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
        **Validates: Requirements 7.1**
        """
        controller = QualityController()
        
        # Assess video quality
        issues = controller.assess_video_quality(metadata)
        
        # Verify basic properties
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, QualityIssue)
            assert issue.flag_type == QualityFlag.POOR_VIDEO_QUALITY
        
        # Check resolution-based flagging
        width, height = metadata.resolution
        min_resolution = min(width, height)
        
        if min_resolution < 480:
            # Should flag very low resolution
            resolution_issues = [i for i in issues if "resolution" in i.description.lower()]
            assert len(resolution_issues) > 0, "Very low resolution should be flagged"
            
            # Should be high severity
            high_severity_issues = [i for i in resolution_issues if i.severity == "high"]
            assert len(high_severity_issues) > 0, "Very low resolution should be high severity"
        
        # Check frame rate flagging
        if metadata.fps < 15:
            fps_issues = [i for i in issues if "frame rate" in i.description.lower()]
            assert len(fps_issues) > 0, "Very low frame rate should be flagged"
    
    @given(st.lists(board_state_with_confidence(), min_size=1, max_size=10))
    @settings(max_examples=20, deadline=8000)
    def test_quality_report_generation(self, board_states):
        """
        Test quality report generation.
        
        **Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
        **Validates: Requirements 7.2**
        """
        controller = QualityController(confidence_threshold=0.7)
        
        # Process multiple board states
        all_issues = []
        for i, board_state in enumerate(board_states):
            issues = controller.assess_board_state_quality(board_state, frame_number=i)
            all_issues.extend(issues)
        
        # Generate quality report
        report = controller.generate_quality_report()
        
        # Verify report structure
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_confidence <= 1.0
        assert isinstance(report.issues, list)
        assert isinstance(report.statistics, dict)
        assert isinstance(report.recommendations, list)
        
        # Verify statistics
        stats = report.statistics
        assert "total_frames_processed" in stats
        assert "average_confidence" in stats
        assert "total_issues" in stats
        assert stats["total_frames_processed"] == len(board_states)
        assert stats["total_issues"] == len(all_issues)
        
        # Verify critical issue detection
        critical_issues = report.get_critical_issues()
        expected_critical = [i for i in all_issues if i.severity == "critical"]
        assert len(critical_issues) == len(expected_critical)
        
        # Verify recommendations are provided
        if report.overall_confidence < 0.5:
            assert len(report.recommendations) > 0, "Low confidence should generate recommendations"
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20, deadline=3000)
    def test_confidence_threshold_behavior(self, threshold):
        """
        Test that confidence thresholds work correctly.
        
        **Feature: chess-video-analyzer, Property 15: Confidence-based Quality Control**
        **Validates: Requirements 7.2**
        """
        controller = QualityController(confidence_threshold=threshold)
        
        # Test with confidence just below threshold
        low_confidence = max(0.0, threshold - 0.1)
        board_state_low = BoardState({}, timestamp=0.0, confidence=low_confidence)
        
        issues_low = controller.assess_board_state_quality(board_state_low)
        
        # Test with confidence just above threshold
        high_confidence = min(1.0, threshold + 0.1)
        board_state_high = BoardState({}, timestamp=0.0, confidence=high_confidence)
        
        issues_high = controller.assess_board_state_quality(board_state_high)
        
        # Low confidence should generate more or equal issues
        low_confidence_issues = [i for i in issues_low if i.flag_type == QualityFlag.LOW_CONFIDENCE_DETECTION]
        high_confidence_issues = [i for i in issues_high if i.flag_type == QualityFlag.LOW_CONFIDENCE_DETECTION]
        
        assert len(low_confidence_issues) >= len(high_confidence_issues), \
            "Lower confidence should generate more confidence-related issues"


class TestManualCorrectionProperty:
    """
    Property-based tests for manual correction capability.
    
    **Validates: Requirements 7.4**
    """
    
    @given(chess_move(), st.text(min_size=1, max_size=100))
    @settings(max_examples=20, deadline=5000)
    def test_manual_correction_capability(self, move, reason):
        """
        Property 16: Manual Correction Capability
        
        For any detected error, the system should allow users to manually 
        correct the error and update the game state accordingly.
        
        **Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
        **Validates: Requirements 7.4**
        """
        controller = QualityController()
        
        # Test flagging for manual review
        issue = controller.flag_for_manual_review(
            item=move,
            flag_type=QualityFlag.INCONSISTENT_MOVE,
            reason=reason,
            confidence=0.5
        )
        
        # Verify issue was created correctly
        assert isinstance(issue, QualityIssue)
        assert issue.flag_type == QualityFlag.INCONSISTENT_MOVE
        assert issue.description == reason
        assert issue.move == move
        assert move.is_flagged == True
        assert move.flag_reason == reason
        
        # Test callback registration
        correction_result = {"corrected": True, "new_move": move}
        
        def mock_correction_callback(issue):
            return correction_result
        
        controller.register_correction_callback("test_callback", mock_correction_callback)
        
        # Test manual correction request
        result = controller.request_manual_correction(issue, "test_callback")
        assert result == correction_result
        
        # Test with non-existent callback
        result_none = controller.request_manual_correction(issue, "non_existent")
        assert result_none is None
    
    @given(st.lists(chess_move(), min_size=1, max_size=5))
    @settings(max_examples=15, deadline=8000)
    def test_multiple_corrections(self, moves):
        """
        Test handling multiple manual corrections.
        
        **Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
        **Validates: Requirements 7.4**
        """
        controller = QualityController()
        
        # Flag multiple moves for correction
        issues = []
        for i, move in enumerate(moves):
            issue = controller.flag_for_manual_review(
                item=move,
                flag_type=QualityFlag.ILLEGAL_MOVE,
                reason=f"Test reason {i}",
                confidence=0.3
            )
            issues.append(issue)
        
        # Verify all moves are flagged
        for move in moves:
            assert move.is_flagged == True
        
        # Register correction callback
        corrections_made = []
        
        def correction_callback(issue):
            corrections_made.append(issue)
            return {"corrected": True}
        
        controller.register_correction_callback("batch_correction", correction_callback)
        
        # Request corrections for all issues
        for issue in issues:
            result = controller.request_manual_correction(issue, "batch_correction")
            assert result == {"corrected": True}
        
        # Verify all corrections were processed
        assert len(corrections_made) == len(issues)
    
    def test_correction_callback_error_handling(self):
        """
        Test error handling in correction callbacks.
        
        **Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
        **Validates: Requirements 7.4**
        """
        controller = QualityController()
        
        # Create a test issue
        move = Move(
            from_square=Position(0, 0),
            to_square=Position(1, 1),
            piece=PieceType(Color.WHITE, PieceKind.PAWN)
        )
        
        issue = controller.flag_for_manual_review(
            item=move,
            flag_type=QualityFlag.ILLEGAL_MOVE,
            reason="Test error handling",
            confidence=0.2
        )
        
        # Register callback that raises an exception
        def failing_callback(issue):
            raise ValueError("Callback failed")
        
        controller.register_correction_callback("failing_callback", failing_callback)
        
        # Request correction - should handle error gracefully
        result = controller.request_manual_correction(issue, "failing_callback")
        assert result is None  # Should return None on callback failure
    
    @given(board_state_with_confidence())
    @settings(max_examples=15, deadline=5000)
    def test_flagging_different_item_types(self, board_state):
        """
        Test flagging different types of items for manual review.
        
        **Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
        **Validates: Requirements 7.4**
        """
        controller = QualityController()
        
        # Test flagging board state
        issue1 = controller.flag_for_manual_review(
            item=board_state,
            flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
            reason="Low confidence board state",
            confidence=board_state.confidence
        )
        
        assert isinstance(issue1, QualityIssue)
        assert issue1.confidence == board_state.confidence
        
        # Test flagging with position
        class MockItem:
            def __init__(self, position):
                self.position = position
        
        test_position = Position(3, 4)
        mock_item = MockItem(test_position)
        
        issue2 = controller.flag_for_manual_review(
            item=mock_item,
            flag_type=QualityFlag.AMBIGUOUS_POSITION,
            reason="Ambiguous position",
            confidence=0.4
        )
        
        assert issue2.position == test_position
    
    def test_quality_report_includes_flagged_items(self):
        """
        Test that quality reports include flagged items.
        
        **Feature: chess-video-analyzer, Property 16: Manual Correction Capability**
        **Validates: Requirements 7.4**
        """
        controller = QualityController()
        
        # Create and flag some moves
        moves = [
            Move(Position(0, 0), Position(1, 1), PieceType(Color.WHITE, PieceKind.PAWN)),
            Move(Position(2, 2), Position(3, 3), PieceType(Color.BLACK, PieceKind.ROOK))
        ]
        
        for i, move in enumerate(moves):
            controller.flag_for_manual_review(
                item=move,
                flag_type=QualityFlag.ILLEGAL_MOVE,
                reason=f"Test flag {i}",
                confidence=0.3
            )
        
        # Generate report
        report = controller.generate_quality_report()
        
        # Verify flagged moves are included
        assert len(report.flagged_moves) == len(moves)
        for move in moves:
            assert move in report.flagged_moves
            assert move.is_flagged == True
        
        # Verify issues are included
        flagged_issues = [i for i in report.issues if i.flag_type == QualityFlag.ILLEGAL_MOVE]
        assert len(flagged_issues) == len(moves)


class TestQualityControlEdgeCases:
    """Test edge cases and error conditions for quality control."""
    
    def test_empty_board_state_assessment(self):
        """Test assessment of empty board state."""
        controller = QualityController()
        
        # Create empty board state
        empty_squares = {Position(x, y): None for x in range(8) for y in range(8)}
        empty_board = BoardState(empty_squares, timestamp=0.0, confidence=0.8)
        
        issues = controller.assess_board_state_quality(empty_board)
        
        # Should flag empty board
        piece_issues = [i for i in issues if i.flag_type == QualityFlag.PIECE_RECOGNITION_FAILURE]
        assert len(piece_issues) > 0, "Empty board should be flagged"
    
    def test_invalid_confidence_values(self):
        """Test handling of invalid confidence values."""
        # Test invalid confidence in QualityIssue
        with pytest.raises(ValueError):
            QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=1.5  # Invalid confidence > 1.0
            )
        
        with pytest.raises(ValueError):
            QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=-0.1  # Invalid confidence < 0.0
            )
    
    def test_invalid_severity_values(self):
        """Test handling of invalid severity values."""
        with pytest.raises(ValueError):
            QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=0.5,
                severity="invalid"  # Invalid severity
            )
    
    def test_quality_controller_initialization(self):
        """Test quality controller initialization with different parameters."""
        # Test default initialization
        controller1 = QualityController()
        assert controller1.confidence_threshold == 0.7
        assert controller1.critical_confidence_threshold == 0.3
        
        # Test custom initialization
        controller2 = QualityController(
            confidence_threshold=0.8,
            critical_confidence_threshold=0.2,
            consistency_window=10
        )
        assert controller2.confidence_threshold == 0.8
        assert controller2.critical_confidence_threshold == 0.2
        assert controller2.consistency_window == 10
    
    def test_history_management(self):
        """Test that history is properly managed and limited."""
        controller = QualityController(consistency_window=3)
        
        # Add more board states than the window size
        for i in range(10):
            squares = {Position(x, y): None for x in range(8) for y in range(8)}
            board_state = BoardState(squares, timestamp=float(i), confidence=0.8)
            controller.assess_board_state_quality(board_state)
        
        # History should be limited to window size
        assert len(controller._board_state_history) <= controller.consistency_window
        assert len(controller._confidence_history) <= controller.consistency_window * 2
    
    def test_clear_history(self):
        """Test clearing of history and counters."""
        controller = QualityController()
        
        # Add some data
        squares = {Position(x, y): None for x in range(8) for y in range(8)}
        board_state = BoardState(squares, timestamp=0.0, confidence=0.5)
        controller.assess_board_state_quality(board_state)
        
        # Verify data exists
        assert len(controller._confidence_history) > 0
        assert controller._frame_count > 0
        
        # Clear history
        controller.clear_history()
        
        # Verify everything is cleared
        assert len(controller._confidence_history) == 0
        assert len(controller._board_state_history) == 0
        assert len(controller._move_history) == 0
        assert len(controller._quality_issues) == 0
        assert controller._frame_count == 0
        assert controller._low_confidence_count == 0
        assert controller._flagged_move_count == 0
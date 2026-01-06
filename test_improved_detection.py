#!/usr/bin/env python3
"""
Test the improved detection and filtering system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from chess_video_analyzer.core.data_models import *
from chess_video_analyzer.detection.piece_recognizer import PieceRecognizer
from chess_video_analyzer.tracking.move_tracker import MoveTracker
from chess_video_analyzer.notation.game_state_manager import GameStateManager

def test_piece_recognition_strictness():
    """Test that piece recognition is more strict and reduces false positives."""
    print("Testing piece recognition strictness...")
    
    recognizer = PieceRecognizer()
    
    # Create test images
    # Empty square (uniform color)
    empty_square = np.full((64, 64, 3), 128, dtype=np.uint8)  # Gray square
    
    # Noisy but empty square
    noisy_empty = np.random.randint(120, 136, (64, 64, 3), dtype=np.uint8)
    
    # Square with very little variation (should be empty)
    low_variation = np.full((64, 64, 3), 130, dtype=np.uint8)
    low_variation[30:34, 30:34] = 135  # Small variation
    
    # Square with clear piece-like features
    piece_square = np.full((64, 64, 3), 100, dtype=np.uint8)
    cv2.circle(piece_square, (32, 32), 15, (200, 200, 200), -1)  # White circle (piece-like)
    # Add some texture to make it more piece-like
    cv2.circle(piece_square, (32, 28), 8, (180, 180, 180), -1)  # Inner detail
    cv2.circle(piece_square, (32, 36), 5, (220, 220, 220), -1)  # Another detail
    
    test_cases = [
        (empty_square, "Empty square", False),
        (noisy_empty, "Noisy empty square", False),
        (low_variation, "Low variation square", False),
        (piece_square, "Clear piece square", True),
    ]
    
    all_passed = True
    for image, description, should_have_piece in test_cases:
        result = recognizer.classify_piece(image)
        has_piece = result is not None
        
        if has_piece == should_have_piece:
            print(f"✓ {description}: correctly {'detected' if should_have_piece else 'rejected'}")
        else:
            print(f"✗ {description}: incorrectly {'detected' if has_piece else 'rejected'}")
            all_passed = False
    
    return all_passed

def test_move_filtering():
    """Test that illegal moves are filtered out during detection."""
    print("\nTesting move filtering...")
    
    manager = GameStateManager()
    tracker = MoveTracker()
    
    # Create two board states with an illegal move
    # Starting position
    starting_state = manager.get_current_game_state().current_position
    
    # Create an illegal state (pawn jumping from f2 to d7)
    illegal_squares = starting_state.squares.copy()
    illegal_squares[Position(5, 6)] = None  # Remove pawn from f2
    illegal_squares[Position(3, 1)] = PieceType(Color.WHITE, PieceKind.PAWN)  # Place at d7
    
    illegal_state = BoardState(
        squares=illegal_squares,
        timestamp=1.0,
        confidence=0.8
    )
    
    # Try to detect the move
    detected_move = tracker.detect_move(starting_state, illegal_state)
    
    if detected_move is None:
        print("✓ Illegal move correctly filtered out during detection")
        return True
    else:
        # If move was detected, it should be rejected by validation
        validation = manager.validate_move(detected_move)
        if not validation.is_legal:
            print(f"✓ Illegal move detected but correctly rejected: {validation.reason}")
            return True
        else:
            print(f"✗ Illegal move incorrectly accepted: {detected_move.piece.type.value} "
                  f"{detected_move.from_square.x},{detected_move.from_square.y} -> "
                  f"{detected_move.to_square.x},{detected_move.to_square.y}")
            return False

def test_realistic_move_sequence():
    """Test a realistic move sequence to ensure legal moves still work."""
    print("\nTesting realistic move sequence...")
    
    manager = GameStateManager()
    
    # Test the user's actual game sequence
    moves = [
        # 1. e4
        Move(Position(4, 6), Position(4, 4), PieceType(Color.WHITE, PieceKind.PAWN)),
        # 1... e5
        Move(Position(4, 1), Position(4, 3), PieceType(Color.BLACK, PieceKind.PAWN)),
        # 2. Nf3
        Move(Position(6, 7), Position(5, 5), PieceType(Color.WHITE, PieceKind.KNIGHT)),
        # 2... d5
        Move(Position(3, 1), Position(3, 3), PieceType(Color.BLACK, PieceKind.PAWN)),
        # 3. exd5
        Move(Position(4, 4), Position(3, 3), PieceType(Color.WHITE, PieceKind.PAWN),
             captured_piece=PieceType(Color.BLACK, PieceKind.PAWN)),
    ]
    
    all_valid = True
    for i, move in enumerate(moves):
        validation = manager.validate_move(move)
        if validation.is_legal:
            print(f"✓ Move {i+1} ({move.piece.type.value}) is legal")
            # Update state for next move
            new_board = _apply_move_to_board(manager.get_current_game_state().current_position, move)
            manager.update_state(move, new_board)
        else:
            print(f"✗ Move {i+1} ({move.piece.type.value}) incorrectly rejected: {validation.reason}")
            all_valid = False
            break
    
    return all_valid

def _apply_move_to_board(board_state: BoardState, move: Move) -> BoardState:
    """Helper to apply a move to a board state."""
    new_squares = board_state.squares.copy()
    new_squares[move.from_square] = None
    new_squares[move.to_square] = move.piece
    
    return BoardState(
        squares=new_squares,
        timestamp=board_state.timestamp + 1.0,
        confidence=board_state.confidence
    )

def main():
    """Run all improved detection tests."""
    print("Chess Video Analyzer - Improved Detection Tests")
    print("=" * 55)
    
    tests = [
        test_piece_recognition_strictness,
        test_move_filtering,
        test_realistic_move_sequence,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 55)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The improved detection should work better.")
        print("\nKey improvements:")
        print("• Stricter piece detection thresholds")
        print("• Better color classification with uncertainty handling")
        print("• Move validation during detection pipeline")
        print("• Illegal moves filtered out before being reported")
    else:
        print("⚠️  Some tests failed. Additional improvements may be needed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
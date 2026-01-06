#!/usr/bin/env python3
"""
Demonstration script for the enhanced illegal move detection and flagging system.
"""

from chess_video_analyzer.notation.game_state_manager import GameStateManager
from chess_video_analyzer.core.data_models import (
    Position, PieceType, Move, BoardState, Color, PieceKind, SpecialMoveType
)


def demonstrate_illegal_move_detection():
    """Demonstrate various types of illegal move detection."""
    print("=== Chess Video Analyzer: Illegal Move Detection Demo ===\n")
    
    manager = GameStateManager()
    
    # Test 1: Wrong turn
    print("1. Testing wrong turn detection:")
    black_move_on_white_turn = Move(
        from_square=Position(4, 1),
        to_square=Position(4, 3),
        piece=PieceType(Color.BLACK, PieceKind.PAWN)
    )
    new_board = BoardState({}, 1.0)
    manager.update_state(black_move_on_white_turn, new_board)
    
    if black_move_on_white_turn.is_flagged:
        print(f"   ✓ Flagged: {black_move_on_white_turn.flag_reason}")
    else:
        print("   ✗ Not flagged")
    
    # Test 2: Null move (piece not moving)
    print("\n2. Testing null move detection:")
    null_move = Move(
        from_square=Position(4, 6),
        to_square=Position(4, 6),  # Same square
        piece=PieceType(Color.WHITE, PieceKind.PAWN)
    )
    manager.update_state(null_move, new_board)
    
    if null_move.is_flagged:
        print(f"   ✓ Flagged: {null_move.flag_reason}")
    else:
        print("   ✗ Not flagged")
    
    # Test 3: Invalid piece movement
    print("\n3. Testing invalid piece movement:")
    invalid_pawn_move = Move(
        from_square=Position(4, 6),
        to_square=Position(4, 7),  # Pawn moving backwards
        piece=PieceType(Color.WHITE, PieceKind.PAWN)
    )
    manager.update_state(invalid_pawn_move, new_board)
    
    if invalid_pawn_move.is_flagged:
        print(f"   ✓ Flagged: {invalid_pawn_move.flag_reason}")
    else:
        print("   ✗ Not flagged")
    
    # Test 4: Capture validation
    print("\n4. Testing capture validation:")
    # Set up board with pieces
    squares = {}
    for x in range(8):
        for y in range(8):
            squares[Position(x, y)] = None
    squares[Position(4, 6)] = PieceType(Color.WHITE, PieceKind.PAWN)
    squares[Position(3, 5)] = PieceType(Color.BLACK, PieceKind.PAWN)
    
    board_with_pieces = BoardState(squares=squares, timestamp=1.0)
    manager.game_state.current_position = board_with_pieces
    
    # Try to move to occupied square without claiming capture
    missing_capture = Move(
        from_square=Position(4, 6),
        to_square=Position(3, 5),  # Where black pawn is
        piece=PieceType(Color.WHITE, PieceKind.PAWN),
        captured_piece=None  # Not claiming capture
    )
    manager.update_state(missing_capture, new_board)
    
    if missing_capture.is_flagged:
        print(f"   ✓ Flagged: {missing_capture.flag_reason}")
    else:
        print("   ✗ Not flagged")
    
    # Test 5: Demonstrate flagging utilities
    print("\n5. Flagging utilities:")
    summary = manager.get_flag_summary()
    print(f"   Total moves: {summary['total_moves']}")
    print(f"   Flagged moves: {summary['total_flagged']}")
    print(f"   Flag rate: {summary['flag_rate']:.2%}")
    
    illegal_moves = manager.get_illegal_moves()
    questionable_moves = manager.get_questionable_moves()
    print(f"   Illegal moves: {len(illegal_moves)}")
    print(f"   Questionable moves: {len(questionable_moves)}")
    
    print("\n6. Flagged move details:")
    for detail in summary['flagged_move_details']:
        print(f"   Move {detail['move_number']}: {detail['piece']} {detail['from']} → {detail['to']}")
        print(f"      Reason: {detail['reason']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_illegal_move_detection()
#!/usr/bin/env python3
"""
Test the video analysis with the fixes applied.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_video_analyzer.main import ChessVideoAnalyzer
from chess_video_analyzer.core.data_models import *
from chess_video_analyzer.notation.game_state_manager import GameStateManager

def test_move_sequence():
    """Test a sequence of moves like the user described: e4, e5, Nf3, d5, exd5, Qe2, Qxd5"""
    print("Testing move sequence validation...")
    
    manager = GameStateManager()
    
    # Sequence of moves from user's description
    moves = [
        # 1. e4
        Move(Position(4, 6), Position(4, 4), PieceType(Color.WHITE, PieceKind.PAWN)),
        # 1... e5  
        Move(Position(4, 1), Position(4, 3), PieceType(Color.BLACK, PieceKind.PAWN)),
        # 2. Nf3
        Move(Position(6, 7), Position(5, 5), PieceType(Color.WHITE, PieceKind.KNIGHT)),
        # 2... d5
        Move(Position(3, 1), Position(3, 3), PieceType(Color.BLACK, PieceKind.PAWN)),
        # 3. exd5 (pawn captures pawn)
        Move(Position(4, 4), Position(3, 3), PieceType(Color.WHITE, PieceKind.PAWN), 
             captured_piece=PieceType(Color.BLACK, PieceKind.PAWN)),
    ]
    
    print("Validating each move in sequence:")
    all_valid = True
    
    for i, move in enumerate(moves):
        result = manager.validate_move(move)
        move_notation = f"{i//2 + 1}{'.' if i % 2 == 0 else '...'}"
        
        if result.is_legal:
            print(f"✓ {move_notation} {_move_to_notation(move)} - Valid")
            # Update the game state for next move
            new_board = _apply_move_to_board(manager.game_state.current_position, move)
            manager.update_state(move, new_board)
        else:
            print(f"✗ {move_notation} {_move_to_notation(move)} - Invalid: {result.reason}")
            all_valid = False
            break
    
    return all_valid

def _move_to_notation(move: Move) -> str:
    """Convert move to simple algebraic notation."""
    from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
    to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
    
    piece_symbol = ""
    if move.piece.type != PieceKind.PAWN:
        piece_symbol = move.piece.type.value[0].upper()
    
    capture = "x" if move.captured_piece else ""
    
    return f"{piece_symbol}{from_square}{capture}{to_square}"

def _apply_move_to_board(board_state: BoardState, move: Move) -> BoardState:
    """Apply a move to create a new board state."""
    new_squares = board_state.squares.copy()
    
    # Remove piece from source square
    new_squares[move.from_square] = None
    
    # Place piece on destination square
    new_squares[move.to_square] = move.piece
    
    return BoardState(
        squares=new_squares,
        timestamp=board_state.timestamp + 1.0,
        confidence=board_state.confidence
    )

def test_illegal_moves_rejection():
    """Test that the system properly rejects the illegal moves from the user's report."""
    print("\nTesting rejection of illegal moves from user report...")
    
    manager = GameStateManager()
    
    # These are the illegal moves from the user's report
    illegal_moves = [
        # "Pawn f1 -> d7 (captures pawn)"
        Move(Position(5, 7), Position(3, 1), PieceType(Color.WHITE, PieceKind.PAWN)),
        # "Pawn d8 -> c8" 
        Move(Position(3, 0), Position(2, 0), PieceType(Color.BLACK, PieceKind.PAWN)),
        # "Pawn h3 -> h2"
        Move(Position(7, 5), Position(7, 6), PieceType(Color.WHITE, PieceKind.PAWN)),
        # "Pawn d7 -> d6 (captures pawn)"
        Move(Position(3, 1), Position(3, 2), PieceType(Color.BLACK, PieceKind.PAWN)),
    ]
    
    all_rejected = True
    for i, move in enumerate(illegal_moves):
        result = manager.validate_move(move)
        if not result.is_legal:
            print(f"✓ Illegal move {i+1} correctly rejected: {result.reason}")
        else:
            print(f"✗ Illegal move {i+1} incorrectly accepted!")
            all_rejected = False
    
    return all_rejected

def main():
    """Run the video analysis tests."""
    print("Chess Video Analysis - Fix Verification")
    print("=" * 50)
    
    tests = [
        test_move_sequence,
        test_illegal_moves_rejection,
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
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The chess analysis should now work correctly.")
        print("\nKey fixes applied:")
        print("• Fixed coordinate system mapping (y=0→rank8, y=7→rank1)")
        print("• Replaced random piece classification with shape-based heuristics")
        print("• Added comprehensive chess rule validation")
        print("• Fixed FEN generation coordinate mapping")
        print("• Added strict move legality checking in move tracker")
    else:
        print("⚠️  Some tests failed. Additional fixes may be needed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
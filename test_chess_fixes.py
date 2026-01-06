#!/usr/bin/env python3
"""
Test script to verify chess analysis fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_video_analyzer.core.data_models import *
from chess_video_analyzer.notation.game_state_manager import GameStateManager
from chess_video_analyzer.notation.fen_generator import FENGenerator

@dataclass
class MoveValidationResult:
    """Result of move validation."""
    is_legal: bool
    reason: Optional[str] = None

def test_coordinate_system():
    """Test that coordinate system is consistent."""
    print("Testing coordinate system...")
    
    manager = GameStateManager()
    fen_gen = FENGenerator()
    
    # Get starting position
    starting_state = manager.get_current_game_state()
    starting_fen = fen_gen.generate_fen(starting_state)
    
    print(f"Starting FEN: {starting_fen}")
    
    # Should be standard starting position
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    if starting_fen == expected_fen:
        print("✓ Coordinate system is correct!")
        return True
    else:
        print(f"✗ Coordinate system error!")
        print(f"Expected: {expected_fen}")
        print(f"Got:      {starting_fen}")
        return False

def test_move_validation():
    """Test move validation with legal and illegal moves."""
    print("\nTesting move validation...")
    
    manager = GameStateManager()
    
    # Test legal moves
    legal_moves = [
        # e4 (pawn two squares forward)
        Move(Position(4, 6), Position(4, 4), PieceType(Color.WHITE, PieceKind.PAWN)),
        # Nf3 (knight move)  
        Move(Position(6, 7), Position(5, 5), PieceType(Color.WHITE, PieceKind.KNIGHT)),
    ]
    
    # Test illegal moves
    illegal_moves = [
        # Pawn f1 -> d7 (impossible pawn move)
        Move(Position(5, 7), Position(3, 1), PieceType(Color.WHITE, PieceKind.PAWN)),
        # Pawn backwards
        Move(Position(4, 6), Position(4, 7), PieceType(Color.WHITE, PieceKind.PAWN)),
        # Knight illegal move
        Move(Position(6, 7), Position(4, 4), PieceType(Color.WHITE, PieceKind.KNIGHT)),
    ]
    
    print("Testing legal moves:")
    all_passed = True
    for i, move in enumerate(legal_moves):
        result = manager.validate_move(move)
        if result.is_legal:
            print(f"✓ Legal move {i+1} correctly validated")
        else:
            print(f"✗ Legal move {i+1} incorrectly rejected: {result.reason}")
            all_passed = False
    
    print("Testing illegal moves:")
    for i, move in enumerate(illegal_moves):
        result = manager.validate_move(move)
        if not result.is_legal:
            print(f"✓ Illegal move {i+1} correctly rejected: {result.reason}")
        else:
            print(f"✗ Illegal move {i+1} incorrectly accepted")
            all_passed = False
    
    return all_passed

def test_pawn_movement():
    """Test specific pawn movement rules."""
    print("\nTesting pawn movement rules...")
    
    manager = GameStateManager()
    
    # Test white pawn moves
    white_pawn_moves = [
        # e2-e4 (two squares from starting position)
        (Position(4, 6), Position(4, 4), True, "Two squares from start"),
        # e2-e3 (one square from starting position)  
        (Position(4, 6), Position(4, 5), True, "One square from start"),
        # e4-e5 (one square forward)
        (Position(4, 4), Position(4, 3), True, "One square forward"),
        # e4-e6 (two squares not from start)
        (Position(4, 4), Position(4, 2), False, "Two squares not from start"),
        # e4-e3 (backwards)
        (Position(4, 4), Position(4, 5), False, "Backwards move"),
    ]
    
    all_passed = True
    for from_pos, to_pos, should_be_valid, description in white_pawn_moves:
        move = Move(from_pos, to_pos, PieceType(Color.WHITE, PieceKind.PAWN))
        result = manager.validate_move(move)
        
        if result.is_legal == should_be_valid:
            print(f"✓ {description}: correctly {'accepted' if should_be_valid else 'rejected'}")
        else:
            print(f"✗ {description}: incorrectly {'accepted' if result.is_legal else 'rejected'}")
            if not result.is_legal:
                print(f"  Reason: {result.reason}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("Chess Video Analyzer - Fix Verification Tests")
    print("=" * 50)
    
    tests = [
        test_coordinate_system,
        test_move_validation, 
        test_pawn_movement,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The fixes appear to be working.")
    else:
        print("⚠️  Some tests failed. Additional fixes may be needed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
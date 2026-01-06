#!/usr/bin/env python3
"""
Validation script to verify the Chess Video Analyzer setup.
"""

import sys
from chess_video_analyzer.core.data_models import (
    Position, PieceType, PieceKind, Color, Move, BoardState, 
    GameState, VideoMetadata, GameMetadata, CastlingRights
)

def test_core_functionality():
    """Test basic functionality of core data models."""
    print("Testing core data models...")
    
    # Test Position
    pos = Position(3, 4)
    assert pos.x == 3 and pos.y == 4
    print("✓ Position class working")
    
    # Test PieceType
    piece = PieceType(Color.WHITE, PieceKind.KING)
    assert piece.color == Color.WHITE and piece.type == PieceKind.KING
    print("✓ PieceType class working")
    
    # Test BoardState
    board = BoardState({}, 0.0, 0.95)
    assert board.confidence == 0.95
    print("✓ BoardState class working")
    
    # Test GameState
    castling = CastlingRights()
    game = GameState(board, [], castling)
    assert game.active_color == Color.WHITE
    print("✓ GameState class working")
    
    # Test VideoMetadata
    video_meta = VideoMetadata(120.0, 30.0, (1920, 1080), "mp4")
    assert video_meta.duration == 120.0
    print("✓ VideoMetadata class working")
    
    print("All core data models are working correctly!")

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV import failed")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy import failed")
        return False
    
    try:
        import chess
        print("✓ python-chess imported successfully")
    except ImportError:
        print("✗ python-chess import failed")
        return False
    
    try:
        import pytest
        print("✓ pytest imported successfully")
    except ImportError:
        print("✗ pytest import failed")
        return False
    
    try:
        import hypothesis
        print("✓ hypothesis imported successfully")
    except ImportError:
        print("✗ hypothesis import failed")
        return False
    
    print("All dependencies are available!")
    return True

def main():
    """Main validation function."""
    print("Chess Video Analyzer Setup Validation")
    print("=" * 40)
    
    try:
        test_core_functionality()
        if test_dependencies():
            print("\n🎉 Setup validation completed successfully!")
            print("The Chess Video Analyzer project is ready for development.")
            return 0
        else:
            print("\n❌ Some dependencies are missing.")
            return 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
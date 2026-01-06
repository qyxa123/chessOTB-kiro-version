#!/usr/bin/env python3
"""
Debug script to diagnose video analysis issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from chess_video_analyzer.main import ChessVideoAnalyzer
from chess_video_analyzer.video.processor import VideoProcessor
from chess_video_analyzer.detection.board_detector import BoardDetector
from chess_video_analyzer.detection.piece_recognizer import PieceRecognizer

def debug_video_loading():
    """Test if video can be loaded properly."""
    print("=== Testing Video Loading ===")
    
    video_path = "IMG_4550.MOV"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        processor = VideoProcessor()
        metadata = processor.load_video(video_path)
        print(f"✅ Video loaded successfully:")
        print(f"   Duration: {metadata.duration:.2f}s")
        print(f"   FPS: {metadata.fps}")
        print(f"   Resolution: {metadata.resolution}")
        print(f"   Format: {metadata.format}")
        
        # Test frame extraction
        frames = list(processor.extract_frames(2.0))  # Extract every 2 seconds
        print(f"   Extracted {len(frames)} frames")
        
        if len(frames) > 0:
            print(f"   First frame shape: {frames[0].shape}")
            return True
        else:
            print("❌ No frames extracted")
            return False
            
    except Exception as e:
        print(f"❌ Video loading failed: {e}")
        return False

def debug_board_detection():
    """Test board detection on sample frames."""
    print("\n=== Testing Board Detection ===")
    
    try:
        processor = VideoProcessor()
        processor.load_video("IMG_4550.MOV")
        
        detector = BoardDetector()
        
        # Test on first few frames
        frames = list(processor.extract_frames(5.0))  # Every 5 seconds
        
        detected_count = 0
        for i, frame in enumerate(frames[:3]):  # Test first 3 frames
            try:
                board_region = detector.detect_board(frame)
                print(f"✅ Frame {i+1}: Board detected with confidence {board_region.confidence:.2f}")
                print(f"   Corners: {board_region.corners}")
                detected_count += 1
            except Exception as e:
                print(f"❌ Frame {i+1}: Board detection failed - {e}")
        
        return detected_count > 0
        
    except Exception as e:
        print(f"❌ Board detection test failed: {e}")
        return False

def debug_piece_recognition():
    """Test piece recognition with relaxed thresholds."""
    print("\n=== Testing Piece Recognition ===")
    
    try:
        # Create test images
        recognizer = PieceRecognizer()
        
        # Test with a simple synthetic piece
        test_piece = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.circle(test_piece, (32, 32), 20, (200, 200, 200), -1)
        cv2.circle(test_piece, (32, 28), 10, (180, 180, 180), -1)
        
        result = recognizer.classify_piece(test_piece)
        if result:
            print(f"✅ Synthetic piece detected: {result.color.value} {result.type.value}")
        else:
            print("❌ Synthetic piece not detected")
        
        # Test with empty square
        empty_square = np.full((64, 64, 3), 130, dtype=np.uint8)
        result = recognizer.classify_piece(empty_square)
        if result is None:
            print("✅ Empty square correctly rejected")
        else:
            print(f"❌ Empty square incorrectly detected as: {result.color.value} {result.type.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Piece recognition test failed: {e}")
        return False

def debug_full_pipeline():
    """Test the full analysis pipeline with debug output."""
    print("\n=== Testing Full Pipeline ===")
    
    try:
        analyzer = ChessVideoAnalyzer(
            confidence_threshold=0.5,  # Lower threshold
            enable_quality_control=True,
            enable_ui=False
        )
        
        # Load video
        metadata = analyzer.load_video("IMG_4550.MOV")
        print(f"✅ Video loaded: {metadata.duration:.2f}s")
        
        # Process with larger frame interval to get fewer frames
        print("Starting analysis with debug logging...")
        
        # Enable debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        results = analyzer.process_video(
            frame_interval=3.0,  # Every 3 seconds
            game_metadata=None
        )
        
        print(f"✅ Analysis completed:")
        print(f"   Processing time: {results['processing_time']:.2f}s")
        print(f"   Board states: {len(results['board_states'])}")
        print(f"   Detected moves: {len(results['detected_moves'])}")
        
        if len(results['detected_moves']) > 0:
            print("   Moves found:")
            for i, move in enumerate(results['detected_moves']):
                print(f"     {i+1}. {move.piece.type.value} {move.from_square.x},{move.from_square.y} -> {move.to_square.x},{move.to_square.y}")
        else:
            print("   No moves detected")
        
        return len(results['detected_moves']) > 0
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def relax_detection_thresholds():
    """Temporarily relax detection thresholds for debugging."""
    print("\n=== Relaxing Detection Thresholds ===")
    
    # We'll modify the piece recognizer to be less strict
    from chess_video_analyzer.detection.piece_recognizer import PieceRecognizer
    
    # Monkey patch the _has_piece method to be less strict
    original_has_piece = PieceRecognizer._has_piece
    
    def relaxed_has_piece(self, square_image):
        try:
            if len(square_image.shape) == 3:
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_image
            
            std_intensity = np.std(gray)
            
            # Much more relaxed threshold
            if std_intensity < 10:  # Very relaxed
                return False
            
            # Simple edge detection
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Lower threshold for edge density
            return edge_density > 0.02  # Very low threshold
            
        except Exception:
            return False
    
    PieceRecognizer._has_piece = relaxed_has_piece
    print("✅ Detection thresholds relaxed")

def main():
    """Run all debug tests."""
    print("Chess Video Analysis - Debug Mode")
    print("=" * 50)
    
    # Relax thresholds first
    relax_detection_thresholds()
    
    tests = [
        ("Video Loading", debug_video_loading),
        ("Board Detection", debug_board_detection),
        ("Piece Recognition", debug_piece_recognition),
        ("Full Pipeline", debug_full_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Debug tests passed: {passed}/{total}")
    
    if passed < total:
        print("\n🔧 Recommendations:")
        if passed == 0:
            print("• Check if IMG_4550.MOV exists and is readable")
            print("• Verify video format is supported")
        elif passed == 1:
            print("• Board detection may be failing - check lighting/angle")
            print("• Try adjusting detection parameters")
        elif passed == 2:
            print("• Piece recognition may be too strict")
            print("• Consider lowering confidence thresholds")
        else:
            print("• Pipeline integration issues")
            print("• Check move validation logic")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run the actual video analysis on IMG_4550.MOV with improved settings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_video_analyzer.main import ChessVideoAnalyzer
from chess_video_analyzer.core.data_models import GameMetadata
from datetime import datetime

def run_analysis():
    """Run the video analysis with optimized settings."""
    print("Chess Video Analysis - IMG_4550.MOV")
    print("=" * 50)
    
    try:
        # Create analyzer with relaxed settings
        analyzer = ChessVideoAnalyzer(
            confidence_threshold=0.4,  # Lower threshold for more permissive detection
            enable_quality_control=True,
            enable_ui=False
        )
        
        # Load the video
        print("Loading video...")
        metadata = analyzer.load_video("IMG_4550.MOV")
        print(f"✅ Video loaded: {metadata.duration:.1f}s, {metadata.resolution[0]}x{metadata.resolution[1]}")
        
        # Create game metadata
        game_metadata = GameMetadata(
            event="Video Analysis",
            site="Unknown",
            date=datetime.now().strftime("%Y.%m.%d"),
            round="1",
            white_player="White",
            black_player="Black",
            result="*"
        )
        
        # Process the video
        print("Starting analysis...")
        results = analyzer.process_video(
            frame_interval=2.0,  # Extract frame every 2 seconds
            game_metadata=game_metadata
        )
        
        print(f"\n✅ Analysis completed in {results['processing_time']:.1f}s")
        print(f"Board states detected: {len(results['board_states'])}")
        print(f"Moves detected: {len(results['detected_moves'])}")
        
        if len(results['detected_moves']) > 0:
            print("\nDetected moves:")
            for i, move in enumerate(results['detected_moves']):
                from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
                to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
                
                piece_name = move.piece.type.value.capitalize()
                color = move.piece.color.value.capitalize()
                
                flag_info = ""
                if move.is_flagged:
                    flag_info = f" [FLAGGED: {move.flag_reason}]"
                
                capture_info = ""
                if move.captured_piece:
                    capture_info = f" (captures {move.captured_piece.type.value})"
                
                print(f"{i+1}. {color}: {piece_name} {from_square} -> {to_square}{capture_info}{flag_info}")
        
        # Show PGN if generated
        if results.get('pgn_content'):
            print(f"\nGenerated PGN:")
            print(results['pgn_content'])
        
        # Show quality report if available
        if results.get('quality_report'):
            quality_report = results['quality_report']
            print(f"\nQuality Report:")
            print(f"Overall confidence: {quality_report.overall_confidence:.2f}")
            if quality_report.issues:
                print(f"Issues detected: {len(quality_report.issues)}")
                for issue in quality_report.issues[:5]:  # Show first 5 issues
                    print(f"  - {issue}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if not os.path.exists("IMG_4550.MOV"):
        print("❌ Video file IMG_4550.MOV not found!")
        print("Please make sure the video file is in the current directory.")
        return False
    
    success = run_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\nExpected moves based on your description:")
        print("1. White: e4, Black: e5")
        print("2. White: Nf3, Black: d5") 
        print("3. White: exd5 (pawn captures pawn)")
        print("4. White: Qe2, Black: Qxd5 (queen captures pawn)")
        print("\nCompare the detected moves above with the expected sequence.")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
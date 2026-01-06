"""
Main Chess Video Analyzer orchestrator class.

This module implements the ChessVideoAnalyzer class that orchestrates all components
to provide a complete pipeline from video input to notation output.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime
import cv2
import numpy as np

from .video.processor import VideoProcessor, VideoProcessingError
from .detection.board_detector import BoardDetector, BoardDetectionError
from .detection.piece_recognizer import PieceRecognizer
from .tracking.move_tracker import MoveTracker
from .notation.game_state_manager import GameStateManager
from .notation.pgn_generator import PGNGenerator
from .notation.fen_generator import FENGenerator
from .quality.quality_controller import QualityController, QualityReport
from .ui.main_interface import MainInterface
from .ui.app_launcher import launch_application
from .core.data_models import (
    VideoMetadata, GameMetadata, GameState, BoardState, Move,
    Position, PieceType, Color, GameResult, BoardRegion, SquareGrid, Orientation,
    PieceKind, CastlingRights
)


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class ChessVideoAnalyzer:
    """
    Main orchestrator class for the Chess Video Analyzer system.
    
    This class coordinates all components to provide a complete pipeline
    from video input to chess notation output, with comprehensive error
    handling and user feedback.
    
    Requirements: All requirements
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 enable_quality_control: bool = True,
                 enable_ui: bool = True):
        """
        Initialize the Chess Video Analyzer.
        
        Args:
            confidence_threshold: Minimum confidence for acceptable quality
            enable_quality_control: Whether to enable quality control features
            enable_ui: Whether to enable the user interface
        """
        # Initialize core components
        self.video_processor = VideoProcessor()
        self.board_detector = BoardDetector()
        self.piece_recognizer = PieceRecognizer()
        self.move_tracker = MoveTracker(confidence_threshold=confidence_threshold)
        self.game_state_manager = GameStateManager()
        self.pgn_generator = PGNGenerator()
        self.fen_generator = FENGenerator()
        
        # Initialize quality control
        self.quality_controller = None
        if enable_quality_control:
            self.quality_controller = QualityController(
                confidence_threshold=confidence_threshold
            )
        
        # Initialize UI components
        self.main_interface = None
        if enable_ui:
            # UI will be launched separately when needed
            pass
        
        # Processing state
        self.current_video_path: Optional[str] = None
        self.video_metadata: Optional[VideoMetadata] = None
        self.processing_results: Dict[str, Any] = {}
        self.is_processing = False
        
        # Callbacks for progress updates
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_video(self, video_path: str) -> VideoMetadata:
        """
        Load a video file for processing.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata: Metadata about the loaded video
            
        Raises:
            ProcessingError: If video loading fails
            
        Requirements: 1.1, 1.2, 1.4
        """
        try:
            self.logger.info(f"Loading video: {video_path}")
            
            # Load video using video processor
            self.video_metadata = self.video_processor.load_video(video_path)
            self.current_video_path = video_path
            
            # Assess video quality if quality control is enabled
            if self.quality_controller:
                quality_issues = self.quality_controller.assess_video_quality(self.video_metadata)
                if quality_issues:
                    self.logger.warning(f"Video quality issues detected: {len(quality_issues)} issues")
                    for issue in quality_issues:
                        self.logger.warning(f"  - {issue.description}")
            
            self.logger.info(f"Video loaded successfully: {self.video_metadata.duration:.1f}s, "
                           f"{self.video_metadata.fps:.1f} fps, "
                           f"{self.video_metadata.resolution[0]}x{self.video_metadata.resolution[1]}")
            
            return self.video_metadata
            
        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def process_video(self, 
                     game_metadata: Optional[GameMetadata] = None,
                     frame_interval: float = 1.0) -> Dict[str, Any]:
        """
        Process the loaded video to extract chess moves and generate notation.
        
        Args:
            game_metadata: Optional metadata about the game
            frame_interval: Interval between processed frames in seconds
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ProcessingError: If processing fails
            
        Requirements: All requirements
        """
        if not self.current_video_path or not self.video_metadata:
            raise ProcessingError("No video loaded. Call load_video() first.")
        
        if self.is_processing:
            raise ProcessingError("Processing already in progress.")
        
        try:
            self.is_processing = True
            start_time = time.time()
            
            self.logger.info("Starting video processing...")
            self._update_progress("Initializing processing...", 0.0)
            
            # Create default game metadata if not provided
            if game_metadata is None:
                game_metadata = GameMetadata(
                    event="Video Analysis",
                    site="Unknown",
                    date=datetime.now().strftime("%Y.%m.%d"),
                    round="1",
                    white_player="White",
                    black_player="Black",
                    result="*"
                )
            
            # Reset components for new processing
            self._reset_components()
            
            # Extract and process frames
            board_states = self._extract_and_process_frames(frame_interval)
            self._update_progress("Analyzing moves...", 0.7)
            
            # Detect moves from board states
            moves = self._detect_moves_from_states(board_states)
            self._update_progress("Validating game state...", 0.8)
            
            # Build final game state
            final_game_state = self._build_final_game_state(moves, board_states)
            self._update_progress("Generating notation...", 0.9)
            
            # Generate PGN and FEN
            pgn_content = self.pgn_generator.generate_pgn(final_game_state, game_metadata)
            fen_sequence = self._generate_fen_sequence(final_game_state)
            
            # Generate quality report
            quality_report = None
            if self.quality_controller:
                quality_report = self.quality_controller.generate_quality_report()
            
            processing_time = time.time() - start_time
            
            # Store results
            self.processing_results = {
                'game_state': final_game_state,
                'game_metadata': game_metadata,
                'pgn_content': pgn_content,
                'fen_sequence': fen_sequence,
                'quality_report': quality_report,
                'processing_time': processing_time,
                'video_metadata': self.video_metadata,
                'board_states': board_states,
                'detected_moves': moves
            }
            
            self._update_progress("Processing complete!", 1.0)
            self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return self.processing_results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.logger.error(error_msg)
            self._handle_error(error_msg)
            raise ProcessingError(error_msg) from e
        finally:
            self.is_processing = False
    
    def _extract_and_process_frames(self, frame_interval: float) -> List[BoardState]:
        """Extract frames and detect board states."""
        board_states = []
        frame_count = 0
        total_frames = int(self.video_metadata.duration / frame_interval)
        
        self.logger.info(f"Processing {total_frames} frames at {frame_interval}s intervals")
        
        try:
            for frame in self.video_processor.extract_frames(frame_interval):
                frame_count += 1
                progress = 0.1 + (0.5 * frame_count / total_frames)  # 10% to 60% progress
                self._update_progress(f"Processing frame {frame_count}/{total_frames}...", progress)
                
                try:
                    # Detect board in frame
                    board_region = self.board_detector.detect_board(frame)
                    
                    # Get square coordinates
                    square_grid = self.board_detector.get_square_coordinates(board_region)
                    
                    # Create board state (simplified - in full implementation would recognize pieces)
                    board_state = self._create_board_state_from_frame(
                        frame, square_grid, frame_count * frame_interval
                    )
                    
                    # Assess quality if enabled
                    if self.quality_controller:
                        quality_issues = self.quality_controller.assess_board_state_quality(
                            board_state, frame_count, frame_count * frame_interval
                        )
                        if quality_issues:
                            self.logger.debug(f"Frame {frame_count}: {len(quality_issues)} quality issues")
                    
                    board_states.append(board_state)
                    
                except BoardDetectionError as e:
                    self.logger.warning(f"Board detection failed for frame {frame_count}: {e}")
                    # Continue processing other frames
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_count}: {e}")
                    # Continue processing other frames
                    continue
            
            self.logger.info(f"Successfully processed {len(board_states)} frames")
            return board_states
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            raise ProcessingError(f"Frame extraction failed: {e}") from e
    
    def _create_board_state_from_frame(self, frame, square_grid, timestamp: float) -> BoardState:
        """Create a board state from a frame with basic piece detection."""
        squares = {}
        
        # Basic piece detection using color and texture analysis
        for position in square_grid.squares:
            piece = self._detect_piece_in_square(frame, square_grid, position)
            squares[position] = piece
        
        # Calculate confidence based on detection quality
        confidence = self._calculate_board_confidence(squares, frame)
        
        return BoardState(
            squares=squares,
            timestamp=timestamp,
            confidence=confidence
        )
    
    def _detect_piece_in_square(self, frame, square_grid, position: Position) -> Optional[PieceType]:
        """Detect if there's a piece in a specific square using proper piece recognition."""
        try:
            # Get square coordinates
            if position not in square_grid.squares:
                return None
            
            x1, y1, x2, y2 = square_grid.squares[position]
            
            # Extract square region from frame
            square_region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if square_region.size == 0:
                return None
            
            # Use the proper piece recognizer instead of basic detection
            piece_type = self.piece_recognizer.classify_piece(square_region)
            
            return piece_type
            
        except Exception as e:
            self.logger.debug(f"Error detecting piece at {position}: {e}")
            return None
    
    def _calculate_board_confidence(self, squares: Dict[Position, Optional[PieceType]], frame) -> float:
        """Calculate confidence score for the board state."""
        try:
            # Count detected pieces
            piece_count = sum(1 for piece in squares.values() if piece is not None)
            
            # Basic confidence calculation
            # More pieces detected = higher confidence (up to a reasonable limit)
            if piece_count == 0:
                return 0.3  # Low confidence if no pieces detected
            elif piece_count < 10:
                return 0.5 + (piece_count * 0.03)  # Gradually increase confidence
            else:
                return 0.8  # High confidence if many pieces detected
                
        except Exception:
            return 0.5  # Default confidence
    
    def _detect_moves_from_states(self, board_states: List[BoardState]) -> List[Move]:
        """Detect moves by comparing consecutive board states."""
        moves = []
        
        if len(board_states) < 2:
            self.logger.warning("Not enough board states to detect moves")
            return moves
        
        self.logger.info(f"Detecting moves from {len(board_states)} board states")
        
        # Initialize game state with the first detected board state instead of standard position
        if board_states:
            self.game_state_manager.set_custom_starting_position(board_states[0])
            self.logger.info("Initialized game state with first detected board position")
        
        for i in range(1, len(board_states)):
            previous_state = board_states[i - 1]
            current_state = board_states[i]
            
            try:
                # Detect move between states
                move = self.move_tracker.detect_move(previous_state, current_state)
                
                if move:
                    # RELAXED: Try to validate move, but be more permissive
                    validation_result = self.game_state_manager.validate_move(move)
                    
                    if validation_result.is_legal:
                        # Accept legal moves
                        moves.append(move)
                        self.logger.debug(f"Accepted legal move {len(moves)}: {move.piece.type.value} "
                                        f"{move.from_square.x},{move.from_square.y} -> "
                                        f"{move.to_square.x},{move.to_square.y}")
                        
                        # Update game state for next validation
                        new_board_state = self._apply_move_to_board_state(current_state, move)
                        self.game_state_manager.update_state(move, new_board_state)
                    else:
                        # For now, be more permissive - accept moves that look reasonable
                        # even if they don't pass strict validation
                        if self._is_reasonable_move(move, previous_state, current_state):
                            move.is_flagged = True
                            move.flag_reason = f"Validation failed: {validation_result.reason}"
                            moves.append(move)
                            self.logger.warning(f"Accepted flagged move: {move.piece.type.value} "
                                              f"{move.from_square.x},{move.from_square.y} -> "
                                              f"{move.to_square.x},{move.to_square.y} - {validation_result.reason}")
                            
                            # Update game state with the actual board state
                            self.game_state_manager.update_state(move, current_state)
                        else:
                            # Log rejected illegal moves
                            self.logger.warning(f"Rejected illegal move: {move.piece.type.value} "
                                              f"{move.from_square.x},{move.from_square.y} -> "
                                              f"{move.to_square.x},{move.to_square.y} - {validation_result.reason}")
                
                    # Assess move quality if enabled
                    if self.quality_controller and move:
                        quality_issues = self.quality_controller.assess_move_quality(
                            move, previous_state, current_state, i, current_state.timestamp
                        )
                        if quality_issues:
                            self.logger.debug(f"Move {len(moves)}: {len(quality_issues)} quality issues")
                
            except Exception as e:
                self.logger.warning(f"Move detection failed between frames {i-1} and {i}: {e}")
                continue
        
        self.logger.info(f"Detected {len(moves)} moves ({sum(1 for m in moves if not m.is_flagged)} legal, {sum(1 for m in moves if m.is_flagged)} flagged)")
        return moves
    
    def _is_reasonable_move(self, move: Move, previous_state: BoardState, current_state: BoardState) -> bool:
        """
        Check if a move is reasonable even if it doesn't pass strict validation.
        This is more permissive than full chess rule validation.
        """
        try:
            # Basic sanity checks
            if move.from_square == move.to_square:
                return False
            
            # Check if the move distance is reasonable (not across the entire board)
            dx = abs(move.to_square.x - move.from_square.x)
            dy = abs(move.to_square.y - move.from_square.y)
            max_distance = max(dx, dy)
            
            if max_distance > 7:  # Can't move more than 7 squares in any direction
                return False
            
            # Check if there was actually a piece at the source in the previous state
            source_piece = previous_state.squares.get(move.from_square)
            if source_piece is None:
                return False
            
            # Check if the piece colors match (allowing for some detection uncertainty)
            if source_piece.color != move.piece.color:
                return False
            
            # Check if the destination changed between states
            prev_dest = previous_state.squares.get(move.to_square)
            curr_dest = current_state.squares.get(move.to_square)
            
            # There should be some change at the destination
            if prev_dest == curr_dest and curr_dest != move.piece:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Error in reasonable move check: {e}")
            return False
    
    def _apply_move_to_board_state(self, board_state: BoardState, move: Move) -> BoardState:
        """Apply a move to create a new board state."""
        new_squares = board_state.squares.copy()
        
        # Remove piece from source square
        new_squares[move.from_square] = None
        
        # Place piece on destination square
        new_squares[move.to_square] = move.piece
        
        return BoardState(
            squares=new_squares,
            timestamp=board_state.timestamp,
            confidence=board_state.confidence
        )
    
    def _build_final_game_state(self, moves: List[Move], board_states: List[BoardState]) -> GameState:
        """Build the final game state from detected moves."""
        self.logger.info("Building final game state...")
        
        # Reset game state manager
        if board_states:
            self.game_state_manager.set_custom_starting_position(board_states[0])
        else:
            self.game_state_manager.reset_to_starting_position()
        
        # Process each move through the game state manager
        for i, move in enumerate(moves):
            try:
                # Get corresponding board state if available
                board_state = board_states[i + 1] if i + 1 < len(board_states) else board_states[-1]
                
                # Update game state with the move
                self.game_state_manager.update_state(move, board_state)
                
            except Exception as e:
                self.logger.warning(f"Failed to process move {i + 1}: {e}")
                # Flag the move for review
                if self.quality_controller:
                    self.quality_controller.flag_for_manual_review(
                        move, 
                        self.quality_controller.QualityFlag.INCONSISTENT_MOVE,
                        f"Failed to process move: {e}"
                    )
        
        final_state = self.game_state_manager.get_current_state()
        self.logger.info(f"Final game state: {len(final_state.move_history)} moves, "
                        f"{len(final_state.flagged_moves)} flagged")
        
        return final_state
    
    def _generate_fen_sequence(self, game_state: GameState) -> List[str]:
        """Generate FEN sequence for all positions in the game."""
        fen_sequence = []
        
        # Generate FEN for starting position
        starting_fen = self.fen_generator.generate_fen(game_state)
        fen_sequence.append(starting_fen)
        
        # For a complete implementation, we would need to reconstruct
        # all intermediate positions. For now, just return the final position.
        
        return fen_sequence
    
    def _reset_components(self):
        """Reset all components for new processing."""
        self.move_tracker.clear_history()
        self.game_state_manager.reset_to_starting_position()
        if self.quality_controller:
            self.quality_controller.clear_history()
    
    def _update_progress(self, message: str, progress: float):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        self.logger.debug(f"Progress: {progress:.1%} - {message}")
    
    def _handle_error(self, error_message: str):
        """Handle error if callback is set."""
        if self.error_callback:
            self.error_callback(error_message)
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error handling."""
        self.error_callback = callback
    
    def export_pgn(self, output_path: str) -> None:
        """
        Export PGN to file.
        
        Args:
            output_path: Path to save the PGN file
            
        Raises:
            ProcessingError: If no results available or export fails
            
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        if 'pgn_content' not in self.processing_results:
            raise ProcessingError("No PGN content available. Process a video first.")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.processing_results['pgn_content'])
            
            self.logger.info(f"PGN exported to: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to export PGN: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def export_fen(self, output_path: str) -> None:
        """
        Export FEN sequence to file.
        
        Args:
            output_path: Path to save the FEN file
            
        Raises:
            ProcessingError: If no results available or export fails
            
        Requirements: 6.1, 6.2, 6.3
        """
        if 'fen_sequence' not in self.processing_results:
            raise ProcessingError("No FEN content available. Process a video first.")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, fen in enumerate(self.processing_results['fen_sequence']):
                    f.write(f"Move {i}: {fen}\n")
            
            self.logger.info(f"FEN sequence exported to: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to export FEN: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def get_quality_report(self) -> Optional[QualityReport]:
        """
        Get the quality report from the last processing run.
        
        Returns:
            QualityReport if available, None otherwise
            
        Requirements: 7.2
        """
        return self.processing_results.get('quality_report')
    
    def get_flagged_moves(self) -> List[Move]:
        """
        Get all moves flagged for review.
        
        Returns:
            List of flagged moves
            
        Requirements: 7.4
        """
        if 'game_state' in self.processing_results:
            return self.processing_results['game_state'].flagged_moves
        return []
    
    def launch_ui(self) -> None:
        """
        Launch the graphical user interface.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        try:
            self.logger.info("Launching user interface...")
            launch_application()
            
        except Exception as e:
            error_msg = f"Failed to launch UI: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def validate_output(self, pgn_path: Optional[str] = None, 
                       fen_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Validate exported output files.
        
        Args:
            pgn_path: Path to PGN file to validate
            fen_path: Path to FEN file to validate
            
        Returns:
            Dictionary with validation results
            
        Requirements: 7.5
        """
        results = {}
        
        if pgn_path:
            try:
                with open(pgn_path, 'r', encoding='utf-8') as f:
                    pgn_content = f.read()
                results['pgn_valid'] = self.pgn_generator.validate_pgn(pgn_content)
            except Exception as e:
                self.logger.error(f"PGN validation failed: {e}")
                results['pgn_valid'] = False
        
        if fen_path:
            try:
                with open(fen_path, 'r', encoding='utf-8') as f:
                    fen_lines = f.readlines()
                
                # Validate each FEN string
                all_valid = True
                for line in fen_lines:
                    if line.strip() and ':' in line:
                        fen_string = line.split(':', 1)[1].strip()
                        if not self.fen_generator.validate_fen(fen_string):
                            all_valid = False
                            break
                
                results['fen_valid'] = all_valid
            except Exception as e:
                self.logger.error(f"FEN validation failed: {e}")
                results['fen_valid'] = False
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the last processing run.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.processing_results:
            return {}
        
        game_state = self.processing_results.get('game_state')
        quality_report = self.processing_results.get('quality_report')
        
        stats = {
            'processing_time': self.processing_results.get('processing_time', 0),
            'video_duration': self.video_metadata.duration if self.video_metadata else 0,
            'total_moves': len(game_state.move_history) if game_state else 0,
            'flagged_moves': len(game_state.flagged_moves) if game_state else 0,
            'board_states_processed': len(self.processing_results.get('board_states', [])),
        }
        
        if quality_report:
            stats.update({
                'overall_confidence': quality_report.overall_confidence,
                'quality_issues': len(quality_report.issues),
                'critical_issues': len(quality_report.get_critical_issues()),
            })
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.video_processor:
            self.video_processor.close()
        
        self.current_video_path = None
        self.video_metadata = None
        self.processing_results.clear()
        self.is_processing = False
        
        self.logger.info("Chess Video Analyzer closed")


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess Video Analyzer")
    parser.add_argument("video_path", nargs="?", help="Path to chess game video")
    parser.add_argument("--output-dir", "-o", help="Output directory for generated files")
    parser.add_argument("--no-ui", action="store_true", help="Run without UI")
    parser.add_argument("--confidence", "-c", type=float, default=0.7, 
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--frame-interval", "-f", type=float, default=1.0,
                       help="Frame processing interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ChessVideoAnalyzer(
        confidence_threshold=args.confidence,
        enable_ui=not args.no_ui
    )
    
    try:
        if args.video_path:
            # Command-line processing
            print(f"Processing video: {args.video_path}")
            
            # Load and process video
            analyzer.load_video(args.video_path)
            results = analyzer.process_video(frame_interval=args.frame_interval)
            
            # Determine output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.video_path).parent
            output_dir.mkdir(exist_ok=True)
            
            # Export results
            video_name = Path(args.video_path).stem
            pgn_path = output_dir / f"{video_name}.pgn"
            fen_path = output_dir / f"{video_name}.fen"
            
            analyzer.export_pgn(str(pgn_path))
            analyzer.export_fen(str(fen_path))
            
            # Print statistics
            stats = analyzer.get_processing_statistics()
            print(f"\nProcessing completed:")
            print(f"  Processing time: {stats['processing_time']:.2f}s")
            print(f"  Total moves: {stats['total_moves']}")
            print(f"  Flagged moves: {stats['flagged_moves']}")
            print(f"  Overall confidence: {stats.get('overall_confidence', 0):.2f}")
            print(f"  PGN exported to: {pgn_path}")
            print(f"  FEN exported to: {fen_path}")
            
        else:
            # Launch UI
            analyzer.launch_ui()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        analyzer.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
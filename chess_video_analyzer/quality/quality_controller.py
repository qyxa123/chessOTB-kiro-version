"""
Quality control and error handling for chess video analysis.

This module implements confidence-based quality control, flagging systems,
and manual correction interface hooks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from statistics import mean, stdev

from ..core.data_models import (
    Position, PieceType, Move, BoardState, GameState, 
    VideoMetadata, GameMetadata
)


class QualityFlag(Enum):
    """Types of quality flags that can be raised."""
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"
    INCONSISTENT_MOVE = "inconsistent_move"
    ILLEGAL_MOVE = "illegal_move"
    POOR_VIDEO_QUALITY = "poor_video_quality"
    BOARD_DETECTION_FAILURE = "board_detection_failure"
    PIECE_RECOGNITION_FAILURE = "piece_recognition_failure"
    TRACKING_LOSS = "tracking_loss"
    AMBIGUOUS_POSITION = "ambiguous_position"
    RAPID_CHANGES = "rapid_changes"
    OCCLUSION_DETECTED = "occlusion_detected"


@dataclass
class QualityIssue:
    """Represents a quality issue detected during analysis."""
    flag_type: QualityFlag
    confidence: float
    position: Optional[Position] = None
    move: Optional[Move] = None
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    suggested_action: str = ""
    
    def __post_init__(self):
        """Validate quality issue data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError(f"Invalid severity level: {self.severity}")


@dataclass
class QualityReport:
    """Comprehensive quality report for chess video analysis."""
    overall_confidence: float
    issues: List[QualityIssue] = field(default_factory=list)
    flagged_moves: List[Move] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate quality report data."""
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(f"Overall confidence must be between 0.0 and 1.0, got {self.overall_confidence}")
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """Get all critical quality issues."""
        return [issue for issue in self.issues if issue.severity == "critical"]
    
    def get_high_priority_issues(self) -> List[QualityIssue]:
        """Get high and critical priority issues."""
        return [issue for issue in self.issues if issue.severity in ["high", "critical"]]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.get_critical_issues()) > 0


class QualityController:
    """
    Controls quality assessment and flagging for chess video analysis.
    
    Implements confidence-based quality control, flagging systems for low-confidence
    detections, and provides hooks for manual correction interfaces.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 critical_confidence_threshold: float = 0.3,
                 consistency_window: int = 5):
        """
        Initialize the quality controller.
        
        Args:
            confidence_threshold: Minimum confidence for acceptable quality
            critical_confidence_threshold: Threshold below which issues are critical
            consistency_window: Number of frames to consider for consistency checks
        """
        self.confidence_threshold = confidence_threshold
        self.critical_confidence_threshold = critical_confidence_threshold
        self.consistency_window = consistency_window
        
        # Tracking for quality assessment
        self._confidence_history: List[float] = []
        self._board_state_history: List[BoardState] = []
        self._move_history: List[Move] = []
        self._quality_issues: List[QualityIssue] = []
        
        # Manual correction hooks
        self._correction_callbacks: Dict[str, Callable] = {}
        
        # Statistics tracking
        self._frame_count = 0
        self._low_confidence_count = 0
        self._flagged_move_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def assess_board_state_quality(self, board_state: BoardState, 
                                 frame_number: Optional[int] = None,
                                 timestamp: Optional[float] = None) -> List[QualityIssue]:
        """
        Assess the quality of a detected board state.
        
        Args:
            board_state: The board state to assess
            frame_number: Optional frame number for tracking
            timestamp: Optional timestamp for tracking
            
        Returns:
            List of quality issues found
            
        Requirements: 7.2
        """
        issues = []
        self._frame_count += 1
        
        # Track confidence history
        self._confidence_history.append(board_state.confidence)
        if len(self._confidence_history) > self.consistency_window * 2:
            self._confidence_history = self._confidence_history[-self.consistency_window * 2:]
        
        # Check confidence level
        if board_state.confidence < self.critical_confidence_threshold:
            self._low_confidence_count += 1
            issue = QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=board_state.confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Board state confidence ({board_state.confidence:.2f}) is critically low",
                severity="critical",
                suggested_action="Manual review required - consider re-recording this section"
            )
            issues.append(issue)
            
        elif board_state.confidence < self.confidence_threshold:
            self._low_confidence_count += 1
            issue = QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=board_state.confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Board state confidence ({board_state.confidence:.2f}) is below threshold",
                severity="high",
                suggested_action="Review detection - may need manual correction"
            )
            issues.append(issue)
        
        # Check for piece recognition issues
        piece_count = sum(1 for piece in board_state.squares.values() if piece is not None)
        if piece_count == 0:
            issue = QualityIssue(
                flag_type=QualityFlag.PIECE_RECOGNITION_FAILURE,
                confidence=board_state.confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description="No pieces detected on board",
                severity="critical",
                suggested_action="Check board detection and piece recognition settings"
            )
            issues.append(issue)
        elif piece_count < 4:  # Very few pieces detected
            issue = QualityIssue(
                flag_type=QualityFlag.PIECE_RECOGNITION_FAILURE,
                confidence=board_state.confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Very few pieces detected ({piece_count})",
                severity="high",
                suggested_action="Verify piece recognition accuracy"
            )
            issues.append(issue)
        
        # Check consistency with recent history
        if len(self._board_state_history) >= 2:
            consistency_issues = self._check_board_state_consistency(
                board_state, frame_number, timestamp
            )
            issues.extend(consistency_issues)
        
        # Update history
        self._board_state_history.append(board_state)
        if len(self._board_state_history) > self.consistency_window:
            self._board_state_history = self._board_state_history[-self.consistency_window:]
        
        # Store issues for reporting
        self._quality_issues.extend(issues)
        
        return issues
    
    def assess_move_quality(self, move: Move, 
                          previous_state: BoardState,
                          current_state: BoardState,
                          frame_number: Optional[int] = None,
                          timestamp: Optional[float] = None) -> List[QualityIssue]:
        """
        Assess the quality of a detected move.
        
        Args:
            move: The move to assess
            previous_state: Board state before the move
            current_state: Board state after the move
            frame_number: Optional frame number for tracking
            timestamp: Optional timestamp for tracking
            
        Returns:
            List of quality issues found
            
        Requirements: 7.2
        """
        issues = []
        
        # Calculate move confidence based on board state confidences
        move_confidence = min(previous_state.confidence, current_state.confidence)
        
        # Check if move is already flagged
        if move.is_flagged:
            issue = QualityIssue(
                flag_type=QualityFlag.ILLEGAL_MOVE,
                confidence=move_confidence,
                move=move,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Move is flagged: {move.flag_reason or 'Unknown reason'}",
                severity="high",
                suggested_action="Manual review required"
            )
            issues.append(issue)
            self._flagged_move_count += 1
        
        # Check move confidence
        if move_confidence < self.critical_confidence_threshold:
            issue = QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=move_confidence,
                move=move,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Move confidence ({move_confidence:.2f}) is critically low",
                severity="critical",
                suggested_action="Manual verification required"
            )
            issues.append(issue)
            
        elif move_confidence < self.confidence_threshold:
            issue = QualityIssue(
                flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                confidence=move_confidence,
                move=move,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Move confidence ({move_confidence:.2f}) is below threshold",
                severity="medium",
                suggested_action="Consider manual review"
            )
            issues.append(issue)
        
        # Check for inconsistent moves
        if len(self._move_history) > 0:
            consistency_issues = self._check_move_consistency(
                move, move_confidence, frame_number, timestamp
            )
            issues.extend(consistency_issues)
        
        # Update move history
        self._move_history.append(move)
        if len(self._move_history) > self.consistency_window:
            self._move_history = self._move_history[-self.consistency_window:]
        
        # Store issues for reporting
        self._quality_issues.extend(issues)
        
        return issues
    
    def assess_video_quality(self, video_metadata: VideoMetadata) -> List[QualityIssue]:
        """
        Assess the quality of the input video.
        
        Args:
            video_metadata: Metadata about the video
            
        Returns:
            List of quality issues found
            
        Requirements: 7.1
        """
        issues = []
        
        # Check resolution
        width, height = video_metadata.resolution
        min_resolution = 480
        recommended_resolution = 720
        
        if min(width, height) < min_resolution:
            issue = QualityIssue(
                flag_type=QualityFlag.POOR_VIDEO_QUALITY,
                confidence=0.3,
                description=f"Video resolution ({width}x{height}) is very low",
                severity="high",
                suggested_action="Use higher resolution video for better results"
            )
            issues.append(issue)
        elif min(width, height) < recommended_resolution:
            issue = QualityIssue(
                flag_type=QualityFlag.POOR_VIDEO_QUALITY,
                confidence=0.6,
                description=f"Video resolution ({width}x{height}) is below recommended",
                severity="medium",
                suggested_action="Consider using higher resolution video"
            )
            issues.append(issue)
        
        # Check frame rate
        if video_metadata.fps < 15:
            issue = QualityIssue(
                flag_type=QualityFlag.POOR_VIDEO_QUALITY,
                confidence=0.4,
                description=f"Frame rate ({video_metadata.fps} fps) is very low",
                severity="high",
                suggested_action="Use higher frame rate for smoother tracking"
            )
            issues.append(issue)
        elif video_metadata.fps < 24:
            issue = QualityIssue(
                flag_type=QualityFlag.POOR_VIDEO_QUALITY,
                confidence=0.7,
                description=f"Frame rate ({video_metadata.fps} fps) is below recommended",
                severity="medium",
                suggested_action="Consider using higher frame rate"
            )
            issues.append(issue)
        
        # Check duration
        if video_metadata.duration < 30:  # Very short video
            issue = QualityIssue(
                flag_type=QualityFlag.POOR_VIDEO_QUALITY,
                confidence=0.8,
                description=f"Video duration ({video_metadata.duration:.1f}s) is very short",
                severity="low",
                suggested_action="Ensure video captures complete game or game segment"
            )
            issues.append(issue)
        
        return issues
    
    def flag_for_manual_review(self, 
                             item: Any,
                             flag_type: QualityFlag,
                             reason: str,
                             confidence: float = 0.0,
                             severity: str = "medium") -> QualityIssue:
        """
        Flag an item for manual review.
        
        Args:
            item: The item to flag (Move, BoardState, etc.)
            flag_type: Type of quality flag
            reason: Reason for flagging
            confidence: Confidence level of the item
            severity: Severity level
            
        Returns:
            The created quality issue
            
        Requirements: 7.2, 7.4
        """
        # Extract relevant information from the item
        position = None
        move = None
        
        if isinstance(item, Move):
            move = item
            # Flag the move itself
            item.is_flagged = True
            item.flag_reason = reason
            # Add to move history if not already there
            if item not in self._move_history:
                self._move_history.append(item)
            self._flagged_move_count += 1
        elif hasattr(item, 'position'):
            position = item.position
        
        issue = QualityIssue(
            flag_type=flag_type,
            confidence=confidence,
            position=position,
            move=move,
            description=reason,
            severity=severity,
            suggested_action="Manual review and correction required"
        )
        
        self._quality_issues.append(issue)
        return issue
    
    def register_correction_callback(self, callback_name: str, callback: Callable) -> None:
        """
        Register a callback function for manual corrections.
        
        Args:
            callback_name: Name of the callback
            callback: Function to call for corrections
            
        Requirements: 7.4
        """
        self._correction_callbacks[callback_name] = callback
    
    def request_manual_correction(self, 
                                issue: QualityIssue,
                                callback_name: str = "default") -> Any:
        """
        Request manual correction for a quality issue.
        
        Args:
            issue: The quality issue requiring correction
            callback_name: Name of the correction callback to use
            
        Returns:
            Result from the correction callback
            
        Requirements: 7.4
        """
        if callback_name in self._correction_callbacks:
            callback = self._correction_callbacks[callback_name]
            try:
                return callback(issue)
            except Exception as e:
                self.logger.error(f"Manual correction callback failed: {e}")
                return None
        else:
            self.logger.warning(f"No correction callback registered for '{callback_name}'")
            return None
    
    def generate_quality_report(self) -> QualityReport:
        """
        Generate a comprehensive quality report.
        
        Returns:
            Quality report with all issues and statistics
            
        Requirements: 7.2
        """
        # Calculate overall confidence
        if self._confidence_history:
            overall_confidence = mean(self._confidence_history)
        else:
            overall_confidence = 0.0
        
        # Get flagged moves
        flagged_moves = [move for move in self._move_history if move.is_flagged]
        
        # Calculate statistics
        statistics = {
            "total_frames_processed": self._frame_count,
            "low_confidence_frames": self._low_confidence_count,
            "flagged_moves": self._flagged_move_count,
            "average_confidence": overall_confidence,
            "confidence_std": stdev(self._confidence_history) if len(self._confidence_history) > 1 else 0.0,
            "critical_issues": len([i for i in self._quality_issues if i.severity == "critical"]),
            "high_priority_issues": len([i for i in self._quality_issues if i.severity == "high"]),
            "total_issues": len(self._quality_issues)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistics)
        
        return QualityReport(
            overall_confidence=overall_confidence,
            issues=self._quality_issues.copy(),
            flagged_moves=flagged_moves,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _check_board_state_consistency(self, 
                                     current_state: BoardState,
                                     frame_number: Optional[int],
                                     timestamp: Optional[float]) -> List[QualityIssue]:
        """Check consistency of board state with recent history."""
        issues = []
        
        if len(self._board_state_history) < 2:
            return issues
        
        previous_state = self._board_state_history[-1]
        
        # Check for rapid changes in piece count
        prev_piece_count = sum(1 for piece in previous_state.squares.values() if piece is not None)
        curr_piece_count = sum(1 for piece in current_state.squares.values() if piece is not None)
        
        piece_count_change = abs(curr_piece_count - prev_piece_count)
        if piece_count_change > 3:  # More than 3 pieces changed at once
            issue = QualityIssue(
                flag_type=QualityFlag.RAPID_CHANGES,
                confidence=current_state.confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description=f"Rapid change in piece count: {piece_count_change} pieces",
                severity="high",
                suggested_action="Check for detection errors or multiple moves"
            )
            issues.append(issue)
        
        # Check confidence consistency
        if len(self._confidence_history) >= 3:
            recent_confidences = self._confidence_history[-3:]
            if all(c < self.confidence_threshold for c in recent_confidences):
                issue = QualityIssue(
                    flag_type=QualityFlag.LOW_CONFIDENCE_DETECTION,
                    confidence=current_state.confidence,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    description="Consistently low confidence over multiple frames",
                    severity="high",
                    suggested_action="Check video quality and lighting conditions"
                )
                issues.append(issue)
        
        return issues
    
    def _check_move_consistency(self, 
                              move: Move,
                              move_confidence: float,
                              frame_number: Optional[int],
                              timestamp: Optional[float]) -> List[QualityIssue]:
        """Check consistency of move with recent history."""
        issues = []
        
        if len(self._move_history) == 0:
            return issues
        
        recent_move = self._move_history[-1]
        
        # Check for impossible rapid moves (same piece moving multiple times quickly)
        if (move.piece == recent_move.piece and 
            move.from_square == recent_move.to_square):
            # This could be normal (piece moving back) or an error
            # Flag for review if confidence is low
            if move_confidence < self.confidence_threshold:
                issue = QualityIssue(
                    flag_type=QualityFlag.INCONSISTENT_MOVE,
                    confidence=move_confidence,
                    move=move,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    description="Same piece moving back immediately with low confidence",
                    severity="medium",
                    suggested_action="Verify move sequence accuracy"
                )
                issues.append(issue)
        
        return issues
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality statistics."""
        recommendations = []
        
        # Confidence-based recommendations
        if statistics["average_confidence"] < 0.5:
            recommendations.append("Overall confidence is very low - consider improving video quality or lighting")
        elif statistics["average_confidence"] < 0.7:
            recommendations.append("Consider improving video quality for better detection accuracy")
        
        # Issue-based recommendations
        if statistics["critical_issues"] > 0:
            recommendations.append("Critical issues detected - manual review required before using results")
        
        if statistics["flagged_moves"] > statistics["total_frames_processed"] * 0.1:
            recommendations.append("High number of flagged moves - verify game recording quality")
        
        if statistics["low_confidence_frames"] > statistics["total_frames_processed"] * 0.3:
            recommendations.append("Many low-confidence detections - check lighting and camera stability")
        
        # General recommendations
        if not recommendations:
            if statistics["average_confidence"] > 0.8:
                recommendations.append("Good quality analysis - results should be reliable")
            else:
                recommendations.append("Moderate quality analysis - spot check recommended")
        
        return recommendations
    
    def clear_history(self) -> None:
        """Clear all tracking history and reset counters."""
        self._confidence_history.clear()
        self._board_state_history.clear()
        self._move_history.clear()
        self._quality_issues.clear()
        self._frame_count = 0
        self._low_confidence_count = 0
        self._flagged_move_count = 0
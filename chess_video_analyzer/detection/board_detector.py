"""
Chess board detection module using computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging
from dataclasses import dataclass

from ..core.data_models import Position, BoardRegion, SquareGrid, Orientation


class BoardDetectionError(Exception):
    """Base exception for board detection errors."""
    pass


class BoardNotFoundError(BoardDetectionError):
    """Raised when no chess board is detected in the frame."""
    pass


@dataclass
class DetectionParams:
    """Parameters for board detection algorithms."""
    # Hough line detection parameters
    hough_threshold: int = 50
    min_line_length: int = 100
    max_line_gap: int = 10
    
    # Corner detection parameters
    corner_quality_level: float = 0.01
    corner_min_distance: int = 10
    corner_block_size: int = 3
    
    # Board validation parameters
    min_board_area: int = 10000
    max_aspect_ratio_deviation: float = 0.3
    
    # Confidence thresholds
    min_confidence: float = 0.3
    line_intersection_tolerance: float = 10.0
    
    # Tracking parameters
    max_corner_movement: float = 50.0  # Max pixel movement between frames
    tracking_smoothing_factor: float = 0.7  # Smoothing for corner tracking
    occlusion_threshold: float = 0.4  # Minimum visible area to maintain tracking
    interpolation_max_frames: int = 5  # Max frames to interpolate
    
    # Camera angle adaptation
    angle_change_threshold: float = 5.0  # Degrees
    perspective_adaptation_rate: float = 0.1


@dataclass 
class TrackingState:
    """State information for board tracking."""
    last_valid_region: Optional[BoardRegion] = None
    frames_since_detection: int = 0
    corner_velocities: List[Tuple[float, float]] = None
    perspective_matrix: Optional[np.ndarray] = None
    stable_frame_count: int = 0
    
    def __post_init__(self):
        if self.corner_velocities is None:
            self.corner_velocities = [(0.0, 0.0)] * 4


class BoardDetector:
    """
    Detects chess board position and orientation in video frames using computer vision.
    
    Uses a combination of Hough line detection, corner detection, and perspective
    transformation to identify the chess board and extract square coordinates.
    """
    
    def __init__(self, params: Optional[DetectionParams] = None):
        """
        Initialize the board detector.
        
        Args:
            params: Detection parameters, uses defaults if None
        """
        self.params = params or DetectionParams()
        self._previous_board_region: Optional[BoardRegion] = None
        self._tracking_history: List[BoardRegion] = []
        self._max_history_size = 10
        self._tracking_state = TrackingState()
        self._optical_flow_tracker: Optional[cv2.TrackerKCF] = None
        self._template_matcher: Optional[np.ndarray] = None
        
    def detect_board(self, frame: np.ndarray) -> BoardRegion:
        """
        Detect the chess board in a video frame with robust tracking and interpolation.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            BoardRegion: Detected board region with corners and confidence
            
        Raises:
            BoardNotFoundError: If no valid chess board is detected
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try tracking first if we have a previous detection
        board_region = None
        if self._tracking_state.last_valid_region is not None:
            board_region = self._track_board(gray, frame)
        
        # If tracking failed, try detection
        if board_region is None:
            board_region = self._detect_board_primary(gray, frame)
        
        if board_region is None:
            # Try fallback methods
            board_region = self._detect_board_fallback(gray, frame)
        
        if board_region is None:
            # Try interpolation if we have previous detections
            board_region = self._interpolate_from_history()
        
        if board_region is None:
            self._tracking_state.frames_since_detection += 1
            if self._tracking_state.frames_since_detection > self.params.interpolation_max_frames:
                raise BoardNotFoundError("No chess board detected in frame")
            # Return last known position with very low confidence
            if self._tracking_state.last_valid_region is not None:
                board_region = BoardRegion(
                    corners=self._tracking_state.last_valid_region.corners,
                    confidence=0.1,
                    orientation=self._tracking_state.last_valid_region.orientation
                )
            else:
                raise BoardNotFoundError("No chess board detected in frame")
        
        if board_region is None:
            raise BoardNotFoundError("No chess board detected in frame")
        
        # Update tracking state
        self._update_tracking_state(board_region, gray)
        
        # Update tracking history
        self._update_tracking_history(board_region)
        self._previous_board_region = board_region
        
        return board_region
    
    def _detect_board_primary(self, gray: np.ndarray, frame: np.ndarray) -> Optional[BoardRegion]:
        """Primary board detection using Hough lines and corner detection."""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Check if there are enough edges to potentially be a chess board
            edge_count = np.sum(edges > 0)
            if edge_count < 1000:  # Minimum edge count for a chess board
                return None
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.params.hough_threshold,
                minLineLength=self.params.min_line_length,
                maxLineGap=self.params.max_line_gap
            )
            
            if lines is None or len(lines) < 4:
                return None
            
            # Find board corners from line intersections
            corners = self._find_board_corners_from_lines(lines, gray.shape)
            
            if corners is None:
                return None
            
            # Validate the detected board region
            confidence = self._calculate_confidence(corners, edges, gray.shape)
            
            if confidence < self.params.min_confidence:
                return None
            
            # Determine board orientation
            orientation = self._determine_orientation(corners, frame)
            
            return BoardRegion(
                corners=corners,
                confidence=confidence,
                orientation=orientation
            )
            
        except Exception as e:
            logging.warning(f"Primary board detection failed: {e}")
            return None
    
    def _detect_board_fallback(self, gray: np.ndarray, frame: np.ndarray) -> Optional[BoardRegion]:
        """Fallback board detection using corner detection."""
        try:
            # Check if there's enough variation in the image
            if np.std(gray) < 10:  # Very low variation suggests blank/uniform image
                return None
                
            # Use Harris corner detection
            corners_harris = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners_harris = cv2.dilate(corners_harris, None)
            
            # Find corner points
            corner_points = np.where(corners_harris > 0.01 * corners_harris.max())
            
            if len(corner_points[0]) < 4:
                return None
            
            # Convert to coordinate pairs
            points = list(zip(corner_points[1], corner_points[0]))
            
            # Find the four corners that form the largest quadrilateral
            board_corners = self._find_largest_quadrilateral(points)
            
            if board_corners is None:
                return None
            
            # Calculate confidence based on corner quality
            confidence = min(0.8, len(points) / 100.0)  # Simple heuristic
            
            # Determine orientation
            orientation = self._determine_orientation(board_corners, frame)
            
            return BoardRegion(
                corners=board_corners,
                confidence=confidence,
                orientation=orientation
            )
            
        except Exception as e:
            logging.warning(f"Fallback board detection failed: {e}")
            return None
    
    def _find_board_corners_from_lines(self, lines: np.ndarray, image_shape: Tuple[int, int]) -> Optional[List[Tuple[float, float]]]:
        """Find board corners from detected lines."""
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            if angle < 45 or angle > 135:  # Horizontal-ish
                horizontal_lines.append(line[0])
            else:  # Vertical-ish
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Find intersections between horizontal and vertical lines
        intersections = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within image bounds
                    if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                        intersections.append((x, y))
        
        if len(intersections) < 4:
            return None
        
        # Find the four corners that form the largest quadrilateral
        return self._find_largest_quadrilateral(intersections)
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def _find_largest_quadrilateral(self, points: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """Find the four points that form the largest quadrilateral."""
        if len(points) < 4:
            return None
        
        # Convert to numpy array for easier processing
        points_array = np.array(points)
        
        # Find convex hull
        hull = cv2.convexHull(points_array.astype(np.float32))
        
        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # If we don't get exactly 4 points, try different epsilon values
        for epsilon_factor in [0.01, 0.03, 0.05]:
            if len(approx) == 4:
                break
            epsilon = epsilon_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) != 4:
            # Fallback: find 4 corner points manually
            return self._find_corner_points(points)
        
        # Convert back to list of tuples
        corners = [(float(point[0][0]), float(point[0][1])) for point in approx]
        
        # Order corners clockwise starting from top-left
        return self._order_corners(corners)
    
    def _find_corner_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find 4 corner points from a list of points."""
        points_array = np.array(points)
        
        # Find extreme points
        top_left = points_array[np.argmin(points_array[:, 0] + points_array[:, 1])]
        top_right = points_array[np.argmax(points_array[:, 0] - points_array[:, 1])]
        bottom_right = points_array[np.argmax(points_array[:, 0] + points_array[:, 1])]
        bottom_left = points_array[np.argmin(points_array[:, 0] - points_array[:, 1])]
        
        corners = [
            (float(top_left[0]), float(top_left[1])),
            (float(top_right[0]), float(top_right[1])),
            (float(bottom_right[0]), float(bottom_right[1])),
            (float(bottom_left[0]), float(bottom_left[1]))
        ]
        
        return corners
    
    def _order_corners(self, corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Order corners clockwise starting from top-left."""
        # Convert to numpy array
        corners_array = np.array(corners)
        
        # Find center point
        center = np.mean(corners_array, axis=0)
        
        # Calculate angles from center
        angles = []
        for corner in corners_array:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        
        # Reorder corners
        ordered_corners = [corners[i] for i in sorted_indices]
        
        return ordered_corners
    
    def _calculate_confidence(self, corners: List[Tuple[float, float]], edges: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate confidence score for detected board region."""
        try:
            # Calculate area of the quadrilateral
            corners_array = np.array(corners, dtype=np.float32)
            area = cv2.contourArea(corners_array)
            
            # Normalize by image area
            image_area = image_shape[0] * image_shape[1]
            area_ratio = area / image_area
            
            # Check if area is reasonable for a chess board
            if area < self.params.min_board_area:
                return 0.0
            
            # Check aspect ratio (should be close to 1:1 for a square board)
            rect = cv2.minAreaRect(corners_array)
            width, height = rect[1]
            if width == 0 or height == 0:
                return 0.0
            
            aspect_ratio = max(width, height) / min(width, height)
            aspect_ratio_score = max(0, 1 - abs(aspect_ratio - 1) / self.params.max_aspect_ratio_deviation)
            
            # Calculate edge density within the region
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corners_array.astype(np.int32)], 255)
            
            edges_in_region = cv2.bitwise_and(edges, mask)
            edge_density = np.sum(edges_in_region > 0) / area if area > 0 else 0
            edge_score = min(1.0, edge_density / 0.1)  # Normalize edge density
            
            # Add penalty for very low edge content (likely blank regions)
            total_edges = np.sum(edges > 0)
            if total_edges < 500:  # Very few edges in entire image
                edge_score *= 0.3  # Heavy penalty
            
            # Combine scores
            confidence = (area_ratio * 0.2 + aspect_ratio_score * 0.3 + edge_score * 0.5)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logging.warning(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _determine_orientation(self, corners: List[Tuple[float, float]], frame: np.ndarray) -> Orientation:
        """Determine board orientation by analyzing corner regions."""
        try:
            # For now, use a simple heuristic based on corner positions
            # In a real implementation, this would analyze the actual pieces or board features
            
            # Calculate center of the board
            center_x = sum(corner[0] for corner in corners) / 4
            center_y = sum(corner[1] for corner in corners) / 4
            
            # Find the bottom corner (highest y value)
            bottom_corner = max(corners, key=lambda c: c[1])
            
            # Determine orientation based on which corner is at the bottom
            if bottom_corner[0] < center_x:
                return Orientation.WHITE_LEFT
            else:
                return Orientation.WHITE_RIGHT
            
        except Exception as e:
            logging.warning(f"Orientation determination failed: {e}")
            return Orientation.WHITE_BOTTOM  # Default orientation
    
    def _interpolate_from_history(self) -> Optional[BoardRegion]:
        """Interpolate board position from tracking history with enhanced motion prediction."""
        if len(self._tracking_history) < 2:
            return None
        
        try:
            # Use motion prediction if we have velocity information
            if (self._tracking_state.last_valid_region is not None and 
                self._tracking_state.frames_since_detection < self.params.interpolation_max_frames):
                
                # Enhanced motion prediction with acceleration consideration
                predicted_corners = []
                for i, (corner, velocity) in enumerate(zip(
                    self._tracking_state.last_valid_region.corners,
                    self._tracking_state.corner_velocities
                )):
                    # Apply velocity with damping factor to account for deceleration
                    damping_factor = 0.95 ** self._tracking_state.frames_since_detection
                    predicted_x = corner[0] + velocity[0] * self._tracking_state.frames_since_detection * damping_factor
                    predicted_y = corner[1] + velocity[1] * self._tracking_state.frames_since_detection * damping_factor
                    predicted_corners.append((predicted_x, predicted_y))
                
                # Validate predicted corners are reasonable
                if self._validate_predicted_corners(predicted_corners):
                    # Calculate confidence based on time since last detection and motion consistency
                    confidence_decay = 0.9 ** self._tracking_state.frames_since_detection
                    motion_consistency = self._calculate_motion_consistency()
                    interpolated_confidence = (self._tracking_state.last_valid_region.confidence * 
                                             confidence_decay * motion_consistency * 0.8)  # Mark as interpolated
                    
                    return BoardRegion(
                        corners=predicted_corners,
                        confidence=interpolated_confidence,
                        orientation=self._tracking_state.last_valid_region.orientation
                    )
            
            # Enhanced fallback interpolation using multiple history points
            if len(self._tracking_history) >= 3:
                # Use weighted average of recent detections
                weights = [0.5, 0.3, 0.2]  # More weight to recent detections
                interpolated_corners = []
                
                for corner_idx in range(4):
                    weighted_x = sum(w * self._tracking_history[-(i+1)].corners[corner_idx][0] 
                                   for i, w in enumerate(weights))
                    weighted_y = sum(w * self._tracking_history[-(i+1)].corners[corner_idx][1] 
                                   for i, w in enumerate(weights))
                    interpolated_corners.append((weighted_x, weighted_y))
                
                # Use confidence from most recent detection with decay
                recent_confidence = self._tracking_history[-1].confidence * 0.7
                
                return BoardRegion(
                    corners=interpolated_corners,
                    confidence=recent_confidence,
                    orientation=self._tracking_history[-1].orientation
                )
            
            # Simple fallback to most recent detection
            recent_region = self._tracking_history[-1]
            interpolated_confidence = recent_region.confidence * 0.6
            
            return BoardRegion(
                corners=recent_region.corners,
                confidence=interpolated_confidence,
                orientation=recent_region.orientation
            )
            
        except Exception as e:
            logging.warning(f"Interpolation failed: {e}")
            return None
    
    def _track_board(self, gray: np.ndarray, frame: np.ndarray) -> Optional[BoardRegion]:
        """Track the board using optical flow and template matching."""
        try:
            last_region = self._tracking_state.last_valid_region
            if last_region is None:
                return None
            
            # Try optical flow tracking of corner points
            tracked_region = self._track_corners_optical_flow(gray, last_region)
            
            if tracked_region is not None:
                # Validate tracking result
                if self._validate_tracking_result(tracked_region, last_region):
                    return tracked_region
            
            # Try template matching as fallback
            template_region = self._track_template_matching(gray, last_region)
            
            if template_region is not None:
                if self._validate_tracking_result(template_region, last_region):
                    return template_region
            
            return None
            
        except Exception as e:
            logging.warning(f"Board tracking failed: {e}")
            return None

    def _validate_predicted_corners(self, corners: List[Tuple[float, float]]) -> bool:
        """Validate that predicted corners form a reasonable quadrilateral."""
        try:
            if len(corners) != 4:
                return False
            
            # Check that corners are within reasonable bounds (allow some margin outside frame)
            # We don't have frame dimensions here, so use a reasonable assumption
            for x, y in corners:
                if x < -100 or x > 2000 or y < -100 or y > 2000:  # Very loose bounds
                    return False
            
            # Check that the quadrilateral has reasonable area
            corners_array = np.array(corners, dtype=np.float32)
            area = cv2.contourArea(corners_array)
            
            if area < 1000:  # Too small
                return False
            
            # Check aspect ratio isn't too distorted
            rect = cv2.minAreaRect(corners_array)
            width, height = rect[1]
            if width == 0 or height == 0:
                return False
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3.0:  # Too distorted
                return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Corner validation failed: {e}")
            return False
    
    def _calculate_motion_consistency(self) -> float:
        """Calculate how consistent the motion has been (for confidence adjustment)."""
        try:
            if len(self._tracking_history) < 3:
                return 1.0
            
            # Calculate velocity changes between recent frames
            velocity_changes = []
            for i in range(len(self._tracking_history) - 2):
                region1 = self._tracking_history[i]
                region2 = self._tracking_history[i + 1]
                region3 = self._tracking_history[i + 2]
                
                # Calculate velocities between consecutive frames
                vel1 = [(c2[0] - c1[0], c2[1] - c1[1]) for c1, c2 in zip(region1.corners, region2.corners)]
                vel2 = [(c2[0] - c1[0], c2[1] - c1[1]) for c1, c2 in zip(region2.corners, region3.corners)]
                
                # Calculate velocity change magnitude
                for v1, v2 in zip(vel1, vel2):
                    change = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                    velocity_changes.append(change)
            
            if not velocity_changes:
                return 1.0
            
            # Lower consistency score for higher velocity changes
            avg_change = np.mean(velocity_changes)
            consistency = max(0.3, 1.0 - avg_change / 50.0)  # Normalize by expected max change
            
            return consistency
            
        except Exception as e:
            logging.warning(f"Motion consistency calculation failed: {e}")
            return 1.0
        """Track the board using optical flow and template matching."""
        try:
            last_region = self._tracking_state.last_valid_region
            if last_region is None:
                return None
            
            # Try optical flow tracking of corner points
            tracked_region = self._track_corners_optical_flow(gray, last_region)
            
            if tracked_region is not None:
                # Validate tracking result
                if self._validate_tracking_result(tracked_region, last_region):
                    return tracked_region
            
            # Try template matching as fallback
            template_region = self._track_template_matching(gray, last_region)
            
            if template_region is not None:
                if self._validate_tracking_result(template_region, last_region):
                    return template_region
            
            return None
            
        except Exception as e:
            logging.warning(f"Board tracking failed: {e}")
            return None
    
    def _track_corners_optical_flow(self, gray: np.ndarray, last_region: BoardRegion) -> Optional[BoardRegion]:
        """Track board corners using optical flow."""
        try:
            if not hasattr(self, '_previous_gray') or self._previous_gray is None:
                self._previous_gray = gray.copy()
                return None
            
            # Convert corners to the format expected by optical flow
            old_corners = np.array(last_region.corners, dtype=np.float32).reshape(-1, 1, 2)
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Calculate optical flow
            new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                self._previous_gray, gray, old_corners, None, **lk_params
            )
            
            # Check if all corners were tracked successfully
            if status is None or not all(status.flatten()):
                return None
            
            # Convert back to list of tuples
            tracked_corners = [(float(corner[0][0]), float(corner[0][1])) for corner in new_corners]
            
            # Calculate confidence based on tracking error
            avg_error = np.mean(error) if error is not None else 0
            confidence = max(0.3, min(0.9, 1.0 - avg_error / 50.0))
            
            # Update previous frame
            self._previous_gray = gray.copy()
            
            return BoardRegion(
                corners=tracked_corners,
                confidence=confidence,
                orientation=last_region.orientation
            )
            
        except Exception as e:
            logging.warning(f"Optical flow tracking failed: {e}")
            return None
    
    def _track_template_matching(self, gray: np.ndarray, last_region: BoardRegion) -> Optional[BoardRegion]:
        """Track board using template matching."""
        try:
            if self._template_matcher is None:
                return None
            
            # Perform template matching
            result = cv2.matchTemplate(gray, self._template_matcher, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val < 0.6:  # Low correlation
                return None
            
            # Calculate new corner positions based on template match
            template_h, template_w = self._template_matcher.shape
            top_left = max_loc
            
            # Estimate corners based on template position
            # This is a simplified approach - in practice, you'd need more sophisticated mapping
            scale_x = template_w / 200  # Assume template is roughly 200x200
            scale_y = template_h / 200
            
            corners = [
                (float(top_left[0]), float(top_left[1])),
                (float(top_left[0] + template_w), float(top_left[1])),
                (float(top_left[0] + template_w), float(top_left[1] + template_h)),
                (float(top_left[0]), float(top_left[1] + template_h))
            ]
            
            return BoardRegion(
                corners=corners,
                confidence=float(max_val * 0.8),  # Reduce confidence for template matching
                orientation=last_region.orientation
            )
            
        except Exception as e:
            logging.warning(f"Template matching failed: {e}")
            return None
    
    def _validate_tracking_result(self, tracked_region: BoardRegion, last_region: BoardRegion) -> bool:
        """Validate that tracking result is reasonable."""
        try:
            # Check corner movement distance
            for i, (new_corner, old_corner) in enumerate(zip(tracked_region.corners, last_region.corners)):
                distance = np.sqrt((new_corner[0] - old_corner[0])**2 + (new_corner[1] - old_corner[1])**2)
                if distance > self.params.max_corner_movement:
                    return False
            
            # Check that the quadrilateral is still reasonable
            corners_array = np.array(tracked_region.corners, dtype=np.float32)
            area = cv2.contourArea(corners_array)
            
            if area < self.params.min_board_area * 0.5:  # Allow some shrinkage
                return False
            
            # Check aspect ratio
            rect = cv2.minAreaRect(corners_array)
            width, height = rect[1]
            if width == 0 or height == 0:
                return False
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 2.0:  # Too distorted
                return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Tracking validation failed: {e}")
            return False
    
    def _update_tracking_state(self, board_region: BoardRegion, gray: np.ndarray):
        """Update the tracking state with new detection."""
        try:
            # Reset frames since detection counter for successful detections
            if board_region.confidence > 0.2:  # Only reset for real detections, not interpolations
                self._tracking_state.frames_since_detection = 0
            
            # Update corner velocities if we have a previous region
            if self._tracking_state.last_valid_region is not None:
                new_velocities = []
                for i, (new_corner, old_corner) in enumerate(zip(
                    board_region.corners, 
                    self._tracking_state.last_valid_region.corners
                )):
                    velocity_x = new_corner[0] - old_corner[0]
                    velocity_y = new_corner[1] - old_corner[1]
                    
                    # Apply smoothing
                    if i < len(self._tracking_state.corner_velocities):
                        old_vel_x, old_vel_y = self._tracking_state.corner_velocities[i]
                        smoothed_vel_x = (self.params.tracking_smoothing_factor * old_vel_x + 
                                        (1 - self.params.tracking_smoothing_factor) * velocity_x)
                        smoothed_vel_y = (self.params.tracking_smoothing_factor * old_vel_y + 
                                        (1 - self.params.tracking_smoothing_factor) * velocity_y)
                        new_velocities.append((smoothed_vel_x, smoothed_vel_y))
                    else:
                        new_velocities.append((velocity_x, velocity_y))
                
                self._tracking_state.corner_velocities = new_velocities
            
            # Update tracking state only for real detections
            if board_region.confidence > 0.2:
                self._tracking_state.last_valid_region = board_region
            
            # Update template for template matching
            if board_region.confidence > 0.7:  # Only use high-confidence detections
                self._update_template(gray, board_region)
            
            # Update stability counter
            if board_region.confidence > 0.8:
                self._tracking_state.stable_frame_count += 1
            else:
                self._tracking_state.stable_frame_count = 0
                
        except Exception as e:
            logging.warning(f"Tracking state update failed: {e}")
    
    def _update_template(self, gray: np.ndarray, board_region: BoardRegion):
        """Update the template for template matching."""
        try:
            # Extract board region as template
            corners_array = np.array(board_region.corners, dtype=np.float32)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(corners_array.astype(np.int32))
            
            # Extract template with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            template = gray[y1:y2, x1:x2]
            
            if template.size > 0:
                # Resize to standard size for consistency
                template_size = (200, 200)
                self._template_matcher = cv2.resize(template, template_size)
                
        except Exception as e:
            logging.warning(f"Template update failed: {e}")
    
    def handle_partial_occlusion(self, frame: np.ndarray, board_region: BoardRegion) -> BoardRegion:
        """
        Handle cases where the chess board is partially occluded.
        
        Args:
            frame: Input video frame
            board_region: Currently detected board region
            
        Returns:
            BoardRegion: Adjusted board region accounting for occlusion
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create mask for the board region
            corners_array = np.array(board_region.corners, dtype=np.int32)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corners_array], 255)
            
            # Calculate visible area ratio more accurately
            visible_pixels = np.sum(mask > 0)
            
            # Calculate total pixels in the board region for proper ratio
            total_pixels = visible_pixels  # Start with visible pixels
            
            # Create a clean mask without occlusions to get true board area
            clean_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(clean_mask, [corners_array], 255)
            total_pixels = np.sum(clean_mask > 0)
            
            if total_pixels == 0:
                return board_region
            
            visibility_ratio = min(1.0, visible_pixels / total_pixels)
            
            # Detect occlusion by analyzing the board region
            board_gray = cv2.bitwise_and(gray, mask)
            occlusion_detected = self._detect_occlusion_in_region(board_gray, mask)
            
            # Adjust confidence based on visibility and occlusion detection
            if occlusion_detected:
                adjusted_confidence = board_region.confidence * visibility_ratio * 0.8
            else:
                adjusted_confidence = board_region.confidence * visibility_ratio
            
            # Ensure confidence doesn't increase due to calculation errors
            adjusted_confidence = min(adjusted_confidence, board_region.confidence)
            
            # If visibility is too low, try to reconstruct from visible parts
            if visibility_ratio < self.params.occlusion_threshold:
                # Use tracking history to estimate occluded parts
                reconstructed_region = self._reconstruct_from_visible_parts(
                    gray, board_region, mask
                )
                if reconstructed_region is not None:
                    return reconstructed_region
            
            return BoardRegion(
                corners=board_region.corners,
                confidence=adjusted_confidence,
                orientation=board_region.orientation
            )
            
        except Exception as e:
            logging.warning(f"Occlusion handling failed: {e}")
            return board_region
    
    def _detect_occlusion_in_region(self, board_gray: np.ndarray, mask: np.ndarray) -> bool:
        """
        Detect if there's occlusion within the board region.
        
        Args:
            board_gray: Grayscale image of the board region
            mask: Mask of the board region
            
        Returns:
            bool: True if occlusion is detected
        """
        try:
            # Calculate statistics of the board region
            masked_pixels = board_gray[mask > 0]
            
            if len(masked_pixels) == 0:
                return True  # No visible pixels suggests complete occlusion
            
            # Check for large dark areas that might indicate occlusion
            dark_threshold = 50  # Pixels darker than this might be occlusions
            dark_pixels = np.sum(masked_pixels < dark_threshold)
            total_pixels = len(masked_pixels)
            
            dark_ratio = dark_pixels / total_pixels
            
            # If more than 30% of the board is very dark, consider it occluded
            return dark_ratio > 0.3
            
        except Exception as e:
            logging.warning(f"Occlusion detection failed: {e}")
            return False

    def _reconstruct_from_visible_parts(self, gray: np.ndarray, board_region: BoardRegion, 
                                      mask: np.ndarray) -> Optional[BoardRegion]:
        """Reconstruct board region from visible parts."""
        try:
            # Find visible corners
            visible_corners = []
            for corner in board_region.corners:
                x, y = int(corner[0]), int(corner[1])
                if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and 
                    mask[y, x] > 0):
                    visible_corners.append(corner)
            
            if len(visible_corners) < 2:
                return None
            
            # Use geometric constraints to estimate missing corners
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            if len(visible_corners) == 2:
                # Try to estimate the other two corners based on chess board geometry
                estimated_corners = self._estimate_missing_corners(visible_corners)
                if estimated_corners is not None:
                    all_corners = visible_corners + estimated_corners
                    return BoardRegion(
                        corners=all_corners,
                        confidence=board_region.confidence * 0.6,  # Reduced confidence
                        orientation=board_region.orientation
                    )
            
            return None
            
        except Exception as e:
            logging.warning(f"Reconstruction from visible parts failed: {e}")
            return None
    
    def _estimate_missing_corners(self, visible_corners: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """Estimate missing corners based on visible ones."""
        # This is a placeholder for more sophisticated corner estimation
        # In practice, you'd use geometric constraints and chess board properties
        return None
    
    def adapt_to_camera_angle_changes(self, frame: np.ndarray, board_region: BoardRegion) -> BoardRegion:
        """
        Adapt detection to gradual camera angle changes with enhanced stability.
        
        Args:
            frame: Input video frame
            board_region: Currently detected board region
            
        Returns:
            BoardRegion: Adjusted board region for camera angle changes
        """
        try:
            # Calculate perspective change from previous frame
            if self._tracking_state.perspective_matrix is not None:
                # Detect if there's been a significant perspective change
                current_corners = np.array(board_region.corners, dtype=np.float32)
                
                # Apply previous perspective matrix to see expected corners
                try:
                    expected_corners = cv2.perspectiveTransform(
                        current_corners.reshape(-1, 1, 2),
                        self._tracking_state.perspective_matrix
                    ).reshape(-1, 2)
                except cv2.error:
                    # If perspective transform fails, reinitialize
                    self._tracking_state.perspective_matrix = None
                    return self._initialize_perspective_matrix(board_region)
                
                # Calculate deviation with enhanced metrics
                deviations = []
                for current, expected in zip(current_corners, expected_corners):
                    deviation = np.sqrt(np.sum((current - expected) ** 2))
                    deviations.append(deviation)
                
                avg_deviation = np.mean(deviations)
                max_deviation = np.max(deviations)
                
                # Enhanced angle change detection
                angle_change_detected = (avg_deviation > self.params.angle_change_threshold or 
                                       max_deviation > self.params.angle_change_threshold * 2)
                
                # If deviation is significant, update perspective matrix gradually
                if angle_change_detected:
                    # Calculate new perspective matrix with error handling
                    try:
                        new_matrix = cv2.getPerspectiveTransform(
                            expected_corners, current_corners
                        )
                        
                        # Validate the new matrix
                        if self._validate_perspective_matrix(new_matrix):
                            # Blend with previous matrix for smooth adaptation
                            adaptation_rate = self._calculate_adaptive_rate(avg_deviation)
                            self._tracking_state.perspective_matrix = (
                                adaptation_rate * new_matrix +
                                (1 - adaptation_rate) * self._tracking_state.perspective_matrix
                            )
                        else:
                            # If new matrix is invalid, reduce confidence
                            board_region = BoardRegion(
                                corners=board_region.corners,
                                confidence=board_region.confidence * 0.8,
                                orientation=board_region.orientation
                            )
                    except cv2.error:
                        # If perspective calculation fails, maintain current state
                        logging.warning("Perspective matrix calculation failed")
                        
                # Update stability tracking
                if avg_deviation < self.params.angle_change_threshold * 0.5:
                    self._tracking_state.stable_frame_count += 1
                else:
                    self._tracking_state.stable_frame_count = 0
            else:
                # Initialize perspective matrix
                board_region = self._initialize_perspective_matrix(board_region)
            
            return board_region
            
        except Exception as e:
            logging.warning(f"Camera angle adaptation failed: {e}")
            return board_region
    
    def _initialize_perspective_matrix(self, board_region: BoardRegion) -> BoardRegion:
        """Initialize the perspective matrix for a board region."""
        try:
            corners = np.array(board_region.corners, dtype=np.float32)
            standard_corners = np.array([
                [0, 0], [1, 0], [1, 1], [0, 1]
            ], dtype=np.float32)
            
            self._tracking_state.perspective_matrix = cv2.getPerspectiveTransform(
                standard_corners, corners
            )
            return board_region
        except cv2.error:
            logging.warning("Failed to initialize perspective matrix")
            return board_region
    
    def _validate_perspective_matrix(self, matrix: np.ndarray) -> bool:
        """Validate that a perspective matrix is reasonable."""
        try:
            # Check for NaN or infinite values
            if not np.isfinite(matrix).all():
                return False
            
            # Check determinant to ensure matrix is invertible
            det = np.linalg.det(matrix[:2, :2])  # Check 2x2 submatrix
            if abs(det) < 1e-6:
                return False
            
            # Check that matrix values are within reasonable bounds
            if np.max(np.abs(matrix)) > 1000:
                return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Perspective matrix validation failed: {e}")
            return False
    
    def _calculate_adaptive_rate(self, deviation: float) -> float:
        """Calculate adaptive rate based on deviation magnitude."""
        # Higher deviation = faster adaptation, but with limits
        base_rate = self.params.perspective_adaptation_rate
        deviation_factor = min(2.0, deviation / self.params.angle_change_threshold)
        adaptive_rate = min(0.5, base_rate * deviation_factor)
        return adaptive_rate
    
    def _update_tracking_history(self, board_region: BoardRegion):
        """Update the tracking history with the new detection."""
        self._tracking_history.append(board_region)
        
        # Keep only recent history
        if len(self._tracking_history) > self._max_history_size:
            self._tracking_history = self._tracking_history[-self._max_history_size:]
    
    def get_square_coordinates(self, board_region: BoardRegion) -> SquareGrid:
        """
        Extract 8x8 grid coordinates from detected board region.
        
        Args:
            board_region: Detected board region
            
        Returns:
            SquareGrid: Grid of square coordinates
        """
        # Define the standard chess board corners in board coordinates
        board_corners = np.array([
            [0, 0],    # Top-left
            [8, 0],    # Top-right  
            [8, 8],    # Bottom-right
            [0, 8]     # Bottom-left
        ], dtype=np.float32)
        
        # Convert detected corners to numpy array
        detected_corners = np.array(board_region.corners, dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(board_corners, detected_corners)
        
        # Generate square coordinates
        squares = {}
        for row in range(8):
            for col in range(8):
                # Calculate square boundaries in board coordinates
                square_corners = np.array([
                    [col, row],
                    [col + 1, row],
                    [col + 1, row + 1],
                    [col, row + 1]
                ], dtype=np.float32)
                
                # Transform to image coordinates
                transformed_corners = cv2.perspectiveTransform(
                    square_corners.reshape(-1, 1, 2), 
                    transform_matrix
                ).reshape(-1, 2)
                
                # Calculate bounding box
                x_coords = transformed_corners[:, 0]
                y_coords = transformed_corners[:, 1]
                
                x1, y1 = float(np.min(x_coords)), float(np.min(y_coords))
                x2, y2 = float(np.max(x_coords)), float(np.max(y_coords))
                
                # Create position (chess notation: a1 = (0,0), h8 = (7,7))
                position = Position(col, 7 - row)  # Flip row for chess notation
                squares[position] = (x1, y1, x2, y2)
        
        return SquareGrid(squares=squares, board_region=board_region)
    
    def determine_orientation(self, board_region: BoardRegion) -> Orientation:
        """
        Determine the correct orientation of the board (which side is white/black).
        
        Args:
            board_region: Detected board region
            
        Returns:
            Orientation: Board orientation
        """
        # This is a simplified implementation
        # In practice, this would analyze piece positions or board features
        return board_region.orientation or Orientation.WHITE_BOTTOM
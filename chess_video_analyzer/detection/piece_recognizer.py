"""
Chess piece recognition module using computer vision and machine learning.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from ..core.data_models import (
    Position, PieceType, PieceKind, Color, BoardState, SquareGrid
)


class PieceRecognitionError(Exception):
    """Base exception for piece recognition errors."""
    pass


@dataclass
class RecognitionParams:
    """Parameters for piece recognition algorithms."""
    # Confidence thresholds
    min_confidence: float = 0.5
    piece_detection_threshold: float = 0.3
    
    # Image preprocessing parameters
    square_resize_size: Tuple[int, int] = (64, 64)
    gaussian_blur_kernel: int = 3
    contrast_enhancement: bool = True
    
    # Feature extraction parameters
    use_color_features: bool = True
    use_texture_features: bool = True
    use_shape_features: bool = True
    
    # Classification parameters
    empty_square_threshold: float = 0.2
    color_classification_threshold: float = 0.6


class PieceRecognizer:
    """
    Recognizes chess pieces and their positions on the board using computer vision.
    
    This implementation uses a combination of traditional computer vision techniques
    and simple heuristics for piece classification. In a production system, this
    would be replaced with a trained CNN model.
    """
    
    def __init__(self, params: Optional[RecognitionParams] = None):
        """
        Initialize the piece recognizer.
        
        Args:
            params: Recognition parameters, uses defaults if None
        """
        self.params = params or RecognitionParams()
        self._piece_templates = {}
        self._background_model = None
        self._calibration_data = {}
        
        # Initialize feature extractors
        self._initialize_feature_extractors()
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction components."""
        try:
            # Initialize ORB detector for keypoint features
            self._orb_detector = cv2.ORB_create(nfeatures=50)
            
            # Initialize SIFT detector as fallback (if available)
            try:
                self._sift_detector = cv2.SIFT_create(nfeatures=30)
            except AttributeError:
                self._sift_detector = None
                logging.info("SIFT detector not available, using ORB only")
            
            # Initialize color histogram parameters
            self._hist_bins = 32
            self._hist_ranges = [0, 256]
            
        except Exception as e:
            logging.warning(f"Feature extractor initialization failed: {e}")
            self._orb_detector = None
            self._sift_detector = None
    
    def recognize_pieces(self, frame: np.ndarray, square_grid: SquareGrid) -> BoardState:
        """
        Recognize all chess pieces in the given frame.
        
        Args:
            frame: Input video frame as numpy array
            square_grid: Grid of square coordinates
            
        Returns:
            BoardState: Detected pieces and their positions
            
        Raises:
            PieceRecognitionError: If recognition fails
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        if not square_grid.squares:
            raise ValueError("Invalid square grid provided")
        
        try:
            squares = {}
            confidence_scores = []
            timestamp = 0.0  # In a real implementation, this would be the actual timestamp
            
            # Process each square
            for position, square_coords in square_grid.squares.items():
                try:
                    # Extract square image
                    square_image = self._extract_square_image(frame, square_coords)
                    
                    if square_image is None or square_image.size == 0:
                        squares[position] = None
                        continue
                    
                    # Classify piece in this square
                    piece_type = self.classify_piece(square_image)
                    confidence = self.get_confidence_score(piece_type, square_image)
                    
                    # Apply confidence threshold
                    if piece_type is not None and confidence >= self.params.min_confidence:
                        squares[position] = piece_type
                        confidence_scores.append(confidence)
                    else:
                        squares[position] = None
                        if piece_type is None:
                            confidence_scores.append(1.0)  # High confidence for empty squares
                        else:
                            confidence_scores.append(confidence)
                
                except Exception as e:
                    logging.warning(f"Failed to process square at {position}: {e}")
                    squares[position] = None
                    confidence_scores.append(0.0)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return BoardState(
                squares=squares,
                timestamp=timestamp,
                confidence=float(overall_confidence)
            )
            
        except Exception as e:
            logging.error(f"Piece recognition failed: {e}")
            raise PieceRecognitionError(f"Failed to recognize pieces: {e}")
    
    def _extract_square_image(self, frame: np.ndarray, square_coords: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Extract and preprocess image of a single square."""
        try:
            x1, y1, x2, y2 = map(int, square_coords)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract square region
            square_image = frame[y1:y2, x1:x2]
            
            if square_image.size == 0:
                return None
            
            # Resize to standard size
            square_image = cv2.resize(square_image, self.params.square_resize_size)
            
            # Apply preprocessing
            square_image = self._preprocess_square_image(square_image)
            
            return square_image
            
        except Exception as e:
            logging.warning(f"Square image extraction failed: {e}")
            return None
    
    def _preprocess_square_image(self, square_image: np.ndarray) -> np.ndarray:
        """Preprocess square image for better recognition."""
        try:
            # Apply Gaussian blur to reduce noise
            if self.params.gaussian_blur_kernel > 0:
                square_image = cv2.GaussianBlur(
                    square_image, 
                    (self.params.gaussian_blur_kernel, self.params.gaussian_blur_kernel), 
                    0
                )
            
            # Enhance contrast if enabled
            if self.params.contrast_enhancement:
                # Convert to LAB color space for better contrast enhancement
                if len(square_image.shape) == 3:
                    lab = cv2.cvtColor(square_image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                    l = clahe.apply(l)
                    
                    # Merge channels and convert back
                    enhanced = cv2.merge([l, a, b])
                    square_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                else:
                    # For grayscale images
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                    square_image = clahe.apply(square_image)
            
            return square_image
            
        except Exception as e:
            logging.warning(f"Image preprocessing failed: {e}")
            return square_image
    
    def classify_piece(self, square_image: np.ndarray) -> Optional[PieceType]:
        """
        Classify the piece in a square image.
        
        Args:
            square_image: Preprocessed square image
            
        Returns:
            PieceType: Detected piece type, or None if empty square
        """
        if square_image is None or square_image.size == 0:
            return None
        
        try:
            # First, determine if square is empty or contains a piece
            if not self._has_piece(square_image):
                return None
            
            # Determine piece color
            piece_color = self._classify_piece_color(square_image)
            if piece_color is None:
                return None
            
            # Determine piece type
            piece_kind = self._classify_piece_type(square_image, piece_color)
            if piece_kind is None:
                return None
            
            return PieceType(color=piece_color, type=piece_kind)
            
        except Exception as e:
            logging.warning(f"Piece classification failed: {e}")
            return None
    
    def _has_piece(self, square_image: np.ndarray) -> bool:
        """Determine if a square contains a piece."""
        try:
            # Convert to grayscale for analysis
            if len(square_image.shape) == 3:
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_image
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Check for sufficient variation (pieces create intensity variation)
            if std_intensity < 15:  # Very uniform - likely empty
                return False
            
            # Check for circular/blob-like features (typical of pieces)
            # Use blob detection
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 2000
            params.filterByCircularity = True
            params.minCircularity = 0.3
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            # If we detect blob-like features, likely a piece
            if len(keypoints) > 0:
                return True
            
            # Fallback: check for edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # High edge density suggests a piece
            return edge_density > self.params.empty_square_threshold
            
        except Exception as e:
            logging.warning(f"Piece detection failed: {e}")
            return False
    
    def _classify_piece_color(self, square_image: np.ndarray) -> Optional[Color]:
        """Classify the color of a piece."""
        try:
            # Convert to different color spaces for analysis
            if len(square_image.shape) == 3:
                # Analyze in multiple color spaces
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(square_image, cv2.COLOR_BGR2HSV)
                
                # Get the central region (where piece is likely to be)
                h, w = gray.shape
                center_region = gray[h//4:3*h//4, w//4:3*w//4]
                
                if center_region.size == 0:
                    center_region = gray
                
                mean_intensity = np.mean(center_region)
                
                # Simple threshold-based classification
                # This is a simplified approach - real implementation would use trained models
                if mean_intensity > 140:  # Bright pieces
                    return Color.WHITE
                elif mean_intensity < 100:  # Dark pieces
                    return Color.BLACK
                else:
                    # Medium intensity - use additional features
                    # Check value channel in HSV
                    v_channel = hsv[:, :, 2]
                    center_v = v_channel[h//4:3*h//4, w//4:3*w//4]
                    if center_v.size > 0:
                        mean_v = np.mean(center_v)
                        return Color.WHITE if mean_v > 120 else Color.BLACK
                    else:
                        return Color.WHITE if mean_intensity > 120 else Color.BLACK
            else:
                # Grayscale image
                mean_intensity = np.mean(square_image)
                return Color.WHITE if mean_intensity > 120 else Color.BLACK
                
        except Exception as e:
            logging.warning(f"Color classification failed: {e}")
            return None
    
    def _classify_piece_type(self, square_image: np.ndarray, piece_color: Color) -> Optional[PieceKind]:
        """
        Classify the type of piece.
        
        This is a simplified implementation using basic shape and feature analysis.
        In a production system, this would use a trained CNN model.
        """
        try:
            # Convert to grayscale for shape analysis
            if len(square_image.shape) == 3:
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_image
            
            # Extract features for classification
            features = self._extract_piece_features(gray, square_image)
            
            # Simple rule-based classification based on features
            # This is a placeholder - real implementation would use ML models
            
            # Analyze shape characteristics
            contours, _ = cv2.findContours(
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return PieceKind.PAWN  # Default fallback
            
            # Find largest contour (main piece shape)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape properties
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            if perimeter == 0:
                return PieceKind.PAWN
            
            # Circularity measure
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = float(w) / h if h > 0 else 1.0
            
            # Simple heuristic classification
            # These thresholds are rough estimates and would need tuning
            if circularity > 0.7:  # Very circular
                if area > 800:  # Large and circular
                    return PieceKind.QUEEN
                else:
                    return PieceKind.PAWN
            elif circularity > 0.5:  # Moderately circular
                if aspect_ratio > 1.2:  # Wide
                    return PieceKind.ROOK
                else:
                    return PieceKind.BISHOP
            else:  # Less circular, more complex shape
                if area > 1000:  # Large complex shape
                    return PieceKind.KING
                else:
                    return PieceKind.KNIGHT
            
        except Exception as e:
            logging.warning(f"Piece type classification failed: {e}")
            # Return a random piece type based on image hash for consistency
            image_hash = abs(hash(square_image.tobytes())) % 6
            return list(PieceKind)[image_hash]
    
    def _extract_piece_features(self, gray_image: np.ndarray, color_image: np.ndarray) -> Dict:
        """Extract features from piece image for classification."""
        features = {}
        
        try:
            # Shape features
            if self.params.use_shape_features:
                features.update(self._extract_shape_features(gray_image))
            
            # Texture features
            if self.params.use_texture_features:
                features.update(self._extract_texture_features(gray_image))
            
            # Color features
            if self.params.use_color_features and len(color_image.shape) == 3:
                features.update(self._extract_color_features(color_image))
            
        except Exception as e:
            logging.warning(f"Feature extraction failed: {e}")
        
        return features
    
    def _extract_shape_features(self, gray_image: np.ndarray) -> Dict:
        """Extract shape-based features."""
        features = {}
        
        try:
            # Moments
            moments = cv2.moments(gray_image)
            if moments['m00'] != 0:
                features['centroid_x'] = moments['m10'] / moments['m00']
                features['centroid_y'] = moments['m01'] / moments['m00']
            else:
                features['centroid_x'] = gray_image.shape[1] / 2
                features['centroid_y'] = gray_image.shape[0] / 2
            
            # Hu moments (shape descriptors)
            hu_moments = cv2.HuMoments(moments)
            for i, hu in enumerate(hu_moments.flatten()):
                features[f'hu_moment_{i}'] = float(hu)
            
            # Contour features
            contours, _ = cv2.findContours(
                cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                features['contour_area'] = cv2.contourArea(main_contour)
                features['contour_perimeter'] = cv2.arcLength(main_contour, True)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(main_contour)
                features['bbox_aspect_ratio'] = float(w) / h if h > 0 else 1.0
                features['bbox_extent'] = features['contour_area'] / (w * h) if w * h > 0 else 0
            
        except Exception as e:
            logging.warning(f"Shape feature extraction failed: {e}")
        
        return features
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict:
        """Extract texture-based features."""
        features = {}
        
        try:
            # Local Binary Pattern (simplified)
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['mean_gradient'] = np.mean(gradient_magnitude)
            features['std_gradient'] = np.std(gradient_magnitude)
            
            # Intensity statistics
            features['mean_intensity'] = np.mean(gray_image)
            features['std_intensity'] = np.std(gray_image)
            features['min_intensity'] = np.min(gray_image)
            features['max_intensity'] = np.max(gray_image)
            
        except Exception as e:
            logging.warning(f"Texture feature extraction failed: {e}")
        
        return features
    
    def _extract_color_features(self, color_image: np.ndarray) -> Dict:
        """Extract color-based features."""
        features = {}
        
        try:
            # Color histograms
            for i, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([color_image], [i], None, [self._hist_bins], self._hist_ranges)
                hist = hist.flatten() / np.sum(hist)  # Normalize
                
                features[f'{color}_hist_mean'] = np.mean(hist)
                features[f'{color}_hist_std'] = np.std(hist)
                features[f'{color}_dominant_bin'] = np.argmax(hist)
            
            # Color moments
            for i, color in enumerate(['b', 'g', 'r']):
                channel = color_image[:, :, i]
                features[f'{color}_mean'] = np.mean(channel)
                features[f'{color}_std'] = np.std(channel)
                features[f'{color}_skewness'] = float(np.mean(((channel - np.mean(channel)) / np.std(channel))**3))
            
        except Exception as e:
            logging.warning(f"Color feature extraction failed: {e}")
        
        return features
    
    def get_confidence_score(self, piece_type: Optional[PieceType], square_image: Optional[np.ndarray] = None) -> float:
        """
        Calculate confidence score for piece classification.
        
        Args:
            piece_type: Classified piece type (None for empty square)
            square_image: Original square image (optional, for additional analysis)
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            if piece_type is None:
                # For empty squares, confidence is based on how "empty" it looks
                if square_image is not None:
                    return self._calculate_empty_square_confidence(square_image)
                else:
                    return 0.8  # Default confidence for empty squares
            
            # Base confidence scores for different piece types
            # Some pieces are easier to recognize than others
            base_confidence = {
                PieceKind.KING: 0.85,    # Distinctive shape
                PieceKind.QUEEN: 0.80,   # Large and distinctive
                PieceKind.ROOK: 0.90,    # Simple, rectangular shape
                PieceKind.BISHOP: 0.75,  # Pointed top, but can be confused
                PieceKind.KNIGHT: 0.70,  # Complex shape, harder to recognize
                PieceKind.PAWN: 0.65     # Small, can be confused with other pieces
            }
            
            confidence = base_confidence.get(piece_type.type, 0.5)
            
            # Adjust confidence based on image quality if available
            if square_image is not None:
                quality_factor = self._assess_image_quality(square_image)
                confidence *= quality_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logging.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    def _calculate_empty_square_confidence(self, square_image: np.ndarray) -> float:
        """Calculate confidence that a square is empty."""
        try:
            if len(square_image.shape) == 3:
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_image
            
            # Calculate uniformity measures
            std_intensity = np.std(gray)
            
            # Low standard deviation suggests uniform (empty) square
            uniformity_score = max(0.0, 1.0 - std_intensity / 50.0)
            
            # Check for absence of blob-like features
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 30
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            # Fewer keypoints suggests empty square
            keypoint_score = max(0.0, 1.0 - len(keypoints) / 5.0)
            
            # Combine scores
            empty_confidence = (uniformity_score * 0.7 + keypoint_score * 0.3)
            
            return min(1.0, max(0.3, empty_confidence))
            
        except Exception as e:
            logging.warning(f"Empty square confidence calculation failed: {e}")
            return 0.7
    
    def _assess_image_quality(self, square_image: np.ndarray) -> float:
        """Assess the quality of a square image for recognition."""
        try:
            if len(square_image.shape) == 3:
                gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_image
            
            # Check for blur (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian_var / 500.0)  # Normalize
            
            # Check for proper contrast
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50.0)
            
            # Check for proper brightness (not too dark or too bright)
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Combine quality factors
            quality_factor = (blur_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2)
            
            return min(1.0, max(0.3, quality_factor))
            
        except Exception as e:
            logging.warning(f"Image quality assessment failed: {e}")
            return 0.7
    
    def calibrate_with_known_position(self, frame: np.ndarray, square_grid: SquareGrid, 
                                    known_pieces: Dict[Position, PieceType]):
        """
        Calibrate the recognizer with a known board position.
        
        This can be used to improve recognition accuracy by learning from
        known positions (e.g., starting position of a game).
        
        Args:
            frame: Video frame with known position
            square_grid: Square grid coordinates
            known_pieces: Dictionary of known piece positions
        """
        try:
            self._calibration_data = {}
            
            for position, piece_type in known_pieces.items():
                if position in square_grid.squares:
                    square_coords = square_grid.squares[position]
                    square_image = self._extract_square_image(frame, square_coords)
                    
                    if square_image is not None:
                        # Store features for this piece type
                        features = self._extract_piece_features(
                            cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY) if len(square_image.shape) == 3 else square_image,
                            square_image
                        )
                        
                        piece_key = f"{piece_type.color.value}_{piece_type.type.value}"
                        if piece_key not in self._calibration_data:
                            self._calibration_data[piece_key] = []
                        
                        self._calibration_data[piece_key].append(features)
            
            logging.info(f"Calibrated with {len(known_pieces)} known pieces")
            
        except Exception as e:
            logging.warning(f"Calibration failed: {e}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set the minimum confidence threshold for piece recognition."""
        if 0.0 <= threshold <= 1.0:
            self.params.min_confidence = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_recognition_stats(self) -> Dict:
        """Get statistics about recognition performance."""
        return {
            'confidence_threshold': self.params.min_confidence,
            'calibration_data_available': len(self._calibration_data) > 0,
            'feature_extractors_available': {
                'orb': self._orb_detector is not None,
                'sift': self._sift_detector is not None
            }
        }
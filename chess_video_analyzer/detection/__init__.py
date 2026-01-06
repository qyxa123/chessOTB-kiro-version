"""
Detection module for board detection and piece recognition.
"""

from .board_detector import (
    BoardDetector,
    BoardDetectionError,
    BoardNotFoundError,
    DetectionParams
)
from .piece_recognizer import (
    PieceRecognizer,
    PieceRecognitionError,
    RecognitionParams
)

__all__ = [
    'BoardDetector',
    'BoardDetectionError', 
    'BoardNotFoundError',
    'DetectionParams',
    'PieceRecognizer',
    'PieceRecognitionError',
    'RecognitionParams'
]
"""
Chess Video Analyzer - A system for analyzing chess game videos and generating notation files.
"""

__version__ = "0.1.0"
__author__ = "Chess Video Analyzer Team"

from .core.data_models import (
    Position,
    PieceType,
    PieceKind,
    Color,
    Move,
    BoardState,
    GameState,
    VideoMetadata,
    GameMetadata,
    SpecialMoveType,
    GameResult,
    CastlingRights,
    Square
)

from .quality import QualityController, QualityReport, QualityFlag

__all__ = [
    "Position",
    "PieceType", 
    "PieceKind",
    "Color",
    "Move",
    "BoardState",
    "GameState",
    "VideoMetadata",
    "GameMetadata",
    "SpecialMoveType",
    "GameResult",
    "CastlingRights",
    "Square",
    "QualityController",
    "QualityReport", 
    "QualityFlag"
]
"""
Core module containing data models and base classes.
"""

from .data_models import (
    Color,
    PieceKind,
    SpecialMoveType,
    GameResult,
    Orientation,
    Position,
    PieceType,
    Square,
    Move,
    BoardState,
    CastlingRights,
    GameState,
    VideoMetadata,
    GameMetadata,
    BoardRegion,
    SquareGrid
)

__all__ = [
    'Color',
    'PieceKind', 
    'SpecialMoveType',
    'GameResult',
    'Orientation',
    'Position',
    'PieceType',
    'Square',
    'Move',
    'BoardState',
    'CastlingRights',
    'GameState',
    'VideoMetadata',
    'GameMetadata',
    'BoardRegion',
    'SquareGrid'
]
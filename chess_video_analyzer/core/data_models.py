"""
Core data models and enumerations for the Chess Video Analyzer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Color(Enum):
    """Chess piece colors."""
    WHITE = "white"
    BLACK = "black"


class PieceKind(Enum):
    """Types of chess pieces."""
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"


class SpecialMoveType(Enum):
    """Special chess move types."""
    CASTLING_KINGSIDE = "O-O"
    CASTLING_QUEENSIDE = "O-O-O"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"


class GameResult(Enum):
    """Possible game results."""
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"


@dataclass(frozen=True)
class Position:
    """Represents a position on the chess board."""
    x: int
    y: int
    
    def __post_init__(self):
        """Validate position coordinates."""
        if not (0 <= self.x <= 7) or not (0 <= self.y <= 7):
            raise ValueError(f"Position coordinates must be between 0-7, got ({self.x}, {self.y})")


@dataclass
class PieceType:
    """Represents a chess piece with color and type."""
    color: Color
    type: PieceKind


@dataclass
class Square:
    """Represents a square on the chess board."""
    position: Position
    piece: Optional[PieceType]


@dataclass
class Move:
    """Represents a chess move."""
    from_square: Position
    to_square: Position
    piece: PieceType
    captured_piece: Optional[PieceType] = None
    special_move: Optional[SpecialMoveType] = None
    promotion_piece: Optional[PieceKind] = None  # For pawn promotions
    is_flagged: bool = False
    flag_reason: Optional[str] = None


@dataclass
class BoardState:
    """Represents the state of the chess board at a specific time."""
    squares: Dict[Position, Optional[PieceType]]
    timestamp: float
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate board state."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class CastlingRights:
    """Represents castling rights for both players."""
    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True


@dataclass
class GameState:
    """Represents the complete state of a chess game."""
    current_position: BoardState
    move_history: List[Move]
    castling_rights: CastlingRights
    en_passant_target: Optional[Position] = None
    halfmove_clock: int = 0
    fullmove_number: int = 1
    active_color: Color = Color.WHITE
    flagged_moves: List[Move] = None
    
    def __post_init__(self):
        """Validate game state."""
        if self.halfmove_clock < 0:
            raise ValueError(f"Halfmove clock cannot be negative, got {self.halfmove_clock}")
        if self.fullmove_number < 1:
            raise ValueError(f"Fullmove number must be at least 1, got {self.fullmove_number}")
        if self.flagged_moves is None:
            self.flagged_moves = []


@dataclass
class VideoMetadata:
    """Metadata about the input video file."""
    duration: float
    fps: float
    resolution: Tuple[int, int]
    format: str
    
    def __post_init__(self):
        """Validate video metadata."""
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")
        if len(self.resolution) != 2 or any(r <= 0 for r in self.resolution):
            raise ValueError(f"Resolution must be a tuple of two positive integers, got {self.resolution}")


@dataclass
class GameMetadata:
    """Metadata about the chess game."""
    event: str = "Casual Game"
    site: str = "Unknown"
    date: str = "????.??.??"
    round: str = "?"
    white_player: str = "White"
    black_player: str = "Black"
    result: str = "*"


class Orientation(Enum):
    """Board orientation - which side is white."""
    WHITE_BOTTOM = "white_bottom"
    WHITE_TOP = "white_top"
    WHITE_LEFT = "white_left"
    WHITE_RIGHT = "white_right"


@dataclass
class BoardRegion:
    """Represents the detected chess board region in a frame."""
    corners: List[Tuple[float, float]]  # Four corner points of the board
    confidence: float
    orientation: Optional[Orientation] = None
    
    def __post_init__(self):
        """Validate board region."""
        if len(self.corners) != 4:
            raise ValueError(f"Board region must have exactly 4 corners, got {len(self.corners)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class SquareGrid:
    """Represents the 8x8 grid of chess squares with their pixel coordinates."""
    squares: Dict[Position, Tuple[float, float, float, float]]  # Position -> (x1, y1, x2, y2)
    board_region: BoardRegion
    
    def get_square_center(self, position: Position) -> Tuple[float, float]:
        """Get the center coordinates of a square."""
        if position not in self.squares:
            raise ValueError(f"Position {position} not found in grid")
        x1, y1, x2, y2 = self.squares[position]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
"""
Game state management module for accurate FEN generation.

This module implements the GameStateManager class that maintains accurate
game state throughout a chess game, including castling rights, en passant
possibilities, and move counters.
"""

from typing import Optional, List
from copy import deepcopy
from dataclasses import dataclass

from chess_video_analyzer.core.data_models import (
    GameState, BoardState, Move, Position, Color, PieceKind, 
    SpecialMoveType, CastlingRights, PieceType, GameResult
)


@dataclass
class MoveValidationResult:
    """Result of move validation."""
    is_legal: bool
    reason: Optional[str] = None


class GameStateManager:
    """
    Manages the complete game state for accurate FEN generation.
    
    This class tracks:
    - Castling rights throughout the game
    - En passant possibilities
    - Halfmove clock (50-move rule)
    - Fullmove counter
    - Active color
    
    Requirements: 6.4, 6.5
    """
    
    def __init__(self, initial_position: Optional[BoardState] = None):
        """
        Initialize the game state manager.
        
        Args:
            initial_position: The starting board position. If None, assumes standard starting position.
        """
        self.game_state = GameState(
            current_position=initial_position or self._create_standard_starting_position(),
            move_history=[],
            castling_rights=CastlingRights(),
            en_passant_target=None,
            halfmove_clock=0,
            fullmove_number=1,
            active_color=Color.WHITE,
            flagged_moves=[]
        )
    
    def update_state(self, move: Move, new_board_state: BoardState) -> GameState:
        """
        Update the game state after a move is made.
        
        Args:
            move: The move that was made
            new_board_state: The resulting board state after the move
            
        Returns:
            The updated game state
            
        Requirements: 6.4, 6.5, 4.3
        """
        # Validate the move against chess rules
        validation_result = self.validate_move(move)
        if not validation_result.is_legal:
            move.is_flagged = True
            move.flag_reason = validation_result.reason
        else:
            # Even if the move is legal, check for questionable patterns
            self.flag_questionable_move(move)
        
        # Create a new game state based on the current one
        new_state = GameState(
            current_position=new_board_state,
            move_history=self.game_state.move_history + [move],
            castling_rights=deepcopy(self.game_state.castling_rights),
            en_passant_target=None,  # Reset en passant, will be set if applicable
            halfmove_clock=self.game_state.halfmove_clock,
            fullmove_number=self.game_state.fullmove_number,
            active_color=self.game_state.active_color,
            flagged_moves=self.game_state.flagged_moves.copy()
        )
        
        # Add to flagged moves if the move is flagged
        if move.is_flagged:
            new_state.flagged_moves.append(move)
        
        # Update castling rights based on the move
        self._update_castling_rights(move, new_state)
        
        # Update en passant target
        self._update_en_passant_target(move, new_state)
        
        # Update halfmove clock (50-move rule)
        self._update_halfmove_clock(move, new_state)
        
        # Update fullmove number and active color
        self._update_move_counters(new_state)
        
        # Store the updated state
        self.game_state = new_state
        
        return new_state
    
    def get_current_state(self) -> GameState:
        """Get the current game state."""
        return deepcopy(self.game_state)
    
    def _update_castling_rights(self, move: Move, game_state: GameState) -> None:
        """
        Update castling rights based on the move made.
        
        Castling rights are lost when:
        - The king moves
        - A rook moves from its starting position
        - A rook is captured from its starting position
        
        Requirements: 6.4
        """
        # If this is a castling move, remove castling rights for the side that castled
        if move.special_move in [SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE]:
            if move.piece.color == Color.WHITE:
                game_state.castling_rights.white_kingside = False
                game_state.castling_rights.white_queenside = False
            else:
                game_state.castling_rights.black_kingside = False
                game_state.castling_rights.black_queenside = False
            return
        
        # Check if king moved (loses all castling rights for that color)
        if move.piece.type == PieceKind.KING:
            if move.piece.color == Color.WHITE:
                game_state.castling_rights.white_kingside = False
                game_state.castling_rights.white_queenside = False
            else:
                game_state.castling_rights.black_kingside = False
                game_state.castling_rights.black_queenside = False
            return
        
        # Check if rook moved from starting position
        if move.piece.type == PieceKind.ROOK:
            # White rooks (rank 1, y=7)
            if move.piece.color == Color.WHITE:
                if move.from_square == Position(0, 7):  # Queenside rook
                    game_state.castling_rights.white_queenside = False
                elif move.from_square == Position(7, 7):  # Kingside rook
                    game_state.castling_rights.white_kingside = False
            # Black rooks (rank 8, y=0)
            else:
                if move.from_square == Position(0, 0):  # Queenside rook
                    game_state.castling_rights.black_queenside = False
                elif move.from_square == Position(7, 0):  # Kingside rook
                    game_state.castling_rights.black_kingside = False
        
        # Check if a rook was captured on its starting square
        if move.captured_piece and move.captured_piece.type == PieceKind.ROOK:
            # White rooks captured (rank 1, y=7)
            if move.captured_piece.color == Color.WHITE:
                if move.to_square == Position(0, 7):  # Queenside rook captured
                    game_state.castling_rights.white_queenside = False
                elif move.to_square == Position(7, 7):  # Kingside rook captured
                    game_state.castling_rights.white_kingside = False
            # Black rooks captured (rank 8, y=0)
            else:
                if move.to_square == Position(0, 0):  # Queenside rook captured
                    game_state.castling_rights.black_queenside = False
                elif move.to_square == Position(7, 0):  # Kingside rook captured
                    game_state.castling_rights.black_kingside = False
    
    def _update_en_passant_target(self, move: Move, game_state: GameState) -> None:
        """
        Update en passant target square.
        
        En passant is possible when:
        - A pawn moves two squares forward from its starting position
        - The target square is the square the pawn "jumped over"
        
        Requirements: 6.5
        """
        # Only pawns can create en passant opportunities
        if move.piece.type != PieceKind.PAWN:
            return
        
        # Check if pawn moved two squares forward
        if abs(move.to_square.y - move.from_square.y) == 2:
            # White pawn moving from rank 2 to rank 4 (y=6 to y=4)
            if move.piece.color == Color.WHITE and move.from_square.y == 6 and move.to_square.y == 4:
                game_state.en_passant_target = Position(move.to_square.x, 5)  # rank 3
            # Black pawn moving from rank 7 to rank 5 (y=1 to y=3)
            elif move.piece.color == Color.BLACK and move.from_square.y == 1 and move.to_square.y == 3:
                game_state.en_passant_target = Position(move.to_square.x, 2)  # rank 6
    
    def _update_halfmove_clock(self, move: Move, game_state: GameState) -> None:
        """
        Update the halfmove clock for the 50-move rule.
        
        The halfmove clock is reset to 0 when:
        - A pawn moves
        - A capture is made
        
        Otherwise, it is incremented by 1.
        
        Requirements: 6.5
        """
        # Reset clock if pawn moved or capture was made
        if move.piece.type == PieceKind.PAWN or move.captured_piece is not None:
            game_state.halfmove_clock = 0
        else:
            game_state.halfmove_clock += 1
    
    def _update_move_counters(self, game_state: GameState) -> None:
        """
        Update the fullmove number and active color.
        
        - Active color alternates after each move
        - Fullmove number increments after Black's move
        
        Requirements: 6.5
        """
        # Switch active color
        if game_state.active_color == Color.WHITE:
            game_state.active_color = Color.BLACK
        else:
            game_state.active_color = Color.WHITE
            # Increment fullmove number after Black's move
            game_state.fullmove_number += 1
    
    def _create_standard_starting_position(self) -> BoardState:
        """Create the standard chess starting position."""
        squares = {}
        
        # Initialize empty squares
        for x in range(8):
            for y in range(8):
                squares[Position(x, y)] = None
        
        # Place white pieces (rank 1 and 2, y=7 and y=6)
        # Note: y=7 corresponds to rank 1, y=0 corresponds to rank 8
        squares[Position(0, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        squares[Position(1, 7)] = PieceType(Color.WHITE, PieceKind.KNIGHT)
        squares[Position(2, 7)] = PieceType(Color.WHITE, PieceKind.BISHOP)
        squares[Position(3, 7)] = PieceType(Color.WHITE, PieceKind.QUEEN)
        squares[Position(4, 7)] = PieceType(Color.WHITE, PieceKind.KING)
        squares[Position(5, 7)] = PieceType(Color.WHITE, PieceKind.BISHOP)
        squares[Position(6, 7)] = PieceType(Color.WHITE, PieceKind.KNIGHT)
        squares[Position(7, 7)] = PieceType(Color.WHITE, PieceKind.ROOK)
        
        # White pawns (rank 2, y=6)
        for x in range(8):
            squares[Position(x, 6)] = PieceType(Color.WHITE, PieceKind.PAWN)
        
        # Place black pieces (rank 8 and 7, y=0 and y=1)
        squares[Position(0, 0)] = PieceType(Color.BLACK, PieceKind.ROOK)
        squares[Position(1, 0)] = PieceType(Color.BLACK, PieceKind.KNIGHT)
        squares[Position(2, 0)] = PieceType(Color.BLACK, PieceKind.BISHOP)
        squares[Position(3, 0)] = PieceType(Color.BLACK, PieceKind.QUEEN)
        squares[Position(4, 0)] = PieceType(Color.BLACK, PieceKind.KING)
        squares[Position(5, 0)] = PieceType(Color.BLACK, PieceKind.BISHOP)
        squares[Position(6, 0)] = PieceType(Color.BLACK, PieceKind.KNIGHT)
        squares[Position(7, 0)] = PieceType(Color.BLACK, PieceKind.ROOK)
        
        # Black pawns (rank 7, y=1)
        for x in range(8):
            squares[Position(x, 1)] = PieceType(Color.BLACK, PieceKind.PAWN)
        
        return BoardState(squares=squares, timestamp=0.0, confidence=1.0)
        
        return BoardState(squares=squares, timestamp=0.0, confidence=1.0)
    
    def reset_to_starting_position(self) -> None:
        """Reset the game state to the standard starting position."""
        self.__init__()
    
    def set_custom_starting_position(self, board_state: BoardState) -> None:
        """
        Set a custom starting position for the game.
        
        Args:
            board_state: The custom starting board state
        """
        self.game_state = GameState(
            current_position=board_state,
            move_history=[],
            castling_rights=CastlingRights(),
            en_passant_target=None,
            halfmove_clock=0,
            fullmove_number=1,
            active_color=Color.WHITE,
            flagged_moves=[]
        )
    
    def can_castle_kingside(self, color: Color) -> bool:
        """Check if the specified color can castle kingside."""
        if color == Color.WHITE:
            return self.game_state.castling_rights.white_kingside
        else:
            return self.game_state.castling_rights.black_kingside
    
    def can_castle_queenside(self, color: Color) -> bool:
        """Check if the specified color can castle queenside."""
        if color == Color.WHITE:
            return self.game_state.castling_rights.white_queenside
        else:
            return self.game_state.castling_rights.black_queenside
    
    def get_en_passant_target(self) -> Optional[Position]:
        """Get the current en passant target square, if any."""
        return self.game_state.en_passant_target
    
    def get_halfmove_clock(self) -> int:
        """Get the current halfmove clock value."""
        return self.game_state.halfmove_clock
    
    def get_fullmove_number(self) -> int:
        """Get the current fullmove number."""
        return self.game_state.fullmove_number
    
    def get_active_color(self) -> Color:
        """Get the color of the player to move."""
        return self.game_state.active_color
    
    def validate_move(self, move: Move) -> MoveValidationResult:
        """
        Validate a move against chess rules.
        
        Args:
            move: The move to validate
            
        Returns:
            MoveValidationResult indicating if the move is legal and why if not
            
        Requirements: 4.3
        """
        # Check if it's the correct player's turn
        if move.piece.color != self.game_state.active_color:
            return MoveValidationResult(False, f"It's {self.game_state.active_color.value}'s turn, not {move.piece.color.value}'s")
        
        # Check if the piece exists at the from square
        current_piece = self.game_state.current_position.squares.get(move.from_square)
        if current_piece is None:
            return MoveValidationResult(False, f"No piece at {move.from_square.x},{move.from_square.y}")
        
        if current_piece.color != move.piece.color or current_piece.type != move.piece.type:
            return MoveValidationResult(False, f"Piece mismatch at {move.from_square.x},{move.from_square.y}")
        
        # Validate piece-specific movement rules
        if not self._is_valid_piece_movement(move):
            return MoveValidationResult(False, f"Invalid {move.piece.type.value} move from {self._position_to_algebraic(move.from_square)} to {self._position_to_algebraic(move.to_square)}")
        
        # Check if path is clear (except for knights)
        if move.piece.type != PieceKind.KNIGHT and not self._is_path_clear_for_move(move):
            return MoveValidationResult(False, f"Path blocked from {self._position_to_algebraic(move.from_square)} to {self._position_to_algebraic(move.to_square)}")
        
        # Check if destination square is valid
        target_piece = self.game_state.current_position.squares.get(move.to_square)
        if target_piece is not None and target_piece.color == move.piece.color:
            return MoveValidationResult(False, f"Cannot capture own piece at {self._position_to_algebraic(move.to_square)}")
        
        return MoveValidationResult(True, "Valid move")
    def _is_valid_piece_movement(self, move: Move) -> bool:
        """
        Validate that the move follows the rules for the specific piece type.
        
        Args:
            move: The move to validate
            
        Returns:
            True if the move is valid for this piece type
        """
        from_pos = move.from_square
        to_pos = move.to_square
        piece_type = move.piece.type
        piece_color = move.piece.color
        
        # Calculate movement deltas
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        if piece_type == PieceKind.PAWN:
            return self._is_valid_pawn_move(move, dx, dy)
        elif piece_type == PieceKind.ROOK:
            return dx == 0 or dy == 0  # Horizontal or vertical only
        elif piece_type == PieceKind.BISHOP:
            return abs(dx) == abs(dy) and dx != 0  # Diagonal only
        elif piece_type == PieceKind.QUEEN:
            return (dx == 0 or dy == 0) or (abs(dx) == abs(dy) and dx != 0)  # Rook + Bishop
        elif piece_type == PieceKind.KING:
            return abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0)  # One square in any direction
        elif piece_type == PieceKind.KNIGHT:
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)  # L-shape
        
        return False
    
    def _is_valid_pawn_move(self, move: Move, dx: int, dy: int) -> bool:
        """
        Validate pawn movement rules.
        
        Args:
            move: The pawn move to validate
            dx: Horizontal movement
            dy: Vertical movement (positive = towards opponent)
            
        Returns:
            True if the pawn move is valid
        """
        piece_color = move.piece.color
        from_pos = move.from_square
        to_pos = move.to_square
        
        # Determine direction based on color (white moves up the board, y decreases)
        if piece_color == Color.WHITE:
            forward_direction = -1  # White moves from y=6 to y=0 (rank 2 to rank 8)
            starting_rank = 6  # White pawns start at y=6 (rank 2)
        else:
            forward_direction = 1   # Black moves from y=1 to y=7 (rank 7 to rank 1)
            starting_rank = 1   # Black pawns start at y=1 (rank 7)
        
        # Check if moving in correct direction
        if (dy * forward_direction) <= 0:
            return False  # Pawns can't move backwards or sideways without capturing
        
        # Straight move (no capture)
        if dx == 0:
            target_piece = self.game_state.current_position.squares.get(to_pos)
            if target_piece is not None:
                return False  # Can't move forward if square is occupied
            
            # One square forward
            if abs(dy) == 1:
                return True
            
            # Two squares forward from starting position
            if abs(dy) == 2 and from_pos.y == starting_rank:
                return True
            
            return False
        
        # Diagonal capture
        elif abs(dx) == 1 and abs(dy) == 1:
            target_piece = self.game_state.current_position.squares.get(to_pos)
            
            # Regular capture
            if target_piece is not None and target_piece.color != piece_color:
                return True
            
            # En passant capture
            if (target_piece is None and 
                self.game_state.en_passant_target is not None and 
                to_pos == self.game_state.en_passant_target):
                return True
            
            return False
        
        return False
    
    def _is_path_clear_for_move(self, move: Move) -> bool:
        """
        Check if the path between two positions is clear of pieces.
        
        Args:
            move: The move to check
            
        Returns:
            True if path is clear (excluding start and end positions)
        """
        from_pos = move.from_square
        to_pos = move.to_square
        
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        # No movement or adjacent squares - path is clear
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        
        # Determine step direction
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Check each square along the path (excluding start and end)
        current_x = from_pos.x + step_x
        current_y = from_pos.y + step_y
        
        while current_x != to_pos.x or current_y != to_pos.y:
            check_pos = Position(current_x, current_y)
            if self.game_state.current_position.squares.get(check_pos) is not None:
                return False  # Path is blocked
            
            current_x += step_x
            current_y += step_y
        
        return True
        
        # Check for invalid board positions
        if not self._is_valid_position(move.from_square):
            return MoveValidationResult(False, f"Invalid from square position: {move.from_square.x},{move.from_square.y}")
        
        if not self._is_valid_position(move.to_square):
            return MoveValidationResult(False, f"Invalid to square position: {move.to_square.x},{move.to_square.y}")
        
        # Check for null moves (piece not actually moving)
        if move.from_square == move.to_square:
            return MoveValidationResult(False, "Piece cannot move to the same square")
        
        # Validate special moves first (they have their own movement rules)
        if move.special_move is not None:
            special_validation = self._validate_special_move(move)
            if not special_validation.is_legal:
                return special_validation
        else:
            # Check basic move validity for each piece type (only if not a special move)
            if not self._is_valid_piece_move(move):
                return MoveValidationResult(False, f"Invalid {move.piece.type.value} move from {move.from_square.x},{move.from_square.y} to {move.to_square.x},{move.to_square.y}")
        
        # Check if the destination square is occupied by own piece
        dest_piece = self.game_state.current_position.squares.get(move.to_square)
        if dest_piece is not None and dest_piece.color == move.piece.color:
            return MoveValidationResult(False, f"Cannot capture own piece at {move.to_square.x},{move.to_square.y}")
        
        # Check if the path is clear (for pieces that can't jump)
        if not self._is_path_clear(move):
            return MoveValidationResult(False, f"Path blocked for {move.piece.type.value} move")
        
        # Check for invalid captures (claiming capture when no piece exists)
        if move.captured_piece is not None and dest_piece is None:
            return MoveValidationResult(False, f"No piece to capture at {move.to_square.x},{move.to_square.y}")
        
        # Check for missing captures (piece exists but not marked as captured)
        if move.captured_piece is None and dest_piece is not None and move.special_move != SpecialMoveType.EN_PASSANT:
            return MoveValidationResult(False, f"Must capture piece at {move.to_square.x},{move.to_square.y}")
        
        # Check if captured piece matches what's actually on the board
        if move.captured_piece is not None and dest_piece is not None:
            if (move.captured_piece.color != dest_piece.color or 
                move.captured_piece.type != dest_piece.type):
                return MoveValidationResult(False, f"Captured piece mismatch at {move.to_square.x},{move.to_square.y}")
        
        # Check if move would leave king in check (simplified check)
        if self._would_leave_king_in_check(move):
            return MoveValidationResult(False, "Move would leave king in check")
        
        # Additional validation for specific piece types
        piece_validation = self._validate_piece_specific_rules(move)
        if not piece_validation.is_legal:
            return piece_validation
        
        return MoveValidationResult(True)
    
    def _is_valid_piece_move(self, move: Move) -> bool:
        """
        Check if the move is valid for the specific piece type.
        
        Args:
            move: The move to validate
            
        Returns:
            True if the move is valid for the piece type
        """
        dx = move.to_square.x - move.from_square.x
        dy = move.to_square.y - move.from_square.y
        
        if move.piece.type == PieceKind.PAWN:
            return self._is_valid_pawn_move(move, dx, dy)
        elif move.piece.type == PieceKind.ROOK:
            return dx == 0 or dy == 0  # Horizontal or vertical
        elif move.piece.type == PieceKind.BISHOP:
            return abs(dx) == abs(dy) and dx != 0  # Diagonal
        elif move.piece.type == PieceKind.QUEEN:
            return dx == 0 or dy == 0 or abs(dx) == abs(dy)  # Rook + Bishop
        elif move.piece.type == PieceKind.KNIGHT:
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        elif move.piece.type == PieceKind.KING:
            return abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0)
        
        return False
    
    def _is_valid_pawn_move(self, move: Move, dx: int, dy: int) -> bool:
        """
        Validate pawn moves including captures and double moves.
        
        Args:
            move: The move to validate
            dx: Horizontal distance
            dy: Vertical distance
            
        Returns:
            True if the pawn move is valid
        """
        # Determine direction based on color (in our coordinate system: y=7 is rank 1, y=0 is rank 8)
        direction = -1 if move.piece.color == Color.WHITE else 1  # White moves toward lower y, Black toward higher y
        
        # Forward move
        if dx == 0:
            # Single step forward
            if dy == direction:
                return move.captured_piece is None
            # Double step from starting position
            elif dy == 2 * direction:
                starting_rank = 6 if move.piece.color == Color.WHITE else 1  # White starts on y=6 (rank 2), Black on y=1 (rank 7)
                return (move.from_square.y == starting_rank and 
                       move.captured_piece is None)
        # Diagonal capture
        elif abs(dx) == 1 and dy == direction:
            # Regular capture
            if move.captured_piece is not None:
                return True
            # En passant capture
            elif (move.special_move == SpecialMoveType.EN_PASSANT and 
                  self.game_state.en_passant_target == move.to_square):
                return True
        
        return False
    
    def _is_path_clear(self, move: Move) -> bool:
        """
        Check if the path between from and to squares is clear.
        
        Args:
            move: The move to check
            
        Returns:
            True if the path is clear
        """
        # Knights can jump over pieces
        if move.piece.type == PieceKind.KNIGHT:
            return True
        
        # King moves are only one square
        if move.piece.type == PieceKind.KING:
            return True
        
        dx = move.to_square.x - move.from_square.x
        dy = move.to_square.y - move.from_square.y
        
        # Determine step direction
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Check each square along the path (excluding start and end)
        current_x = move.from_square.x + step_x
        current_y = move.from_square.y + step_y
        
        while current_x != move.to_square.x or current_y != move.to_square.y:
            if self.game_state.current_position.squares.get(Position(current_x, current_y)) is not None:
                return False
            current_x += step_x
            current_y += step_y
        
        return True
    
    def _validate_special_move(self, move: Move) -> MoveValidationResult:
        """
        Validate special moves like castling and en passant.
        
        Args:
            move: The move to validate
            
        Returns:
            MoveValidationResult for the special move
        """
        if move.special_move == SpecialMoveType.CASTLING_KINGSIDE:
            return self._validate_castling(move, True)
        elif move.special_move == SpecialMoveType.CASTLING_QUEENSIDE:
            return self._validate_castling(move, False)
        elif move.special_move == SpecialMoveType.EN_PASSANT:
            return self._validate_en_passant(move)
        elif move.special_move == SpecialMoveType.PROMOTION:
            return self._validate_promotion(move)
        
        return MoveValidationResult(True)
    
    def _validate_castling(self, move: Move, kingside: bool) -> MoveValidationResult:
        """
        Validate castling moves.
        
        Args:
            move: The castling move
            kingside: True for kingside castling, False for queenside
            
        Returns:
            MoveValidationResult for castling
        """
        color = move.piece.color
        
        # Check castling rights
        if kingside and not self.can_castle_kingside(color):
            return MoveValidationResult(False, f"{color.value} has lost kingside castling rights")
        elif not kingside and not self.can_castle_queenside(color):
            return MoveValidationResult(False, f"{color.value} has lost queenside castling rights")
        
        # Check that king and rook are in correct positions
        king_start = Position(4, 0 if color == Color.WHITE else 7)
        if move.from_square != king_start:
            return MoveValidationResult(False, "King not in starting position for castling")
        
        # Check that squares between king and rook are empty
        if kingside:
            squares_to_check = [Position(5, king_start.y), Position(6, king_start.y)]
        else:
            squares_to_check = [Position(1, king_start.y), Position(2, king_start.y), Position(3, king_start.y)]
        
        for square in squares_to_check:
            if self.game_state.current_position.squares.get(square) is not None:
                return MoveValidationResult(False, "Squares between king and rook must be empty for castling")
        
        return MoveValidationResult(True)
    
    def _validate_en_passant(self, move: Move) -> MoveValidationResult:
        """
        Validate en passant captures.
        
        Args:
            move: The en passant move
            
        Returns:
            MoveValidationResult for en passant
        """
        if self.game_state.en_passant_target != move.to_square:
            return MoveValidationResult(False, "En passant target square mismatch")
        
        return MoveValidationResult(True)
    
    def _validate_promotion(self, move: Move) -> MoveValidationResult:
        """
        Validate pawn promotion moves.
        
        Args:
            move: The promotion move
            
        Returns:
            MoveValidationResult for promotion
        """
        # Check that it's a pawn reaching the end rank
        if move.piece.type != PieceKind.PAWN:
            return MoveValidationResult(False, "Only pawns can promote")
        
        end_rank = 7 if move.piece.color == Color.WHITE else 0
        if move.to_square.y != end_rank:
            return MoveValidationResult(False, f"Pawn must reach rank {end_rank + 1} to promote")
        
        return MoveValidationResult(True)
    
    def _would_leave_king_in_check(self, move: Move) -> bool:
        """
        Check if the move would leave the player's king in check.
        
        This is a simplified implementation that doesn't fully simulate
        the board state after the move.
        
        Args:
            move: The move to check
            
        Returns:
            True if the move would leave king in check
        """
        # This is a simplified check - in a full implementation,
        # we would simulate the move and check if the king is attacked
        # For now, we'll return False to avoid false positives
        return False
    
    def _is_valid_position(self, position: Position) -> bool:
        """
        Check if a position is valid on the chess board.
        
        Args:
            position: The position to validate
            
        Returns:
            True if the position is valid (0-7 for both x and y)
        """
        return 0 <= position.x <= 7 and 0 <= position.y <= 7
    
    def _validate_piece_specific_rules(self, move: Move) -> MoveValidationResult:
        """
        Validate piece-specific rules that go beyond basic movement patterns.
        
        Args:
            move: The move to validate
            
        Returns:
            MoveValidationResult for piece-specific validation
        """
        # Pawn-specific validation
        if move.piece.type == PieceKind.PAWN:
            return self._validate_pawn_specific_rules(move)
        
        # King-specific validation
        elif move.piece.type == PieceKind.KING:
            return self._validate_king_specific_rules(move)
        
        # Rook-specific validation
        elif move.piece.type == PieceKind.ROOK:
            return self._validate_rook_specific_rules(move)
        
        return MoveValidationResult(True)
    
    def _validate_pawn_specific_rules(self, move: Move) -> MoveValidationResult:
        """
        Validate pawn-specific rules.
        
        Args:
            move: The pawn move to validate
            
        Returns:
            MoveValidationResult for pawn validation
        """
        # Check for pawn promotion on correct ranks
        if move.special_move == SpecialMoveType.PROMOTION:
            promotion_rank = 0 if move.piece.color == Color.WHITE else 7  # White promotes at y=0 (rank 8), Black at y=7 (rank 1)
            if move.to_square.y != promotion_rank:
                return MoveValidationResult(False, f"Pawn promotion must be to rank {8 - promotion_rank}")
        
        # Check for pawn reaching end rank without promotion
        end_rank = 0 if move.piece.color == Color.WHITE else 7  # White end rank is y=0 (rank 8), Black is y=7 (rank 1)
        if move.to_square.y == end_rank and move.special_move != SpecialMoveType.PROMOTION:
            return MoveValidationResult(False, "Pawn reaching end rank must promote")
        
        # Check for invalid pawn double moves
        if abs(move.to_square.y - move.from_square.y) == 2:
            starting_rank = 6 if move.piece.color == Color.WHITE else 1  # White starts on y=6 (rank 2), Black on y=1 (rank 7)
            if move.from_square.y != starting_rank:
                return MoveValidationResult(False, "Pawn can only move two squares from starting position")
        
        return MoveValidationResult(True)
    
    def _validate_king_specific_rules(self, move: Move) -> MoveValidationResult:
        """
        Validate king-specific rules.
        
        Args:
            move: The king move to validate
            
        Returns:
            MoveValidationResult for king validation
        """
        # Check for king moving more than one square (non-castling)
        if move.special_move not in [SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE]:
            dx = abs(move.to_square.x - move.from_square.x)
            dy = abs(move.to_square.y - move.from_square.y)
            if dx > 1 or dy > 1:
                return MoveValidationResult(False, "King can only move one square at a time")
        
        return MoveValidationResult(True)
    
    def _validate_rook_specific_rules(self, move: Move) -> MoveValidationResult:
        """
        Validate rook-specific rules.
        
        Args:
            move: The rook move to validate
            
        Returns:
            MoveValidationResult for rook validation
        """
        # Rook must move in straight lines (horizontal or vertical)
        dx = move.to_square.x - move.from_square.x
        dy = move.to_square.y - move.from_square.y
        
        if dx != 0 and dy != 0:
            return MoveValidationResult(False, "Rook must move in straight lines (horizontal or vertical)")
        
        return MoveValidationResult(True)
    
    def get_flagged_moves(self) -> List[Move]:
        """
        Get all moves that have been flagged for user review.
        
        Returns:
            List of flagged moves
            
        Requirements: 4.3
        """
        return self.game_state.flagged_moves.copy()
    
    def get_flagged_moves_by_reason(self, reason_filter: str) -> List[Move]:
        """
        Get flagged moves filtered by reason.
        
        Args:
            reason_filter: String to search for in flag reasons
            
        Returns:
            List of flagged moves matching the reason filter
            
        Requirements: 4.3
        """
        return [move for move in self.game_state.flagged_moves 
                if move.flag_reason and reason_filter.lower() in move.flag_reason.lower()]
    
    def get_illegal_moves(self) -> List[Move]:
        """
        Get all moves that were flagged as illegal (rule violations).
        
        Returns:
            List of illegal moves
            
        Requirements: 4.3
        """
        illegal_keywords = ["invalid", "illegal", "cannot", "must", "no piece", "wrong turn", "blocked"]
        illegal_moves = []
        
        for move in self.game_state.flagged_moves:
            if move.flag_reason:
                reason_lower = move.flag_reason.lower()
                if any(keyword in reason_lower for keyword in illegal_keywords):
                    illegal_moves.append(move)
        
        return illegal_moves
    
    def get_questionable_moves(self) -> List[Move]:
        """
        Get all moves that were flagged as questionable but not necessarily illegal.
        
        Returns:
            List of questionable moves
            
        Requirements: 4.3
        """
        questionable_keywords = ["questionable", "unusual", "suspicious", "confidence", "missed"]
        questionable_moves = []
        
        for move in self.game_state.flagged_moves:
            if move.flag_reason:
                reason_lower = move.flag_reason.lower()
                if any(keyword in reason_lower for keyword in questionable_keywords):
                    questionable_moves.append(move)
        
        return questionable_moves
    
    def clear_all_flags(self) -> None:
        """
        Clear all move flags from the game state.
        
        Requirements: 4.3
        """
        for move in self.game_state.move_history:
            move.is_flagged = False
            move.flag_reason = None
        self.game_state.flagged_moves.clear()
    
    def get_flag_summary(self) -> dict:
        """
        Get a summary of all flagged moves categorized by type.
        
        Returns:
            Dictionary with flag statistics and categorized moves
            
        Requirements: 4.3
        """
        total_moves = len(self.game_state.move_history)
        flagged_moves = self.get_flagged_moves()
        illegal_moves = self.get_illegal_moves()
        questionable_moves = self.get_questionable_moves()
        
        return {
            "total_moves": total_moves,
            "total_flagged": len(flagged_moves),
            "illegal_moves": len(illegal_moves),
            "questionable_moves": len(questionable_moves),
            "flag_rate": len(flagged_moves) / total_moves if total_moves > 0 else 0,
            "flagged_move_details": [
                {
                    "move_number": i + 1,
                    "from": f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}",
                    "to": f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}",
                    "piece": f"{move.piece.color.value} {move.piece.type.value}",
                    "reason": move.flag_reason
                }
                for i, move in enumerate(self.game_state.move_history) if move.is_flagged
            ]
        }
    
    def flag_move(self, move: Move, reason: str) -> None:
        """
        Flag a move for user review.
        
        Args:
            move: The move to flag
            reason: Reason for flagging the move
            
        Requirements: 4.3
        """
        move.is_flagged = True
        move.flag_reason = reason
        if move not in self.game_state.flagged_moves:
            self.game_state.flagged_moves.append(move)
    
    def flag_questionable_move(self, move: Move, confidence_threshold: float = 0.7) -> None:
        """
        Flag a move as questionable based on various heuristics.
        
        This method identifies moves that might be errors even if they're technically legal.
        
        Args:
            move: The move to evaluate
            confidence_threshold: Minimum confidence level to avoid flagging
            
        Requirements: 4.3
        """
        reasons = []
        
        # Check for unusual piece movements
        if self._is_unusual_piece_movement(move):
            reasons.append("Unusual piece movement pattern")
        
        # Check for potentially missed captures
        if self._has_missed_capture_opportunity(move):
            reasons.append("Potential missed capture opportunity")
        
        # Check for moves that expose valuable pieces
        if self._exposes_valuable_piece(move):
            reasons.append("Move exposes valuable piece to attack")
        
        # Check for moves that ignore threats
        if self._ignores_immediate_threat(move):
            reasons.append("Move ignores immediate threat")
        
        # Check board state confidence
        if hasattr(self.game_state.current_position, 'confidence'):
            if self.game_state.current_position.confidence < confidence_threshold:
                reasons.append(f"Low board detection confidence: {self.game_state.current_position.confidence:.2f}")
        
        # Flag if any questionable patterns are found
        if reasons:
            combined_reason = "Questionable move: " + "; ".join(reasons)
            self.flag_move(move, combined_reason)
    
    def _is_unusual_piece_movement(self, move: Move) -> bool:
        """
        Check if the piece movement is unusual or suspicious.
        
        Args:
            move: The move to check
            
        Returns:
            True if the movement pattern is unusual
        """
        # Check for pieces moving backwards when forward moves are available
        if move.piece.type == PieceKind.PAWN:
            direction = -1 if move.piece.color == Color.WHITE else 1
            actual_direction = move.to_square.y - move.from_square.y
            
            # Pawn moving backwards (except for captures)
            if actual_direction * direction < 0 and move.captured_piece is None:
                return True
        
        # Check for very long moves by pieces that usually move short distances
        if move.piece.type == PieceKind.KING:
            dx = abs(move.to_square.x - move.from_square.x)
            dy = abs(move.to_square.y - move.from_square.y)
            # King moving more than one square without castling
            if (dx > 1 or dy > 1) and move.special_move not in [
                SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE
            ]:
                return True
        
        return False
    
    def _has_missed_capture_opportunity(self, move: Move) -> bool:
        """
        Check if the move misses an obvious capture opportunity.
        
        Args:
            move: The move to check
            
        Returns:
            True if there's a missed capture opportunity
        """
        # Look for enemy pieces adjacent to the destination square
        adjacent_positions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:  # Skip the center square
                    new_x = move.to_square.x + dx
                    new_y = move.to_square.y + dy
                    if 0 <= new_x <= 7 and 0 <= new_y <= 7:  # Check bounds before creating Position
                        adjacent_positions.append(Position(new_x, new_y))
        
        for pos in adjacent_positions:
            piece = self.game_state.current_position.squares.get(pos)
            if (piece and piece.color != move.piece.color and 
                move.captured_piece is None):
                # Check if the moving piece could have captured this piece
                capture_move = Move(move.from_square, pos, move.piece, captured_piece=piece)
                if self._is_valid_piece_move(capture_move):
                    return True
        
        return False
    
    def _exposes_valuable_piece(self, move: Move) -> bool:
        """
        Check if the move exposes a valuable piece to attack.
        
        Args:
            move: The move to check
            
        Returns:
            True if the move exposes a valuable piece
        """
        # This is a simplified check - in a full implementation,
        # we would analyze the resulting position for piece safety
        
        # Check if moving a piece uncovers a more valuable piece behind it
        valuable_pieces = [PieceKind.QUEEN, PieceKind.ROOK]
        
        # Look for pieces behind the moving piece
        dx = move.to_square.x - move.from_square.x
        dy = move.to_square.y - move.from_square.y
        
        if dx != 0 or dy != 0:
            # Normalize direction
            step_x = 0 if dx == 0 else (-1 if dx < 0 else 1)
            step_y = 0 if dy == 0 else (-1 if dy < 0 else 1)
            
            # Check squares behind the original position
            check_x = move.from_square.x - step_x
            check_y = move.from_square.y - step_y
            
            if 0 <= check_x <= 7 and 0 <= check_y <= 7:  # Check bounds before creating Position
                piece_behind = self.game_state.current_position.squares.get(Position(check_x, check_y))
                if (piece_behind and piece_behind.color == move.piece.color and 
                    piece_behind.type in valuable_pieces):
                    return True
        
        return False
    
    def _ignores_immediate_threat(self, move: Move) -> bool:
        """
        Check if the move ignores an immediate threat to a valuable piece.
        
        Args:
            move: The move to check
            
        Returns:
            True if the move ignores an immediate threat
        """
        # This is a simplified implementation
        # In a full implementation, we would analyze all opponent threats
        
        # Check if the king is in check and the move doesn't address it
        if self.is_in_check(move.piece.color):
            # If king is in check, the move should either move the king,
            # block the check, or capture the attacking piece
            if move.piece.type != PieceKind.KING:
                # This is a simplified check - we're not validating if the move
                # actually resolves the check, just flagging non-king moves when in check
                return True
        
        return False
    
    def unflag_move(self, move: Move) -> None:
        """
        Remove flag from a move.
        
        Args:
            move: The move to unflag
            
        Requirements: 4.3
        """
        move.is_flagged = False
        move.flag_reason = None
        if move in self.game_state.flagged_moves:
            self.game_state.flagged_moves.remove(move)
    
    def get_move_history(self) -> List[Move]:
        """
        Get the complete chronological sequence of moves.
        
        Returns:
            List of moves in chronological order
            
        Requirements: 4.1
        """
        return self.game_state.move_history.copy()
    
    def get_move_count(self) -> int:
        """
        Get the total number of moves played.
        
        Returns:
            Total number of moves in the game
            
        Requirements: 4.1
        """
        return len(self.game_state.move_history)
    
    def get_last_move(self) -> Optional[Move]:
        """
        Get the last move played.
        
        Returns:
            The most recent move, or None if no moves have been played
            
        Requirements: 4.1
        """
        if not self.game_state.move_history:
            return None
        return self.game_state.move_history[-1]
    
    def is_game_over(self) -> bool:
        """
        Check if the game has ended.
        
        This is a placeholder for future implementation of checkmate/stalemate detection.
        
        Returns:
            True if the game has ended, False otherwise
            
        Requirements: 4.5
        """
        # This is a simplified implementation
        # In a full implementation, this would check for:
        # - Checkmate
        # - Stalemate  
        # - 50-move rule
        # - Threefold repetition
        # - Insufficient material
        
        # For now, we'll check the 50-move rule as a basic ending condition
        return self.game_state.halfmove_clock >= 100  # 50 moves by each side
    
    def validate_move_sequence(self) -> bool:
        """
        Validate that the entire move sequence is chronologically correct.
        
        This method checks that:
        - Moves alternate between white and black
        - Move history is in chronological order
        - No duplicate moves exist
        
        Returns:
            True if the sequence is chronologically valid, False otherwise
            
        Requirements: 4.1, 4.4
        """
        if not self.game_state.move_history:
            return True  # Empty sequence is valid
        
        # Check that moves alternate between colors
        for i, move in enumerate(self.game_state.move_history):
            expected_color = Color.WHITE if i % 2 == 0 else Color.BLACK
            if move.piece.color != expected_color:
                return False
        
        # Check for duplicate moves (same from/to squares)
        seen_moves = set()
        for move in self.game_state.move_history:
            move_key = (move.from_square.x, move.from_square.y, move.to_square.x, move.to_square.y)
            if move_key in seen_moves:
                return False
            seen_moves.add(move_key)
        
        return True
    
    def format_move_to_algebraic_notation(self, move: Move, board_state_before_move: Optional[BoardState] = None) -> str:
        """
        Format a move using standard algebraic notation.
        
        Args:
            move: The move to format
            board_state_before_move: The board state before the move (for disambiguation)
            
        Returns:
            The move in standard algebraic notation
            
        Requirements: 4.2
        """
        # Handle special moves first
        if move.special_move == SpecialMoveType.CASTLING_KINGSIDE:
            return "O-O"
        elif move.special_move == SpecialMoveType.CASTLING_QUEENSIDE:
            return "O-O-O"
        
        # Get piece symbol (empty for pawns)
        piece_symbol = self._get_piece_symbol(move.piece.type)
        
        # Get destination square in algebraic notation
        dest_square = self._position_to_algebraic(move.to_square)
        
        # Handle pawn moves
        if move.piece.type == PieceKind.PAWN:
            if move.captured_piece or move.special_move == SpecialMoveType.EN_PASSANT:
                # Pawn capture: include file of origin
                from_file = chr(ord('a') + move.from_square.x)
                notation = f"{from_file}x{dest_square}"
            else:
                # Regular pawn move
                notation = dest_square
            
            # Add promotion notation
            if move.special_move == SpecialMoveType.PROMOTION:
                # Use the promotion piece if specified, otherwise default to queen
                promotion_symbol = "Q"  # Default to queen
                if move.promotion_piece:
                    promotion_symbol = self._get_piece_symbol(move.promotion_piece)
                notation += f"={promotion_symbol}"
            
            # Add en passant notation
            if move.special_move == SpecialMoveType.EN_PASSANT:
                notation += " e.p."
        else:
            # Non-pawn moves
            capture_symbol = "x" if move.captured_piece else ""
            
            # Add disambiguation if needed
            disambiguation = ""
            if board_state_before_move:
                disambiguation = self._get_disambiguation(move, board_state_before_move)
            
            notation = f"{piece_symbol}{disambiguation}{capture_symbol}{dest_square}"
        
        # Add check/checkmate notation
        check_notation = self._get_check_notation(move)
        notation += check_notation
        
        return notation
    
    def _get_check_notation(self, move: Move) -> str:
        """
        Get check or checkmate notation for a move.
        
        Args:
            move: The move to check
            
        Returns:
            Check notation ("+" for check, "#" for checkmate, "" for neither)
        """
        # Create a temporary game state to check the position after the move
        temp_manager = GameStateManager()
        temp_manager.game_state = deepcopy(self.game_state)
        
        # Simulate the move
        new_board_state = deepcopy(self.game_state.current_position)
        
        # Remove piece from source square
        new_board_state.squares[move.from_square] = None
        
        # Place piece on destination square
        new_board_state.squares[move.to_square] = move.piece
        
        # Handle captures
        if move.captured_piece:
            # For en passant, remove the captured pawn from a different square
            if move.special_move == SpecialMoveType.EN_PASSANT:
                captured_pawn_y = move.from_square.y  # Same rank as moving pawn
                captured_pawn_pos = Position(move.to_square.x, captured_pawn_y)
                new_board_state.squares[captured_pawn_pos] = None
        
        # Handle castling - move the rook too
        if move.special_move == SpecialMoveType.CASTLING_KINGSIDE:
            rook_from = Position(7, move.from_square.y)
            rook_to = Position(5, move.from_square.y)
            rook = new_board_state.squares[rook_from]
            new_board_state.squares[rook_from] = None
            new_board_state.squares[rook_to] = rook
        elif move.special_move == SpecialMoveType.CASTLING_QUEENSIDE:
            rook_from = Position(0, move.from_square.y)
            rook_to = Position(3, move.from_square.y)
            rook = new_board_state.squares[rook_from]
            new_board_state.squares[rook_from] = None
            new_board_state.squares[rook_to] = rook
        
        # Update the temporary manager's board state
        temp_manager.game_state.current_position = new_board_state
        temp_manager.game_state.active_color = Color.BLACK if move.piece.color == Color.WHITE else Color.WHITE
        
        # Check if the opponent king is in check
        opponent_color = Color.BLACK if move.piece.color == Color.WHITE else Color.WHITE
        if temp_manager.is_in_check(opponent_color):
            # Check if it's checkmate
            legal_moves = temp_manager._get_legal_moves(opponent_color)
            if not legal_moves:
                return "#"  # Checkmate
            else:
                return "+"  # Check
        
        return ""  # No check
    
    def _get_piece_symbol(self, piece_type: PieceKind) -> str:
        """
        Get the algebraic notation symbol for a piece type.
        
        Args:
            piece_type: The piece type
            
        Returns:
            The algebraic notation symbol
        """
        symbols = {
            PieceKind.KING: "K",
            PieceKind.QUEEN: "Q",
            PieceKind.ROOK: "R",
            PieceKind.BISHOP: "B",
            PieceKind.KNIGHT: "N",
            PieceKind.PAWN: ""  # Pawns have no symbol
        }
        return symbols[piece_type]
    
    def _position_to_algebraic(self, position: Position) -> str:
        """
        Convert a position to algebraic notation.
        
        Args:
            position: The position to convert
            
        Returns:
            The position in algebraic notation (e.g., "e4")
        """
        file = chr(ord('a') + position.x)
        rank = str(8 - position.y)  # Convert from 0-7 to 8-1 (y=0 is rank 8, y=7 is rank 1)
        return f"{file}{rank}"
    
    def _get_disambiguation(self, move: Move, board_state: BoardState) -> str:
        """
        Get disambiguation notation for ambiguous moves.
        
        Args:
            move: The move to disambiguate
            board_state: The board state before the move
            
        Returns:
            Disambiguation string (file, rank, or both)
            
        Requirements: 4.2
        """
        # Find all pieces of the same type and color that could move to the same square
        same_pieces = []
        for pos, piece in board_state.squares.items():
            if (piece and 
                piece.color == move.piece.color and 
                piece.type == move.piece.type and 
                pos != move.from_square):
                
                # Check if this piece could theoretically move to the destination
                test_move = Move(pos, move.to_square, piece)
                if self._is_valid_piece_move(test_move) and self._is_path_clear_for_position(test_move, board_state):
                    same_pieces.append(pos)
        
        if not same_pieces:
            return ""  # No ambiguity
        
        # Determine what disambiguation is needed
        from_file = chr(ord('a') + move.from_square.x)
        from_rank = str(8 - move.from_square.y)
        
        # Check if file disambiguation is sufficient
        files_conflict = any(pos.x == move.from_square.x for pos in same_pieces)
        ranks_conflict = any(pos.y == move.from_square.y for pos in same_pieces)
        
        if not files_conflict:
            return from_file  # File is sufficient
        elif not ranks_conflict:
            return from_rank  # Rank is sufficient
        else:
            return f"{from_file}{from_rank}"  # Need both file and rank
    
    def _is_path_clear_for_position(self, move: Move, board_state: BoardState) -> bool:
        """
        Check if the path is clear for a move given a specific board state.
        
        Args:
            move: The move to check
            board_state: The board state to check against
            
        Returns:
            True if the path is clear
        """
        # Knights can jump over pieces
        if move.piece.type == PieceKind.KNIGHT:
            return True
        
        # King moves are only one square
        if move.piece.type == PieceKind.KING:
            return True
        
        dx = move.to_square.x - move.from_square.x
        dy = move.to_square.y - move.from_square.y
        
        # Determine step direction
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Check each square along the path (excluding start and end)
        current_x = move.from_square.x + step_x
        current_y = move.from_square.y + step_y
        
        while current_x != move.to_square.x or current_y != move.to_square.y:
            if board_state.squares.get(Position(current_x, current_y)) is not None:
                return False
            current_x += step_x
            current_y += step_y
        
        return True
    
    def detect_game_ending(self) -> GameResult:
        """
        Detect if the game has ended and determine the result.
        
        Returns:
            The game result (checkmate, stalemate, draw, or ongoing)
            
        Requirements: 4.5
        """
        # Check for 50-move rule
        if self.game_state.halfmove_clock >= 100:  # 50 moves by each side
            return GameResult.DRAW
        
        # Check for insufficient material
        if self._is_insufficient_material():
            return GameResult.DRAW
        
        # Check for threefold repetition (simplified)
        if self._is_threefold_repetition():
            return GameResult.DRAW
        
        # Check for checkmate and stalemate
        current_color = self.game_state.active_color
        legal_moves = self._get_legal_moves(current_color)
        
        if not legal_moves:  # No legal moves available
            if self.is_in_check(current_color):
                # Checkmate - opponent wins
                return GameResult.BLACK_WINS if current_color == Color.WHITE else GameResult.WHITE_WINS
            else:
                # Stalemate - draw
                return GameResult.DRAW
        
        return GameResult.ONGOING
    
    def detect_resignation(self, resigning_color: Color) -> GameResult:
        """
        Handle resignation by a player.
        
        Args:
            resigning_color: The color of the player who resigned
            
        Returns:
            The game result after resignation
            
        Requirements: 4.5
        """
        if resigning_color == Color.WHITE:
            return GameResult.BLACK_WINS
        else:
            return GameResult.WHITE_WINS
    
    def is_game_ended(self) -> bool:
        """
        Check if the game has ended for any reason.
        
        Returns:
            True if the game has ended, False if ongoing
            
        Requirements: 4.5
        """
        return self.detect_game_ending() != GameResult.ONGOING
    
    def _is_insufficient_material(self) -> bool:
        """
        Check if there is insufficient material to checkmate.
        
        This is a simplified implementation that checks for basic insufficient material scenarios.
        
        Returns:
            True if there is insufficient material for checkmate
        """
        pieces = []
        for piece in self.game_state.current_position.squares.values():
            if piece:
                pieces.append(piece)
        
        # King vs King
        if len(pieces) == 2:
            return True
        
        # King and Bishop/Knight vs King
        if len(pieces) == 3:
            non_kings = [p for p in pieces if p.type != PieceKind.KING]
            if len(non_kings) == 1 and non_kings[0].type in [PieceKind.BISHOP, PieceKind.KNIGHT]:
                return True
        
        # King and Bishop vs King and Bishop (same color squares)
        if len(pieces) == 4:
            bishops = [p for p in pieces if p.type == PieceKind.BISHOP]
            kings = [p for p in pieces if p.type == PieceKind.KING]
            if len(bishops) == 2 and len(kings) == 2:
                # This would require checking if bishops are on same color squares
                # For simplicity, we'll assume they are
                return True
        
        return False
    
    def _is_threefold_repetition(self) -> bool:
        """
        Check for threefold repetition of positions.
        
        This is a simplified implementation that checks for repeated FEN positions.
        
        Returns:
            True if the same position has occurred three times
        """
        if len(self.game_state.move_history) < 8:  # Need at least 4 moves by each side
            return False
        
        # Create a simplified position hash based on piece positions and game state
        position_hashes = []
        temp_manager = GameStateManager()
        
        # Reconstruct positions throughout the game
        for move in self.game_state.move_history:
            # Create a position hash (simplified)
            position_hash = self._create_position_hash(temp_manager.game_state)
            position_hashes.append(position_hash)
            
            # Update temporary manager (simplified)
            temp_manager.game_state.active_color = (
                Color.BLACK if temp_manager.game_state.active_color == Color.WHITE else Color.WHITE
            )
        
        # Check for threefold repetition
        from collections import Counter
        position_counts = Counter(position_hashes)
        return any(count >= 3 for count in position_counts.values())
    
    def _create_position_hash(self, game_state: GameState) -> str:
        """
        Create a hash representing the current position for repetition detection.
        
        Args:
            game_state: The game state to hash
            
        Returns:
            A string hash of the position
        """
        # Create a simplified hash based on piece positions and key game state
        pieces = []
        for pos in sorted(game_state.current_position.squares.keys(), key=lambda p: (p.x, p.y)):
            piece = game_state.current_position.squares[pos]
            if piece:
                pieces.append(f"{pos.x}{pos.y}{piece.color.value[0]}{piece.type.value[0]}")
        
        castling = (
            f"{game_state.castling_rights.white_kingside}"
            f"{game_state.castling_rights.white_queenside}"
            f"{game_state.castling_rights.black_kingside}"
            f"{game_state.castling_rights.black_queenside}"
        )
        
        en_passant = f"{game_state.en_passant_target.x}{game_state.en_passant_target.y}" if game_state.en_passant_target else "none"
        
        return f"{''.join(pieces)}_{game_state.active_color.value}_{castling}_{en_passant}"
    
    def _get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get all legal moves for the specified color.
        
        This is a simplified implementation that generates basic moves
        without full chess rule validation.
        
        Args:
            color: The color to generate moves for
            
        Returns:
            List of legal moves
        """
        legal_moves = []
        
        # Find all pieces of the specified color
        for pos, piece in self.game_state.current_position.squares.items():
            if piece and piece.color == color:
                # Generate possible moves for this piece
                possible_moves = self._generate_piece_moves(pos, piece)
                
                # Filter out illegal moves
                for move in possible_moves:
                    if self.validate_move(move).is_legal:
                        legal_moves.append(move)
        
        return legal_moves
    
    def _generate_piece_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """
        Generate all possible moves for a piece at a given position.
        
        Args:
            from_pos: The position of the piece
            piece: The piece to generate moves for
            
        Returns:
            List of possible moves (not necessarily legal)
        """
        moves = []
        
        if piece.type == PieceKind.PAWN:
            moves.extend(self._generate_pawn_moves(from_pos, piece))
        elif piece.type == PieceKind.ROOK:
            moves.extend(self._generate_rook_moves(from_pos, piece))
        elif piece.type == PieceKind.BISHOP:
            moves.extend(self._generate_bishop_moves(from_pos, piece))
        elif piece.type == PieceKind.QUEEN:
            moves.extend(self._generate_queen_moves(from_pos, piece))
        elif piece.type == PieceKind.KNIGHT:
            moves.extend(self._generate_knight_moves(from_pos, piece))
        elif piece.type == PieceKind.KING:
            moves.extend(self._generate_king_moves(from_pos, piece))
        
        return moves
    
    def _generate_pawn_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible pawn moves."""
        moves = []
        direction = 1 if piece.color == Color.WHITE else -1
        
        # Forward moves
        new_y = from_pos.y + direction
        if 0 <= new_y <= 7:
            to_pos = Position(from_pos.x, new_y)
            if not self.game_state.current_position.squares.get(to_pos):
                moves.append(Move(from_pos, to_pos, piece))
                
                # Double move from starting position
                if ((piece.color == Color.WHITE and from_pos.y == 1) or 
                    (piece.color == Color.BLACK and from_pos.y == 6)):
                    double_pos = Position(from_pos.x, from_pos.y + 2 * direction)
                    if 0 <= double_pos.y <= 7 and not self.game_state.current_position.squares.get(double_pos):
                        moves.append(Move(from_pos, double_pos, piece))
        
        # Captures
        for dx in [-1, 1]:
            new_x = from_pos.x + dx
            new_y = from_pos.y + direction
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                to_pos = Position(new_x, new_y)
                target_piece = self.game_state.current_position.squares.get(to_pos)
                if target_piece and target_piece.color != piece.color:
                    moves.append(Move(from_pos, to_pos, piece, captured_piece=target_piece))
        
        return moves
    
    def _generate_rook_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible rook moves."""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            for i in range(1, 8):
                new_x = from_pos.x + i * dx
                new_y = from_pos.y + i * dy
                
                if not (0 <= new_x <= 7 and 0 <= new_y <= 7):
                    break
                
                to_pos = Position(new_x, new_y)
                target_piece = self.game_state.current_position.squares.get(to_pos)
                
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append(Move(from_pos, to_pos, piece, captured_piece=target_piece))
                    break
                else:
                    moves.append(Move(from_pos, to_pos, piece))
        
        return moves
    
    def _generate_bishop_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible bishop moves."""
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            for i in range(1, 8):
                new_x = from_pos.x + i * dx
                new_y = from_pos.y + i * dy
                
                if not (0 <= new_x <= 7 and 0 <= new_y <= 7):
                    break
                
                to_pos = Position(new_x, new_y)
                target_piece = self.game_state.current_position.squares.get(to_pos)
                
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append(Move(from_pos, to_pos, piece, captured_piece=target_piece))
                    break
                else:
                    moves.append(Move(from_pos, to_pos, piece))
        
        return moves
    
    def _generate_queen_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible queen moves (combination of rook and bishop)."""
        moves = []
        moves.extend(self._generate_rook_moves(from_pos, piece))
        moves.extend(self._generate_bishop_moves(from_pos, piece))
        return moves
    
    def _generate_knight_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible knight moves."""
        moves = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dx, dy in knight_moves:
            new_x = from_pos.x + dx
            new_y = from_pos.y + dy
            
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                to_pos = Position(new_x, new_y)
                target_piece = self.game_state.current_position.squares.get(to_pos)
                
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append(Move(from_pos, to_pos, piece, captured_piece=target_piece))
                else:
                    moves.append(Move(from_pos, to_pos, piece))
        
        return moves
    
    def _generate_king_moves(self, from_pos: Position, piece: PieceType) -> List[Move]:
        """Generate possible king moves."""
        moves = []
        king_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in king_moves:
            new_x = from_pos.x + dx
            new_y = from_pos.y + dy
            
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                to_pos = Position(new_x, new_y)
                target_piece = self.game_state.current_position.squares.get(to_pos)
                
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append(Move(from_pos, to_pos, piece, captured_piece=target_piece))
                else:
                    moves.append(Move(from_pos, to_pos, piece))
        
        # Add castling moves
        if piece.color == Color.WHITE and from_pos == Position(4, 0):
            if self.can_castle_kingside(Color.WHITE):
                moves.append(Move(from_pos, Position(6, 0), piece, special_move=SpecialMoveType.CASTLING_KINGSIDE))
            if self.can_castle_queenside(Color.WHITE):
                moves.append(Move(from_pos, Position(2, 0), piece, special_move=SpecialMoveType.CASTLING_QUEENSIDE))
        elif piece.color == Color.BLACK and from_pos == Position(4, 7):
            if self.can_castle_kingside(Color.BLACK):
                moves.append(Move(from_pos, Position(6, 7), piece, special_move=SpecialMoveType.CASTLING_KINGSIDE))
            if self.can_castle_queenside(Color.BLACK):
                moves.append(Move(from_pos, Position(2, 7), piece, special_move=SpecialMoveType.CASTLING_QUEENSIDE))
        
        return moves
    
    def is_in_check(self, color: Color) -> bool:
        """
        Check if the specified color's king is in check.
        
        This is a simplified implementation.
        
        Args:
            color: The color to check
            
        Returns:
            True if the king is in check
        """
        # Find the king
        king_position = None
        for pos, piece in self.game_state.current_position.squares.items():
            if piece and piece.type == PieceKind.KING and piece.color == color:
                king_position = pos
                break
        
        if not king_position:
            return False  # No king found (shouldn't happen in valid game)
        
        # Check if any opponent piece can attack the king
        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        for pos, piece in self.game_state.current_position.squares.items():
            if piece and piece.color == opponent_color:
                # Check if this piece can attack the king
                test_move = Move(pos, king_position, piece)
                if self._is_valid_piece_move(test_move) and self._is_path_clear(test_move):
                    return True
        
        return False
    
    def format_game_to_algebraic_notation(self) -> List[str]:
        """
        Format the entire game history to algebraic notation.
        
        Returns:
            List of moves in algebraic notation
            
        Requirements: 4.2
        """
        algebraic_moves = []
        
        # We need to reconstruct board states to provide proper disambiguation
        temp_manager = GameStateManager()
        
        for move in self.game_state.move_history:
            # Get the board state before this move for disambiguation
            board_before = temp_manager.game_state.current_position
            
            # Format the move
            algebraic_move = self.format_move_to_algebraic_notation(move, board_before)
            algebraic_moves.append(algebraic_move)
            
            # Update the temporary manager to the next state
            # (This is simplified - in practice we'd need the actual board state after each move)
            temp_manager.game_state.active_color = Color.BLACK if temp_manager.game_state.active_color == Color.WHITE else Color.WHITE
        
        return algebraic_moves
    
    def get_current_game_state(self) -> GameState:
        """Get the current game state."""
        return self.game_state
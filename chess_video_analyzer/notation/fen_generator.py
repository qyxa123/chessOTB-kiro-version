"""
FEN (Forsyth-Edwards Notation) generation module.

This module implements the FENGenerator class that creates FEN strings
for chess positions, including all six FEN components.
"""

from typing import Optional, List
from chess_video_analyzer.core.data_models import (
    GameState, BoardState, Position, Color, PieceKind, PieceType
)


class FENGenerator:
    """
    Generates FEN (Forsyth-Edwards Notation) strings for chess positions.
    
    FEN is a standard notation for describing a particular board position
    of a chess game. It includes six components:
    1. Piece placement (from white's perspective)
    2. Active color (whose turn it is)
    3. Castling availability
    4. En passant target square
    5. Halfmove clock (for 50-move rule)
    6. Fullmove number
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    def generate_fen(self, game_state: GameState) -> str:
        """
        Generate a complete FEN string for the given game state.
        
        Args:
            game_state: The current game state
            
        Returns:
            Complete FEN string with all six components
            
        Requirements: 6.1, 6.2
        """
        # Generate each component of the FEN string
        piece_placement = self._encode_piece_placement(game_state.current_position)
        active_color = self._encode_active_color(game_state.active_color)
        castling_rights = self._encode_castling_rights(game_state.castling_rights)
        en_passant = self._encode_en_passant(game_state.en_passant_target)
        halfmove_clock = str(game_state.halfmove_clock)
        fullmove_number = str(game_state.fullmove_number)
        
        # Combine all components with spaces
        fen_string = f"{piece_placement} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"
        
        return fen_string
    
    def generate_fen_for_position(self, board_state: BoardState, 
                                active_color: Color = Color.WHITE,
                                castling_rights: Optional[str] = None,
                                en_passant_target: Optional[Position] = None,
                                halfmove_clock: int = 0,
                                fullmove_number: int = 1) -> str:
        """
        Generate FEN string for a specific board position with custom parameters.
        
        This method is useful for non-standard starting positions.
        
        Args:
            board_state: The board position
            active_color: Color to move (default: WHITE)
            castling_rights: Castling availability string (default: "KQkq")
            en_passant_target: En passant target square (default: None)
            halfmove_clock: Halfmove clock value (default: 0)
            fullmove_number: Fullmove number (default: 1)
            
        Returns:
            Complete FEN string
            
        Requirements: 6.3
        """
        piece_placement = self._encode_piece_placement(board_state)
        active_color_str = self._encode_active_color(active_color)
        
        # Use provided castling rights or default to all available
        if castling_rights is None:
            castling_rights = "KQkq"
        
        en_passant = self._encode_en_passant(en_passant_target)
        halfmove_str = str(halfmove_clock)
        fullmove_str = str(fullmove_number)
        
        fen_string = f"{piece_placement} {active_color_str} {castling_rights} {en_passant} {halfmove_str} {fullmove_str}"
        
        return fen_string
    
    def get_fen_sequence(self, game_states: List[GameState]) -> List[str]:
        """
        Generate FEN strings for a sequence of game states.
        
        Args:
            game_states: List of game states in chronological order
            
        Returns:
            List of FEN strings corresponding to each game state
            
        Requirements: 6.1
        """
        return [self.generate_fen(state) for state in game_states]
    
    def _encode_piece_placement(self, board_state: BoardState) -> str:
        """
        Encode the piece placement component of FEN.
        
        This represents the board from white's perspective, starting from
        rank 8 (top) and going to rank 1 (bottom), left to right.
        
        Args:
            board_state: The current board state
            
        Returns:
            Piece placement string (first component of FEN)
        """
        fen_rows = []
        
        # Process each rank from 8 to 1 (y coordinates 0 to 7)
        for rank_number in range(8, 0, -1):  # Rank 8 to 1
            y = 8 - rank_number  # Convert rank to y-coordinate (rank 8 = y=0, rank 1 = y=7)
            row_string = ""
            empty_count = 0
            
            # Process each file from a to h (x coordinates 0 to 7)
            for file in range(8):
                position = Position(file, y)
                piece = board_state.squares.get(position)
                
                if piece is None:
                    empty_count += 1
                else:
                    # Add empty squares count if any
                    if empty_count > 0:
                        row_string += str(empty_count)
                        empty_count = 0
                    
                    # Add piece symbol
                    row_string += self._get_fen_piece_symbol(piece)
            
            # Add remaining empty squares at end of rank
            if empty_count > 0:
                row_string += str(empty_count)
            
            fen_rows.append(row_string)
        
        # Join ranks with '/' separator
        return "/".join(fen_rows)
    
    def _get_fen_piece_symbol(self, piece: PieceType) -> str:
        """
        Get the FEN symbol for a piece.
        
        White pieces are uppercase, black pieces are lowercase.
        
        Args:
            piece: The piece to get symbol for
            
        Returns:
            FEN piece symbol
        """
        symbols = {
            PieceKind.KING: "K",
            PieceKind.QUEEN: "Q",
            PieceKind.ROOK: "R",
            PieceKind.BISHOP: "B",
            PieceKind.KNIGHT: "N",
            PieceKind.PAWN: "P"
        }
        
        symbol = symbols[piece.type]
        
        # Black pieces are lowercase
        if piece.color == Color.BLACK:
            symbol = symbol.lower()
        
        return symbol
    
    def _encode_active_color(self, active_color: Color) -> str:
        """
        Encode the active color component of FEN.
        
        Args:
            active_color: The color to move
            
        Returns:
            "w" for white, "b" for black
        """
        return "w" if active_color == Color.WHITE else "b"
    
    def _encode_castling_rights(self, castling_rights) -> str:
        """
        Encode the castling rights component of FEN.
        
        Args:
            castling_rights: The castling rights object
            
        Returns:
            Castling rights string (e.g., "KQkq", "Kq", "-")
        """
        rights = ""
        
        # White castling rights (uppercase)
        if castling_rights.white_kingside:
            rights += "K"
        if castling_rights.white_queenside:
            rights += "Q"
        
        # Black castling rights (lowercase)
        if castling_rights.black_kingside:
            rights += "k"
        if castling_rights.black_queenside:
            rights += "q"
        
        # Return "-" if no castling rights available
        return rights if rights else "-"
    
    def _encode_en_passant(self, en_passant_target: Optional[Position]) -> str:
        """
        Encode the en passant target square component of FEN.
        
        Args:
            en_passant_target: The en passant target square, if any
            
        Returns:
            En passant target in algebraic notation, or "-" if none
        """
        if en_passant_target is None:
            return "-"
        
        # Convert position to algebraic notation
        file = chr(ord('a') + en_passant_target.x)
        rank = str(8 - en_passant_target.y)
        
        return f"{file}{rank}"
    
    def validate_fen(self, fen_string: str) -> bool:
        """
        Validate that a FEN string is properly formatted.
        
        Args:
            fen_string: The FEN string to validate
            
        Returns:
            True if the FEN string is valid, False otherwise
        """
        try:
            parts = fen_string.strip().split()
            
            # Must have exactly 6 components
            if len(parts) != 6:
                return False
            
            piece_placement, active_color, castling, en_passant, halfmove, fullmove = parts
            
            # Validate piece placement
            if not self._validate_piece_placement(piece_placement):
                return False
            
            # Validate active color
            if active_color not in ["w", "b"]:
                return False
            
            # Validate castling rights
            if not self._validate_castling_rights(castling):
                return False
            
            # Validate en passant
            if not self._validate_en_passant(en_passant):
                return False
            
            # Validate halfmove clock
            try:
                halfmove_int = int(halfmove)
                if halfmove_int < 0:
                    return False
            except ValueError:
                return False
            
            # Validate fullmove number
            try:
                fullmove_int = int(fullmove)
                if fullmove_int < 1:
                    return False
            except ValueError:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_piece_placement(self, piece_placement: str) -> bool:
        """
        Validate the piece placement component of FEN.
        
        Args:
            piece_placement: The piece placement string
            
        Returns:
            True if valid, False otherwise
        """
        ranks = piece_placement.split("/")
        
        # Must have exactly 8 ranks
        if len(ranks) != 8:
            return False
        
        valid_pieces = set("KQRBNPkqrbnp")
        
        for rank in ranks:
            file_count = 0
            
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    empty_count = int(char)
                    if empty_count < 1 or empty_count > 8:
                        return False
                    file_count += empty_count
                elif char in valid_pieces:
                    # Piece
                    file_count += 1
                else:
                    # Invalid character
                    return False
            
            # Each rank must have exactly 8 files
            if file_count != 8:
                return False
        
        return True
    
    def _validate_castling_rights(self, castling: str) -> bool:
        """
        Validate the castling rights component of FEN.
        
        Args:
            castling: The castling rights string
            
        Returns:
            True if valid, False otherwise
        """
        if castling == "-":
            return True
        
        # Valid castling characters
        valid_chars = set("KQkq")
        
        # Check that all characters are valid and no duplicates
        seen = set()
        for char in castling:
            if char not in valid_chars or char in seen:
                return False
            seen.add(char)
        
        return True
    
    def _validate_en_passant(self, en_passant: str) -> bool:
        """
        Validate the en passant component of FEN.
        
        Args:
            en_passant: The en passant string
            
        Returns:
            True if valid, False otherwise
        """
        if en_passant == "-":
            return True
        
        # Must be exactly 2 characters: file + rank
        if len(en_passant) != 2:
            return False
        
        file, rank = en_passant[0], en_passant[1]
        
        # File must be a-h
        if file not in "abcdefgh":
            return False
        
        # Rank must be 1-8
        if rank not in "12345678":
            return False
        
        # En passant squares are only valid on ranks 3 and 6
        if rank not in "36":
            return False
        
        return True
    
    def parse_fen(self, fen_string: str) -> dict:
        """
        Parse a FEN string into its components.
        
        Args:
            fen_string: The FEN string to parse
            
        Returns:
            Dictionary with FEN components
            
        Raises:
            ValueError: If FEN string is invalid
        """
        if not self.validate_fen(fen_string):
            raise ValueError(f"Invalid FEN string: {fen_string}")
        
        parts = fen_string.strip().split()
        piece_placement, active_color, castling, en_passant, halfmove, fullmove = parts
        
        return {
            "piece_placement": piece_placement,
            "active_color": Color.WHITE if active_color == "w" else Color.BLACK,
            "castling_rights": castling,
            "en_passant_target": None if en_passant == "-" else self._parse_algebraic_position(en_passant),
            "halfmove_clock": int(halfmove),
            "fullmove_number": int(fullmove)
        }
    
    def _parse_algebraic_position(self, algebraic: str) -> Position:
        """
        Parse algebraic notation into a Position object.
        
        Args:
            algebraic: Algebraic notation (e.g., "e4")
            
        Returns:
            Position object
        """
        file = ord(algebraic[0]) - ord('a')
        rank = 8 - int(algebraic[1])
        
        return Position(file, rank)
    
    def get_standard_starting_fen(self) -> str:
        """
        Get the FEN string for the standard chess starting position.
        
        Returns:
            FEN string for starting position
        """
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
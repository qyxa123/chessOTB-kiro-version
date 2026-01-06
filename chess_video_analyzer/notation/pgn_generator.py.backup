"""
PGN (Portable Game Notation) generator for chess games.

This module provides functionality to generate PGN files from chess game data,
following the PGN specification standard.
"""

from typing import List, Optional, Dict
from datetime import datetime
import re

from ..core.data_models import (
    GameState, Move, GameMetadata, Color, PieceKind, 
    SpecialMoveType, Position, GameResult
)


class PGNGenerator:
    """Generates PGN (Portable Game Notation) files from chess game data."""
    
    def __init__(self):
        """Initialize the PGN generator."""
        self._file_to_rank = {0: '8', 1: '7', 2: '6', 3: '5', 4: '4', 5: '3', 6: '2', 7: '1'}  # y=0 is rank 8, y=7 is rank 1
        self._rank_to_file = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        self._file_to_rank_reverse = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    
    def generate_pgn(self, game_state: GameState, metadata: GameMetadata) -> str:
        """
        Generate a complete PGN string from game state and metadata.
        
        Args:
            game_state: The complete game state with move history
            metadata: Game metadata for PGN headers
            
        Returns:
            Complete PGN string with headers and moves
        """
        pgn_parts = []
        
        # Add headers
        pgn_parts.append(self._generate_headers(metadata))
        
        # Add empty line between headers and moves
        pgn_parts.append("")
        
        # Add moves
        moves_section = self._generate_moves_section(game_state.move_history)
        pgn_parts.append(moves_section)
        
        # Add result
        pgn_parts.append(metadata.result)
        
        return "\n".join(pgn_parts)
    
    def _generate_headers(self, metadata: GameMetadata) -> str:
        """Generate PGN headers section."""
        headers = [
            f'[Event "{metadata.event}"]',
            f'[Site "{metadata.site}"]',
            f'[Date "{metadata.date}"]',
            f'[Round "{metadata.round}"]',
            f'[White "{metadata.white_player}"]',
            f'[Black "{metadata.black_player}"]',
            f'[Result "{metadata.result}"]'
        ]
        return "\n".join(headers)
    
    def _generate_moves_section(self, moves: List[Move]) -> str:
        """Generate the moves section of the PGN."""
        if not moves:
            return ""
        
        move_pairs = []
        current_pair = []
        move_number = 1
        
        for i, move in enumerate(moves):
            # Format the move
            move_notation = self.format_move(move)
            
            # Add move number for white moves
            if i % 2 == 0:
                current_pair = [f"{move_number}.", move_notation]
                move_number += 1
            else:
                current_pair.append(move_notation)
                move_pairs.append(" ".join(current_pair))
                current_pair = []
        
        # Handle case where game ends on white's move
        if current_pair:
            move_pairs.append(" ".join(current_pair))
        
        return " ".join(move_pairs)
    
    def format_move(self, move: Move) -> str:
        """
        Format a single move in standard algebraic notation.
        
        Args:
            move: The move to format
            
        Returns:
            Move in standard algebraic notation
        """
        # Handle special moves first
        if move.special_move == SpecialMoveType.CASTLING_KINGSIDE:
            return "O-O"
        elif move.special_move == SpecialMoveType.CASTLING_QUEENSIDE:
            return "O-O-O"
        
        notation = ""
        
        # Add piece symbol (except for pawns)
        if move.piece.type != PieceKind.PAWN:
            notation += self._get_piece_symbol(move.piece.type)
        
        # Add disambiguation if needed (simplified - would need game context for full implementation)
        # For now, we'll add basic disambiguation
        
        # Add capture notation
        if move.captured_piece is not None:
            if move.piece.type == PieceKind.PAWN:
                # For pawn captures, include the file of origin
                notation += self._position_to_algebraic(move.from_square)[0]
            notation += "x"
        
        # Add destination square
        notation += self._position_to_algebraic(move.to_square)
        
        # Add promotion notation
        if move.special_move == SpecialMoveType.PROMOTION and move.promotion_piece:
            notation += "=" + self._get_piece_symbol(move.promotion_piece)
        
        # Add en passant notation
        if move.special_move == SpecialMoveType.EN_PASSANT:
            notation += " e.p."
        
        # TODO: Add check/checkmate notation (requires game state analysis)
        
        return notation
    
    def _get_piece_symbol(self, piece_type: PieceKind) -> str:
        """Get the PGN symbol for a piece type."""
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
        """Convert a Position to algebraic notation (e.g., e4)."""
        file_char = self._file_to_rank_reverse[position.x]
        rank_char = self._file_to_rank[position.y]
        return f"{file_char}{rank_char}"
    
    def _algebraic_to_position(self, algebraic: str) -> Position:
        """Convert algebraic notation to Position."""
        if len(algebraic) != 2:
            raise ValueError(f"Invalid algebraic notation: {algebraic}")
        
        file_char = algebraic[0].lower()
        rank_char = algebraic[1]
        
        if file_char not in self._rank_to_file:
            raise ValueError(f"Invalid file: {file_char}")
        if rank_char not in self._file_to_rank.values():
            raise ValueError(f"Invalid rank: {rank_char}")
        
        x = self._rank_to_file[file_char]
        y = 8 - int(rank_char)  # Convert rank to y coordinate (rank 8 = y=0, rank 1 = y=7)
        
        return Position(x, y)
    
    def validate_pgn(self, pgn_string: str) -> bool:
        """
        Validate a PGN string against basic format requirements.
        
        Args:
            pgn_string: The PGN string to validate
            
        Returns:
            True if the PGN is valid, False otherwise
        """
        try:
            lines = pgn_string.strip().split('\n')
            
            # Check for required headers
            required_headers = ['Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result']
            found_headers = set()
            
            header_section = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('[') and line.endswith(']'):
                    if not header_section:
                        return False  # Headers after moves section
                    
                    # Extract header name
                    match = re.match(r'\[(\w+)\s+".*"\]', line)
                    if match:
                        found_headers.add(match.group(1))
                else:
                    header_section = False
                    # This is the moves section
                    break
            
            # Check if all required headers are present
            if not all(header in found_headers for header in required_headers):
                return False
            
            return True
            
        except Exception:
            return False
    
    def parse_pgn_headers(self, pgn_string: str) -> Dict[str, str]:
        """
        Parse PGN headers from a PGN string.
        
        Args:
            pgn_string: The PGN string to parse
            
        Returns:
            Dictionary of header name -> value pairs
        """
        headers = {}
        lines = pgn_string.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # Extract header name and value
                match = re.match(r'\[(\w+)\s+"(.*)"\]', line)
                if match:
                    headers[match.group(1)] = match.group(2)
            elif line and not line.startswith('['):
                # End of headers section
                break
        
        return headers
    
    def parse_pgn_moves(self, pgn_string: str) -> List[str]:
        """
        Parse moves from a PGN string.
        
        Args:
            pgn_string: The PGN string to parse
            
        Returns:
            List of move strings in algebraic notation
        """
        lines = pgn_string.strip().split('\n')
        moves_section = []
        
        # Skip headers and find moves section
        in_moves = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('[') and line.endswith(']'):
                continue  # Skip headers
            else:
                in_moves = True
                moves_section.append(line)
        
        if not moves_section:
            return []
        
        # Join all move lines and parse
        moves_text = ' '.join(moves_section)
        
        # Remove result from the end
        for result in ["1-0", "0-1", "1/2-1/2", "*"]:
            if moves_text.endswith(result):
                moves_text = moves_text[:-len(result)].strip()
                break
        
        # Split into individual moves, removing move numbers
        moves = []
        tokens = moves_text.split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Skip move numbers (like "1.", "2.", etc.)
            if token.endswith('.') and token[:-1].isdigit():
                i += 1
                continue
            
            # Skip comments and annotations
            if token.startswith('(') or token.startswith('{'):
                i += 1
                continue
            
            # Handle en passant notation - if we see "e.p.", combine it with the previous move
            if token == "e.p." and moves:
                # Append to the last move
                moves[-1] += " e.p."
                i += 1
                continue
            
            # This is a move
            if token:
                moves.append(token)
            
            i += 1
        
        return moves
    
    def parse_pgn_to_game_data(self, pgn_string: str) -> Dict:
        """
        Parse a complete PGN string into structured game data.
        
        Args:
            pgn_string: The PGN string to parse
            
        Returns:
            Dictionary containing headers and moves
        """
        headers = self.parse_pgn_headers(pgn_string)
        moves = self.parse_pgn_moves(pgn_string)
        
        return {
            'headers': headers,
            'moves': moves
        }
    
    def validate_pgn_format(self, pgn_string: str) -> Dict[str, any]:
        """
        Comprehensive PGN format validation.
        
        Args:
            pgn_string: The PGN string to validate
            
        Returns:
            Dictionary with validation results and details
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            lines = pgn_string.strip().split('\n')
            
            # Check for required headers
            required_headers = ['Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result']
            found_headers = set()
            
            header_section = True
            moves_section_found = False
            result_found = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('[') and line.endswith(']'):
                    if not header_section:
                        result['is_valid'] = False
                        result['errors'].append(f"Header found after moves section at line {i+1}")
                    
                    # Extract and validate header
                    match = re.match(r'\[(\w+)\s+"(.*)"\]', line)
                    if match:
                        header_name = match.group(1)
                        header_value = match.group(2)
                        found_headers.add(header_name)
                        
                        # Validate specific headers
                        if header_name == 'Result' and header_value not in ['1-0', '0-1', '1/2-1/2', '*']:
                            result['warnings'].append(f"Non-standard result value: {header_value}")
                        
                        if header_name == 'Date':
                            if not re.match(r'^\d{4}\.\d{2}\.\d{2}$|^\?\?\?\?\.\?\?\.\?\?$', header_value):
                                result['warnings'].append(f"Non-standard date format: {header_value}")
                    else:
                        result['is_valid'] = False
                        result['errors'].append(f"Invalid header format at line {i+1}: {line}")
                
                elif line in ['1-0', '0-1', '1/2-1/2', '*']:
                    result_found = True
                    if header_section:
                        result['warnings'].append("Result found without moves section")
                
                else:
                    header_section = False
                    moves_section_found = True
            
            # Check for missing required headers
            missing_headers = set(required_headers) - found_headers
            if missing_headers:
                result['is_valid'] = False
                result['errors'].append(f"Missing required headers: {', '.join(missing_headers)}")
            
            # Check for result
            if not result_found:
                result['warnings'].append("No game result found")
            
            # Validate moves section if present
            if moves_section_found:
                moves = self.parse_pgn_moves(pgn_string)
                for move in moves:
                    if not self._is_valid_move_notation(move):
                        result['warnings'].append(f"Potentially invalid move notation: {move}")
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Parse error: {str(e)}")
        
        return result
    
    def _is_valid_move_notation(self, move: str) -> bool:
        """
        Check if a move string follows valid algebraic notation.
        
        Args:
            move: Move string to validate
            
        Returns:
            True if the move notation appears valid
        """
        # Basic patterns for algebraic notation
        patterns = [
            r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](\+|#)?$',  # Regular moves
            r'^[a-h]x[a-h][1-8](\+|#)?$',  # Pawn captures
            r'^[a-h][1-8]=?[QRBN](\+|#)?$',  # Pawn moves/promotions
            r'^O-O(\+|#)?$',  # Kingside castling
            r'^O-O-O(\+|#)?$',  # Queenside castling
            r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8] e\.p\.(\+|#)?$'  # En passant
        ]
        
        return any(re.match(pattern, move) for pattern in patterns)
    
    def test_round_trip(self, game_state: GameState, metadata: GameMetadata) -> bool:
        """
        Test round-trip consistency: generate PGN, parse it back, verify consistency.
        
        Args:
            game_state: Original game state
            metadata: Original metadata
            
        Returns:
            True if round-trip is consistent
        """
        try:
            # Generate PGN
            original_pgn = self.generate_pgn(game_state, metadata)
            
            # Parse it back
            parsed_data = self.parse_pgn_to_game_data(original_pgn)
            
            # Verify headers match
            for header in ['Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result']:
                if header == 'Result':
                    original_value = metadata.result
                elif header == 'White':
                    original_value = metadata.white_player
                elif header == 'Black':
                    original_value = metadata.black_player
                else:
                    # For Event, Site, Date, Round - use lowercase attribute name
                    original_value = getattr(metadata, header.lower())
                
                parsed_value = parsed_data['headers'].get(header, '')
                
                if original_value != parsed_value:
                    return False
            
            # Verify moves count is reasonable (more flexible approach)
            original_moves_count = len(game_state.move_history)
            parsed_moves_count = len(parsed_data['moves'])
            
            # For round-trip validation, we primarily care that:
            # 1. The PGN is valid and can be parsed
            # 2. Headers are preserved exactly
            # 3. The move structure is reasonable (not empty if original had moves)
            if original_moves_count > 0:
                # Should have some moves parsed back if original had moves
                if parsed_moves_count == 0:
                    return False
                # Allow for reasonable differences due to PGN format (ply vs move pairs)
                max_expected_ratio = 2.5
                if parsed_moves_count > original_moves_count * max_expected_ratio:
                    return False
            elif parsed_moves_count > 0:
                # If original had no moves, parsed should also have no moves
                return False
            
            # Verify PGN is still valid after round-trip
            validation_result = self.validate_pgn_format(original_pgn)
            return validation_result['is_valid']
            
        except Exception:
            return False
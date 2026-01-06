"""
Move tracking and detection module for chess video analysis.

This module implements the MoveTracker class that compares board states between frames
to detect piece movements, captures, and other chess moves.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from chess_video_analyzer.core.data_models import (
    Position, PieceType, Move, BoardState, SpecialMoveType, Color, PieceKind
)


@dataclass
class MoveCandidate:
    """Represents a potential move detected between board states."""
    from_square: Position
    to_square: Position
    piece: PieceType
    captured_piece: Optional[PieceType] = None
    confidence: float = 1.0
    special_move: Optional[SpecialMoveType] = None


class MoveTracker:
    """
    Tracks piece movements and detects chess moves by comparing board states.
    
    This class analyzes differences between consecutive board states to identify:
    - Regular piece movements
    - Capture events
    - Piece disappearances
    - Special moves (castling, en passant, promotion)
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the move tracker.
        
        Args:
            confidence_threshold: Minimum confidence required to consider a move valid
        """
        self.confidence_threshold = confidence_threshold
        self.move_history: List[Move] = []
    
    def detect_move(self, previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """
        Detect a chess move by comparing two board states.
        
        Args:
            previous_state: The board state before the move
            current_state: The board state after the move
            
        Returns:
            The detected move, or None if no clear move is found
            
        Requirements: 3.2, 3.3
        """
        if not self._validate_board_states(previous_state, current_state):
            return None
        
        # Find all position changes
        changes = []
        disappeared_pieces = []  # (position, piece)
        appeared_pieces = []     # (position, piece)
        
        for pos in previous_state.squares:
            prev_piece = previous_state.squares.get(pos)
            curr_piece = current_state.squares.get(pos)
            
            if prev_piece != curr_piece:
                changes.append((pos, prev_piece, curr_piece))
                
                if prev_piece is not None and curr_piece is None:
                    disappeared_pieces.append((pos, prev_piece))
                elif prev_piece is None and curr_piece is not None:
                    appeared_pieces.append((pos, curr_piece))
        
        # Simple move detection: one piece disappeared, one appeared
        if len(disappeared_pieces) == 1 and len(appeared_pieces) == 1:
            from_pos, piece = disappeared_pieces[0]
            to_pos, moved_piece = appeared_pieces[0]
            
            # Check if it's the same piece (or similar enough)
            if self._pieces_match(piece, moved_piece):
                # Create the move
                captured_piece = previous_state.squares.get(to_pos)
                
                move = Move(
                    from_square=from_pos,
                    to_square=to_pos,
                    piece=piece,
                    captured_piece=captured_piece
                )
                
                # Only do basic legality check here, not full chess rules
                if self._is_basic_legal_move(from_pos, to_pos, piece, previous_state):
                    self.move_history.append(move)
                    return move
        
        # Handle more complex cases
        if len(changes) == 2:
            return self._handle_two_position_change(changes, previous_state, current_state)
        elif len(changes) > 2:
            return self._handle_multiple_position_changes(changes, previous_state, current_state)
        
        return None
    
    def _pieces_match(self, piece1: PieceType, piece2: PieceType) -> bool:
        """Check if two pieces are the same (allowing for some detection uncertainty)."""
        if piece1 is None or piece2 is None:
            return piece1 == piece2
        
        # Check both color and type match (more strict than before)
        return piece1.color == piece2.color and piece1.type == piece2.type
    
    def _is_legal_move(self, from_pos: Position, to_pos: Position, piece: PieceType, board_state: BoardState) -> bool:
        """
        Check if a move is legal according to chess rules.
        
        Args:
            from_pos: Starting position
            to_pos: Ending position  
            piece: The piece being moved
            board_state: Current board state
            
        Returns:
            True if the move is legal for this piece type
        """
        # Calculate movement deltas
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        # No movement is not a valid move
        if dx == 0 and dy == 0:
            return False
        
        # Check piece-specific movement rules
        if piece.type == PieceKind.PAWN:
            return self._is_legal_pawn_move(from_pos, to_pos, piece, board_state, dx, dy)
        elif piece.type == PieceKind.ROOK:
            return (dx == 0 or dy == 0) and self._is_path_clear_simple(from_pos, to_pos, board_state)
        elif piece.type == PieceKind.BISHOP:
            return (abs(dx) == abs(dy)) and self._is_path_clear_simple(from_pos, to_pos, board_state)
        elif piece.type == PieceKind.QUEEN:
            return ((dx == 0 or dy == 0) or (abs(dx) == abs(dy))) and self._is_path_clear_simple(from_pos, to_pos, board_state)
        elif piece.type == PieceKind.KING:
            return abs(dx) <= 1 and abs(dy) <= 1
        elif piece.type == PieceKind.KNIGHT:
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        return False
    
    def _is_legal_pawn_move(self, from_pos: Position, to_pos: Position, piece: PieceType, 
                           board_state: BoardState, dx: int, dy: int) -> bool:
        """Check if a pawn move is legal."""
        # Determine direction based on color
        if piece.color == Color.WHITE:
            forward_direction = -1  # White moves up (y decreases)
            starting_rank = 6  # White pawns start at y=6
        else:
            forward_direction = 1   # Black moves down (y increases)  
            starting_rank = 1   # Black pawns start at y=1
        
        # Check if moving in correct direction
        if (dy * forward_direction) <= 0:
            return False
        
        # Straight move (no capture)
        if dx == 0:
            # One square forward
            if abs(dy) == 1:
                return board_state.squares.get(to_pos) is None
            # Two squares forward from starting position
            elif abs(dy) == 2 and from_pos.y == starting_rank:
                return (board_state.squares.get(to_pos) is None and 
                       board_state.squares.get(Position(from_pos.x, from_pos.y + forward_direction)) is None)
            return False
        
        # Diagonal capture
        elif abs(dx) == 1 and abs(dy) == 1:
            target_piece = board_state.squares.get(to_pos)
            return target_piece is not None and target_piece.color != piece.color
        
        return False
    
    def _is_path_clear_simple(self, from_pos: Position, to_pos: Position, board_state: BoardState) -> bool:
        """Check if path between positions is clear (simplified version)."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        # Adjacent squares - path is clear
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        
        # Determine step direction
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Check each square along the path (excluding start and end)
        current_x = from_pos.x + step_x
        current_y = from_pos.y + step_y
        
        while current_x != to_pos.x or current_y != to_pos.y:
            if board_state.squares.get(Position(current_x, current_y)) is not None:
                return False  # Path is blocked
            current_x += step_x
            current_y += step_y
        
        return True
    
    def _find_move_for_appeared_piece(self, to_pos: Position, piece: PieceType, 
                                    previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """Find where a piece came from when it appeared at a new position."""
        # Look for positions where this piece type disappeared
        for pos, prev_piece in previous_state.squares.items():
            if prev_piece == piece and pos != to_pos:
                curr_piece = current_state.squares.get(pos)
                if curr_piece != prev_piece:
                    # This piece moved from pos to to_pos
                    captured_piece = previous_state.squares.get(to_pos)
                    confidence = min(previous_state.confidence, current_state.confidence)
                    
                    special_move = self._detect_special_move(
                        pos, to_pos, piece, previous_state, current_state
                    )
                    
                    move = Move(
                        from_square=pos,
                        to_square=to_pos,
                        piece=piece,
                        captured_piece=captured_piece,
                        special_move=special_move,
                        promotion_piece=current_state.squares.get(to_pos).type if special_move == SpecialMoveType.PROMOTION else None
                    )
                    self.move_history.append(move)
                    return move
        return None
    
    def _find_move_for_disappeared_piece(self, from_pos: Position, piece: PieceType,
                                       previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """Find where a piece went when it disappeared from a position."""
        # Look for positions where this piece type appeared or where a different piece was replaced
        candidates = []
        
        for pos, curr_piece in current_state.squares.items():
            if pos != from_pos:
                prev_piece = previous_state.squares.get(pos)
                
                # Case 1: This piece type appeared at a new position (prev_piece was different or None)
                if curr_piece == piece and prev_piece != curr_piece:
                    captured_piece = prev_piece
                    confidence = min(previous_state.confidence, current_state.confidence)
                    
                    # Calculate distance to help with disambiguation
                    distance = abs(pos.x - from_pos.x) + abs(pos.y - from_pos.y)
                    
                    special_move = self._detect_special_move(
                        from_pos, pos, piece, previous_state, current_state
                    )
                    
                    candidates.append({
                        'move': Move(
                            from_square=from_pos,
                            to_square=pos,
                            piece=piece,
                            captured_piece=captured_piece,
                            special_move=special_move,
                            promotion_piece=current_state.squares.get(pos).type if special_move == SpecialMoveType.PROMOTION else None
                        ),
                        'distance': distance,
                        'confidence': confidence
                    })
                
                # Case 2: The piece might have moved to capture an identical piece
                # In this case, the position would still have the same piece type, but it's actually our moved piece
                elif (curr_piece == piece and prev_piece == piece and pos != from_pos and
                      self._could_be_capture_of_identical_piece(from_pos, pos, piece, previous_state, current_state)):
                    captured_piece = prev_piece
                    confidence = min(previous_state.confidence, current_state.confidence)
                    
                    # Calculate distance to help with disambiguation
                    distance = abs(pos.x - from_pos.x) + abs(pos.y - from_pos.y)
                    
                    special_move = self._detect_special_move(
                        from_pos, pos, piece, previous_state, current_state
                    )
                    
                    candidates.append({
                        'move': Move(
                            from_square=from_pos,
                            to_square=pos,
                            piece=piece,
                            captured_piece=captured_piece,
                            special_move=special_move,
                            promotion_piece=current_state.squares.get(pos).type if special_move == SpecialMoveType.PROMOTION else None
                        ),
                        'distance': distance,
                        'confidence': confidence
                    })
        
        # If we have candidates, select the best one
        if candidates:
            # Prioritize moves where the piece actually changed (Case 1) over identical piece captures (Case 2)
            # Then prioritize by shorter distance and higher confidence
            def candidate_score(candidate):
                move = candidate['move']
                # Prefer moves where there was actually a capture (captured_piece is not None)
                capture_bonus = 2.0 if move.captured_piece is not None else 0.0
                # Prefer moves where something actually changed at the destination
                change_bonus = 1.0 if move.captured_piece != piece else 0.5
                # Prefer shorter distances
                distance_penalty = candidate['distance'] * 0.1
                # Include confidence
                confidence_bonus = candidate['confidence']
                
                return capture_bonus + change_bonus + confidence_bonus - distance_penalty
            
            best_candidate = max(candidates, key=candidate_score)
            move = best_candidate['move']
            self.move_history.append(move)
            return move
        
        return None
    
    def _could_be_capture_of_identical_piece(self, from_pos: Position, to_pos: Position, piece: PieceType,
                                           previous_state: BoardState, current_state: BoardState) -> bool:
        """
        Determine if a piece could have moved to capture an identical piece.
        
        This is a heuristic to handle cases where a piece captures another piece of the same type.
        We use distance and piece count to make this determination.
        """
        # Count total pieces of this type in both states
        prev_count = sum(1 for p in previous_state.squares.values() if p == piece)
        curr_count = sum(1 for p in current_state.squares.values() if p == piece)
        
        # If the count decreased by 1, it's likely a capture occurred
        if prev_count == curr_count + 1:
            # Calculate distance - closer moves are more likely
            distance = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)
            # Allow moves up to a reasonable distance (e.g., 7 squares for full board moves)
            result = distance <= 7
            return result
        
        return False
    
    def _find_identical_piece_capture(self, previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """
        Find moves where a piece captures an identical piece.
        
        This handles cases where no position changes are detected because a piece moved
        to capture an identical piece, making the board look unchanged except for piece count.
        """
        # Count pieces by type in both states
        prev_counts = {}
        curr_counts = {}
        
        for piece in previous_state.squares.values():
            if piece is not None:
                key = (piece.color, piece.type)
                prev_counts[key] = prev_counts.get(key, 0) + 1
        
        for piece in current_state.squares.values():
            if piece is not None:
                key = (piece.color, piece.type)
                curr_counts[key] = curr_counts.get(key, 0) + 1
        
        # Find piece types that decreased in count
        for piece_key, prev_count in prev_counts.items():
            curr_count = curr_counts.get(piece_key, 0)
            if prev_count > curr_count:
                # This piece type decreased - find potential moves
                color, piece_type = piece_key
                target_piece = PieceType(color, piece_type)
                
                # Find all positions with this piece type in both states
                prev_positions = [pos for pos, piece in previous_state.squares.items() if piece == target_piece]
                curr_positions = [pos for pos, piece in current_state.squares.items() if piece == target_piece]
                
                # Find positions that had the piece in previous state but not in current state
                disappeared_positions = [pos for pos in prev_positions if pos not in curr_positions]
                
                # Find the most likely move by looking for the closest remaining piece
                best_move = None
                best_distance = float('inf')
                
                for from_pos in disappeared_positions:
                    for to_pos in curr_positions:
                        if from_pos != to_pos:
                            # Check if this could be a valid move
                            distance = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)
                            if distance < best_distance and distance <= 7:  # Reasonable move distance
                                # Check if there was a piece at the destination to capture
                                captured_piece = previous_state.squares.get(to_pos)
                                if captured_piece == target_piece:
                                    # This looks like a capture of an identical piece
                                    confidence = min(previous_state.confidence, current_state.confidence)
                                    
                                    special_move = self._detect_special_move(
                                        from_pos, to_pos, target_piece, previous_state, current_state
                                    )
                                    
                                    best_move = Move(
                                        from_square=from_pos,
                                        to_square=to_pos,
                                        piece=target_piece,
                                        captured_piece=captured_piece,
                                        special_move=special_move,
                                        promotion_piece=current_state.squares.get(to_pos).type if special_move == SpecialMoveType.PROMOTION else None
                                    )
                                    best_distance = distance
                
                if best_move:
                    self.move_history.append(best_move)
                    return best_move
        
        # If no piece count decreased, check for moves where piece count stayed the same
        # but pieces might have moved (this handles edge cases in test generation)
        for piece_key, prev_count in prev_counts.items():
            curr_count = curr_counts.get(piece_key, 0)
            if prev_count == curr_count and prev_count > 1:  # Multiple pieces of same type
                color, piece_type = piece_key
                target_piece = PieceType(color, piece_type)
                
                # Find all positions with this piece type in both states
                prev_positions = set(pos for pos, piece in previous_state.squares.items() if piece == target_piece)
                curr_positions = set(pos for pos, piece in current_state.squares.items() if piece == target_piece)
                
                # Find positions that changed
                disappeared_positions = prev_positions - curr_positions
                appeared_positions = curr_positions - prev_positions
                
                # If we have equal numbers of disappeared and appeared positions, try to match them
                if len(disappeared_positions) == len(appeared_positions) and len(disappeared_positions) == 1:
                    from_pos = list(disappeared_positions)[0]
                    to_pos = list(appeared_positions)[0]
                    
                    # Check if this is a reasonable move
                    distance = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)
                    if distance <= 7:  # Reasonable move distance
                        captured_piece = previous_state.squares.get(to_pos)
                        confidence = min(previous_state.confidence, current_state.confidence)
                        
                        special_move = self._detect_special_move(
                            from_pos, to_pos, target_piece, previous_state, current_state
                        )
                        
                        move = Move(
                            from_square=from_pos,
                            to_square=to_pos,
                            piece=target_piece,
                            captured_piece=captured_piece,
                            special_move=special_move,
                            promotion_piece=current_state.squares.get(to_pos).type if special_move == SpecialMoveType.PROMOTION else None
                        )
                        self.move_history.append(move)
                        return move
        
        return None
    
    def _handle_piece_type_change(self, pos: Position, prev_piece: PieceType, curr_piece: PieceType,
                                previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """Handle case where piece type changed at a position (likely promotion)."""
        if prev_piece.color == curr_piece.color and prev_piece.type == PieceKind.PAWN:
            # This looks like pawn promotion
            confidence = min(previous_state.confidence, current_state.confidence)
            
            move = Move(
                from_square=pos,
                to_square=pos,
                piece=prev_piece,
                captured_piece=None,
                special_move=SpecialMoveType.PROMOTION,
                promotion_piece=curr_piece.type
            )
            self.move_history.append(move)
            return move
        return None
    
    def _handle_two_position_change(self, changes: List[Tuple[Position, Optional[PieceType], Optional[PieceType]]],
                                  previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """Handle case where exactly two positions changed."""
        pos1, prev1, curr1 = changes[0]
        pos2, prev2, curr2 = changes[1]
        
        # Check if this is a simple move: piece disappeared from one position and appeared at another
        if (prev1 is not None and curr1 is None and 
            prev2 is None and curr2 is not None and 
            prev1 == curr2):
            # Piece moved from pos1 to pos2
            captured_piece = prev2  # This should be None for a simple move
            
            special_move = self._detect_special_move(
                pos1, pos2, prev1, previous_state, current_state
            )
            
            move = Move(
                from_square=pos1,
                to_square=pos2,
                piece=prev1,
                captured_piece=captured_piece,
                special_move=special_move,
                promotion_piece=current_state.squares.get(pos2).type if special_move == SpecialMoveType.PROMOTION else None
            )
            self.move_history.append(move)
            return move
        
        # Check the reverse case (pos2 -> pos1)
        elif (prev2 is not None and curr2 is None and 
              prev1 is None and curr1 is not None and 
              prev2 == curr1):
            # Piece moved from pos2 to pos1
            captured_piece = prev1  # This should be None for a simple move
            
            special_move = self._detect_special_move(
                pos2, pos1, prev2, previous_state, current_state
            )
            
            move = Move(
                from_square=pos2,
                to_square=pos1,
                piece=prev2,
                captured_piece=captured_piece,
                special_move=special_move,
                promotion_piece=current_state.squares.get(pos1).type if special_move == SpecialMoveType.PROMOTION else None
            )
            self.move_history.append(move)
            return move
        
        # Check if this is a promotion move: pawn disappeared and different piece appeared
        elif (prev1 is not None and curr1 is None and 
              prev2 is None and curr2 is not None and 
              prev1.type == PieceKind.PAWN and prev1.color == curr2.color and
              curr2.type in [PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]):
            # Pawn from pos1 promoted to curr2 at pos2
            captured_piece = prev2  # This should be None for a simple promotion
            
            move = Move(
                from_square=pos1,
                to_square=pos2,
                piece=prev1,
                captured_piece=captured_piece,
                special_move=SpecialMoveType.PROMOTION,
                promotion_piece=curr2.type
            )
            self.move_history.append(move)
            return move
        
        # Check the reverse promotion case (pos2 -> pos1)
        elif (prev2 is not None and curr2 is None and 
              prev1 is None and curr1 is not None and 
              prev2.type == PieceKind.PAWN and prev2.color == curr1.color and
              curr1.type in [PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]):
            # Pawn from pos2 promoted to curr1 at pos1
            captured_piece = prev1  # This should be None for a simple promotion
            
            move = Move(
                from_square=pos2,
                to_square=pos1,
                piece=prev2,
                captured_piece=captured_piece,
                special_move=SpecialMoveType.PROMOTION,
                promotion_piece=curr1.type
            )
            self.move_history.append(move)
            return move
        
        # Check if this is a capture: piece disappeared from one position and different piece appeared at another
        elif (prev1 is not None and curr1 is None and 
              prev2 is not None and curr2 is not None and 
              prev1 != prev2 and curr2 != prev2):
            # Piece from pos1 captured piece at pos2
            
            special_move = self._detect_special_move(
                pos1, pos2, prev1, previous_state, current_state
            )
            
            move = Move(
                from_square=pos1,
                to_square=pos2,
                piece=prev1,
                captured_piece=prev2,
                special_move=special_move,
                promotion_piece=current_state.squares.get(pos2).type if special_move == SpecialMoveType.PROMOTION else None
            )
            self.move_history.append(move)
            return move
        
        # Check the reverse capture case
        elif (prev2 is not None and curr2 is None and 
              prev1 is not None and curr1 is not None and 
              prev2 != prev1 and curr1 != prev1):
            # Piece from pos2 captured piece at pos1
            
            special_move = self._detect_special_move(
                pos2, pos1, prev2, previous_state, current_state
            )
            
            move = Move(
                from_square=pos2,
                to_square=pos1,
                piece=prev2,
                captured_piece=prev1,
                special_move=special_move,
                promotion_piece=current_state.squares.get(pos1).type if special_move == SpecialMoveType.PROMOTION else None
            )
            self.move_history.append(move)
            return move
        
        return None
    
    def _handle_multiple_position_changes(self, changes: List[Tuple[Position, Optional[PieceType], Optional[PieceType]]],
                                        previous_state: BoardState, current_state: BoardState) -> Optional[Move]:
        """Handle case where multiple positions changed (castling, en passant, etc.)."""
        # Generate candidates using the existing logic
        disappeared_pieces = {}
        appeared_pieces = {}
        
        for pos, prev_piece, curr_piece in changes:
            if prev_piece is not None and curr_piece != prev_piece:
                disappeared_pieces[pos] = prev_piece
            if curr_piece is not None and prev_piece != curr_piece:
                appeared_pieces[pos] = curr_piece
        
        candidates = self._generate_move_candidates(
            disappeared_pieces, appeared_pieces, previous_state, current_state
        )
        
        if not candidates:
            return None
        
        # Select the best candidate
        best_candidate = self._select_best_candidate(candidates)
        
        if best_candidate and best_candidate.confidence >= self.confidence_threshold:
            move = Move(
                from_square=best_candidate.from_square,
                to_square=best_candidate.to_square,
                piece=best_candidate.piece,
                captured_piece=best_candidate.captured_piece,
                special_move=best_candidate.special_move,
                promotion_piece=self._get_promotion_piece_type(best_candidate, current_state) if best_candidate.special_move == SpecialMoveType.PROMOTION else None
            )
            self.move_history.append(move)
            return move
        
        return None
    
    def detect_capture(self, previous_state: BoardState, current_state: BoardState) -> List[Position]:
        """
        Detect capture events by identifying pieces that disappeared.
        
        Args:
            previous_state: The board state before the capture
            current_state: The board state after the capture
            
        Returns:
            List of positions where captures occurred (where pieces were captured, not where they came from)
            
        Requirements: 3.3
        """
        captures = []
        
        for position in previous_state.squares:
            prev_piece = previous_state.squares.get(position)
            curr_piece = current_state.squares.get(position)
            
            # A capture occurred at this position if:
            # 1. A piece was replaced by a different piece (direct capture)
            # 2. A piece disappeared and the total piece count decreased (piece was captured)
            if prev_piece is not None and curr_piece != prev_piece:
                if curr_piece is not None:
                    # Different piece now occupies the square - this is definitely a capture at this position
                    captures.append(position)
                else:
                    # Square is now empty - check if this was a capture or just a move
                    # Count total pieces to determine if any were captured
                    prev_piece_count = sum(1 for piece in previous_state.squares.values() if piece is not None)
                    curr_piece_count = sum(1 for piece in current_state.squares.values() if piece is not None)
                    
                    # Only count as capture if total piece count decreased AND the piece didn't just move elsewhere
                    if prev_piece_count > curr_piece_count:
                        # Check if this specific piece appears elsewhere (moved) or is gone (captured)
                        piece_found_elsewhere = any(
                            current_state.squares.get(pos) == prev_piece
                            for pos in current_state.squares
                            if pos != position
                        )
                        if not piece_found_elsewhere:
                            # Piece disappeared completely - it was captured
                            captures.append(position)
        
        return captures
    
    def detect_piece_disappearances(self, previous_state: BoardState, current_state: BoardState) -> List[Tuple[Position, PieceType]]:
        """
        Detect pieces that have disappeared from the board.
        
        Args:
            previous_state: The board state before
            current_state: The board state after
            
        Returns:
            List of (position, piece_type) tuples for disappeared pieces
            
        Requirements: 3.3
        """
        disappearances = []
        
        # Count pieces by type to detect true disappearances
        prev_piece_counts = {}
        curr_piece_counts = {}
        
        # Count pieces in previous state
        for piece in previous_state.squares.values():
            if piece is not None:
                key = (piece.color, piece.type)
                prev_piece_counts[key] = prev_piece_counts.get(key, 0) + 1
        
        # Count pieces in current state
        for piece in current_state.squares.values():
            if piece is not None:
                key = (piece.color, piece.type)
                curr_piece_counts[key] = curr_piece_counts.get(key, 0) + 1
        
        # Check each position for disappearances
        for position, piece in previous_state.squares.items():
            if piece is not None:
                current_piece = current_state.squares.get(position)
                
                # Check if piece disappeared from this position
                if current_piece is None or current_piece != piece:
                    piece_key = (piece.color, piece.type)
                    prev_count = prev_piece_counts.get(piece_key, 0)
                    curr_count = curr_piece_counts.get(piece_key, 0)
                    
                    # If the total count of this piece type decreased, some pieces disappeared
                    if prev_count > curr_count:
                        # This position had a piece that's no longer there, and the total count decreased
                        disappearances.append((position, piece))
                    elif prev_count == curr_count:
                        # Same total count, so this was likely a move, not a disappearance
                        # But check if the piece actually appears elsewhere
                        piece_found_elsewhere = any(
                            current_state.squares.get(pos) == piece
                            for pos in current_state.squares
                            if pos != position
                        )
                        
                        if not piece_found_elsewhere:
                            # Piece not found elsewhere but count is same - this shouldn't happen
                            # but if it does, count as disappearance
                            disappearances.append((position, piece))
        
        return disappearances
    
    def _validate_board_states(self, previous_state: BoardState, current_state: BoardState) -> bool:
        """Validate that board states are suitable for move detection."""
        if not previous_state or not current_state:
            return False
        
        # Check confidence levels
        if (previous_state.confidence < self.confidence_threshold or 
            current_state.confidence < self.confidence_threshold):
            return False
        
        # Check that states have the same board structure
        if set(previous_state.squares.keys()) != set(current_state.squares.keys()):
            return False
        
        return True
    
    def _find_disappeared_pieces(self, previous_state: BoardState, current_state: BoardState) -> Dict[Position, PieceType]:
        """Find pieces that were present in previous state but not in current state."""
        disappeared = {}
        
        for position, piece in previous_state.squares.items():
            if piece is not None:
                current_piece = current_state.squares.get(position)
                if current_piece != piece:
                    disappeared[position] = piece
        
        return disappeared
    
    def _find_appeared_pieces(self, previous_state: BoardState, current_state: BoardState) -> Dict[Position, PieceType]:
        """Find pieces that are present in current state but not in previous state."""
        appeared = {}
        
        for position, piece in current_state.squares.items():
            if piece is not None:
                previous_piece = previous_state.squares.get(position)
                if previous_piece != piece:
                    appeared[position] = piece
        
        return appeared
    
    def _generate_move_candidates(self, 
                                disappeared: Dict[Position, PieceType],
                                appeared: Dict[Position, PieceType],
                                previous_state: BoardState,
                                current_state: BoardState) -> List[MoveCandidate]:
        """
        Generate possible move candidates from piece differences.
        
        Enhanced to handle special moves and multiple identical pieces:
        - Regular moves and captures
        - Castling (king and rook movements)
        - En passant (pawn capture with enemy pawn removal)
        - Pawn promotion (piece type changes)
        - Proper handling of multiple identical pieces
        """
        candidates = []
        
        # Create a copy of appeared pieces to track which ones we've matched
        unmatched_appeared = appeared.copy()
        
        # Match disappeared pieces with appeared pieces of the same type
        for from_pos, disappeared_piece in disappeared.items():
            # Find all appeared pieces of the same type
            matching_appeared = [
                (to_pos, appeared_piece) for to_pos, appeared_piece in unmatched_appeared.items()
                if disappeared_piece == appeared_piece
            ]
            
            if matching_appeared:
                # For each potential match, create a candidate
                for to_pos, appeared_piece in matching_appeared:
                    # Check if there was a capture
                    captured_piece = previous_state.squares.get(to_pos)
                    
                    # Calculate confidence based on piece type consistency and board state confidence
                    confidence = min(previous_state.confidence, current_state.confidence)
                    
                    # Detect special moves
                    special_move = self._detect_special_move(
                        from_pos, to_pos, disappeared_piece, previous_state, current_state
                    )
                    
                    candidate = MoveCandidate(
                        from_square=from_pos,
                        to_square=to_pos,
                        piece=disappeared_piece,
                        captured_piece=captured_piece if captured_piece != appeared_piece else None,
                        confidence=confidence,
                        special_move=special_move
                    )
                    candidates.append(candidate)
                
                # Remove the first match to avoid double-matching
                # (This is a simplification - in reality we'd want more sophisticated matching)
                if matching_appeared:
                    first_match_pos = matching_appeared[0][0]
                    unmatched_appeared.pop(first_match_pos, None)
        
        # Handle castling: Generate candidates for both king and rook moves
        castling_candidates = self._generate_castling_candidates(
            disappeared, appeared, previous_state, current_state
        )
        candidates.extend(castling_candidates)
        
        # Handle en passant: Generate candidates for pawn captures with enemy pawn removal
        en_passant_candidates = self._generate_en_passant_candidates(
            disappeared, appeared, previous_state, current_state
        )
        candidates.extend(en_passant_candidates)
        
        # Handle pawn promotion: Generate candidates for piece type changes
        promotion_candidates = self._generate_promotion_candidates(
            disappeared, appeared, previous_state, current_state
        )
        candidates.extend(promotion_candidates)
        
        return candidates
    
    def _generate_castling_candidates(self,
                                    disappeared: Dict[Position, PieceType],
                                    appeared: Dict[Position, PieceType],
                                    previous_state: BoardState,
                                    current_state: BoardState) -> List[MoveCandidate]:
        """Generate castling move candidates by detecting king and rook movements."""
        candidates = []
        
        # Look for king movements that could be castling
        for from_pos, piece in disappeared.items():
            if piece.type == PieceKind.KING:
                for to_pos, appeared_piece in appeared.items():
                    if appeared_piece == piece:
                        # Check if this is a castling move
                        castling_type = self._detect_castling(
                            from_pos, to_pos, piece, previous_state, current_state
                        )
                        if castling_type:
                            confidence = min(previous_state.confidence, current_state.confidence)
                            candidate = MoveCandidate(
                                from_square=from_pos,
                                to_square=to_pos,
                                piece=piece,
                                captured_piece=None,  # Castling never captures
                                confidence=confidence,
                                special_move=castling_type
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _generate_en_passant_candidates(self,
                                      disappeared: Dict[Position, PieceType],
                                      appeared: Dict[Position, PieceType],
                                      previous_state: BoardState,
                                      current_state: BoardState) -> List[MoveCandidate]:
        """Generate en passant move candidates by detecting pawn captures with enemy pawn removal."""
        candidates = []
        
        # Look for pawn movements that could be en passant
        for from_pos, piece in disappeared.items():
            if piece.type == PieceKind.PAWN:
                for to_pos, appeared_piece in appeared.items():
                    if appeared_piece == piece:
                        # Check if this is an en passant move
                        if self._detect_en_passant(
                            from_pos, to_pos, piece, previous_state, current_state
                        ):
                            confidence = min(previous_state.confidence, current_state.confidence)
                            
                            # Determine captured pawn position and piece
                            enemy_color = Color.BLACK if piece.color == Color.WHITE else Color.WHITE
                            if piece.color == Color.WHITE:
                                captured_pawn_pos = Position(to_pos.x, 3)
                            else:
                                captured_pawn_pos = Position(to_pos.x, 4)
                            
                            captured_piece = PieceType(enemy_color, PieceKind.PAWN)
                            
                            candidate = MoveCandidate(
                                from_square=from_pos,
                                to_square=to_pos,
                                piece=piece,
                                captured_piece=captured_piece,
                                confidence=confidence,
                                special_move=SpecialMoveType.EN_PASSANT
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _generate_promotion_candidates(self,
                                     disappeared: Dict[Position, PieceType],
                                     appeared: Dict[Position, PieceType],
                                     previous_state: BoardState,
                                     current_state: BoardState) -> List[MoveCandidate]:
        """Generate pawn promotion candidates by detecting piece type changes."""
        candidates = []
        
        # Look for pawn disappearances that could be promotions
        for from_pos, disappeared_piece in disappeared.items():
            if disappeared_piece.type == PieceKind.PAWN:
                for to_pos, appeared_piece in appeared.items():
                    # Check if this could be a promotion (same color, different piece type, pawn reaches end)
                    if (disappeared_piece.color == appeared_piece.color and 
                        appeared_piece.type != PieceKind.PAWN and
                        self._detect_promotion(from_pos, to_pos, disappeared_piece, previous_state, current_state)):
                        
                        captured_piece = previous_state.squares.get(to_pos)
                        confidence = min(previous_state.confidence, current_state.confidence)
                        
                        candidate = MoveCandidate(
                            from_square=from_pos,
                            to_square=to_pos,
                            piece=disappeared_piece,  # Original pawn
                            captured_piece=captured_piece if captured_piece != appeared_piece else None,
                            confidence=confidence,
                            special_move=SpecialMoveType.PROMOTION
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _detect_special_move(self, 
                           from_pos: Position, 
                           to_pos: Position, 
                           piece: PieceType,
                           previous_state: BoardState,
                           current_state: BoardState) -> Optional[SpecialMoveType]:
        """
        Detect if a move is a special move (castling, en passant, promotion).
        
        Enhanced detection algorithms for special moves:
        - Castling: Detects king and rook movement patterns
        - En passant: Identifies pawn captures with captured pawn removal
        - Promotion: Detects pawn reaching end rank and piece type change
        
        Requirements: 3.4, 3.5, 3.6
        """
        
        # Enhanced castling detection
        if piece.type == PieceKind.KING:
            castling_type = self._detect_castling(from_pos, to_pos, piece, previous_state, current_state)
            if castling_type:
                return castling_type
        
        # Enhanced en passant detection
        if piece.type == PieceKind.PAWN:
            if self._detect_en_passant(from_pos, to_pos, piece, previous_state, current_state):
                return SpecialMoveType.EN_PASSANT
            
            # Enhanced promotion detection
            if self._detect_promotion(from_pos, to_pos, piece, previous_state, current_state):
                return SpecialMoveType.PROMOTION
        
        return None
    
    def _detect_castling(self, 
                        from_pos: Position, 
                        to_pos: Position, 
                        piece: PieceType,
                        previous_state: BoardState,
                        current_state: BoardState) -> Optional[SpecialMoveType]:
        """
        Enhanced castling detection involving king and rook movement patterns.
        
        Detects both kingside and queenside castling by analyzing:
        - King movement (2 squares horizontally)
        - Corresponding rook movement
        - Proper starting positions
        
        Requirements: 3.4
        """
        # King must move exactly 2 squares horizontally on the same rank
        if abs(to_pos.x - from_pos.x) != 2 or to_pos.y != from_pos.y:
            return None
        
        # Determine expected rank based on piece color
        expected_rank = 7 if piece.color == Color.WHITE else 0
        if from_pos.y != expected_rank or to_pos.y != expected_rank:
            return None
        
        # King must start from e-file (x=4)
        if from_pos.x != 4:
            return None
        
        # Determine castling type and expected rook positions
        if to_pos.x == 6:  # Kingside castling (king to g-file)
            castling_type = SpecialMoveType.CASTLING_KINGSIDE
            rook_from = Position(7, expected_rank)  # h-file
            rook_to = Position(5, expected_rank)    # f-file
        elif to_pos.x == 2:  # Queenside castling (king to c-file)
            castling_type = SpecialMoveType.CASTLING_QUEENSIDE
            rook_from = Position(0, expected_rank)  # a-file
            rook_to = Position(3, expected_rank)    # d-file
        else:
            return None
        
        # Verify rook movement occurred
        expected_rook = PieceType(piece.color, PieceKind.ROOK)
        rook_was_there = previous_state.squares.get(rook_from) == expected_rook
        rook_moved_correctly = current_state.squares.get(rook_to) == expected_rook
        rook_left_origin = current_state.squares.get(rook_from) is None
        
        if rook_was_there and rook_moved_correctly and rook_left_origin:
            return castling_type
        
        return None
    
    def _detect_en_passant(self, 
                          from_pos: Position, 
                          to_pos: Position, 
                          piece: PieceType,
                          previous_state: BoardState,
                          current_state: BoardState) -> bool:
        """
        Enhanced en passant detection identifying pawn captures with captured pawn removal.
        
        Detects en passant by analyzing:
        - Diagonal pawn movement to empty square
        - Captured pawn removal from adjacent square
        - Proper rank positioning
        
        Requirements: 3.5
        """
        # Must be diagonal move (1 square each direction)
        if abs(to_pos.x - from_pos.x) != 1 or abs(to_pos.y - from_pos.y) != 1:
            return False
        
        # Destination square must have been empty in previous state
        if previous_state.squares.get(to_pos) is not None:
            return False
        
        # Determine expected ranks for en passant based on pawn color
        if piece.color == Color.WHITE:
            # White pawn moving from rank 5 to rank 6 (4 to 3 in 0-indexed)
            if from_pos.y != 3 or to_pos.y != 2:
                return False
            captured_pawn_pos = Position(to_pos.x, 3)  # Same file as destination, rank 5
        else:
            # Black pawn moving from rank 4 to rank 3 (3 to 4 in 0-indexed)
            if from_pos.y != 4 or to_pos.y != 5:
                return False
            captured_pawn_pos = Position(to_pos.x, 4)  # Same file as destination, rank 4
        
        # Check if enemy pawn was captured (disappeared from adjacent square)
        enemy_color = Color.BLACK if piece.color == Color.WHITE else Color.WHITE
        expected_captured_pawn = PieceType(enemy_color, PieceKind.PAWN)
        
        pawn_was_there = previous_state.squares.get(captured_pawn_pos) == expected_captured_pawn
        pawn_disappeared = current_state.squares.get(captured_pawn_pos) is None
        
        return pawn_was_there and pawn_disappeared
    
    def _detect_promotion(self, 
                         from_pos: Position, 
                         to_pos: Position, 
                         piece: PieceType,
                         previous_state: BoardState,
                         current_state: BoardState) -> bool:
        """
        Enhanced pawn promotion detection with piece type identification.
        
        Detects promotion by analyzing:
        - Pawn reaching end rank
        - Piece type change at destination
        - Valid promotion piece types
        
        Requirements: 3.6
        """
        # Check if pawn reached the promotion rank
        if piece.color == Color.WHITE:
            # White pawn promotes on rank 8 (y=0 in 0-indexed)
            if to_pos.y != 0:
                return False
        else:
            # Black pawn promotes on rank 1 (y=7 in 0-indexed)
            if to_pos.y != 7:
                return False
        
        # Check if piece at destination is different from pawn
        dest_piece = current_state.squares.get(to_pos)
        if dest_piece is None or dest_piece.type == PieceKind.PAWN:
            return False
        
        # Verify the promoted piece is the same color as the original pawn
        if dest_piece.color != piece.color:
            return False
        
        # Verify it's a valid promotion piece (Queen, Rook, Bishop, or Knight)
        valid_promotion_pieces = {PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT}
        if dest_piece.type not in valid_promotion_pieces:
            return False
        
        return True
    
    def _select_best_candidate(self, candidates: List[MoveCandidate]) -> Optional[MoveCandidate]:
        """
        Select the best move candidate from a list of possibilities.
        
        Enhanced selection logic for special moves:
        - Prioritizes special moves over regular moves
        - For castling, prefers king moves over rook moves
        - Uses confidence scores as tiebreaker
        """
        if not candidates:
            return None
        
        # Separate candidates by type
        special_candidates = [c for c in candidates if c.special_move is not None]
        regular_candidates = [c for c in candidates if c.special_move is None]
        
        # Prioritize special moves
        if special_candidates:
            # For castling, prioritize king moves over rook moves
            castling_candidates = [c for c in special_candidates 
                                 if c.special_move in [SpecialMoveType.CASTLING_KINGSIDE, SpecialMoveType.CASTLING_QUEENSIDE]]
            if castling_candidates:
                # Among castling candidates, prefer king moves
                king_candidates = [c for c in castling_candidates if c.piece.type == PieceKind.KING]
                if king_candidates:
                    return max(king_candidates, key=lambda c: c.confidence)
                else:
                    return max(castling_candidates, key=lambda c: c.confidence)
            
            # For other special moves, select by confidence
            return max(special_candidates, key=lambda c: c.confidence)
        
        # If no special moves, select best regular move
        if regular_candidates:
            # For regular moves, prefer moves that make logical sense
            # Sort by confidence first, then by move distance (shorter moves are more likely)
            def candidate_score(candidate):
                distance = abs(candidate.to_square.x - candidate.from_square.x) + abs(candidate.to_square.y - candidate.from_square.y)
                # Higher confidence is better, shorter distance is better
                return candidate.confidence - (distance * 0.01)  # Small penalty for longer moves
            
            return max(regular_candidates, key=candidate_score)
        
        return None
    
    def get_move_history(self) -> List[Move]:
        """Get the history of detected moves."""
        return self.move_history.copy()
    
    def clear_history(self) -> None:
        """Clear the move history."""
        self.move_history.clear()
    
    def _get_promotion_piece_type(self, candidate: MoveCandidate, current_state: BoardState) -> Optional[PieceKind]:
        """
        Get the piece type that a pawn was promoted to.
        
        Args:
            candidate: The move candidate representing the promotion
            current_state: The board state after the promotion
            
        Returns:
            The piece type the pawn was promoted to, or None if not a promotion
        """
        if candidate.special_move != SpecialMoveType.PROMOTION:
            return None
        
        promoted_piece = current_state.squares.get(candidate.to_square)
        if promoted_piece is None:
            return None
        
        return promoted_piece.type
    def _is_basic_legal_move(self, from_pos: Position, to_pos: Position, piece: PieceType, board_state: BoardState) -> bool:
        """
        Basic move legality check - less strict than full chess rules.
        Just checks basic sanity, not full chess rule compliance.
        """
        # Calculate movement deltas
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        # No movement is not a valid move
        if dx == 0 and dy == 0:
            return False
        
        # Can't move more than 7 squares in any direction
        if abs(dx) > 7 or abs(dy) > 7:
            return False
        
        # Can't capture own piece
        target_piece = board_state.squares.get(to_pos)
        if target_piece is not None and target_piece.color == piece.color:
            return False
        
        # Basic piece movement patterns (relaxed)
        if piece.type == PieceKind.PAWN:
            # Pawns can move forward or diagonally (relaxed - don't check direction strictly)
            return abs(dx) <= 1 and abs(dy) <= 2
        elif piece.type == PieceKind.ROOK:
            # Rooks move in straight lines
            return dx == 0 or dy == 0
        elif piece.type == PieceKind.BISHOP:
            # Bishops move diagonally
            return abs(dx) == abs(dy)
        elif piece.type == PieceKind.QUEEN:
            # Queens move like rooks or bishops
            return (dx == 0 or dy == 0) or (abs(dx) == abs(dy))
        elif piece.type == PieceKind.KING:
            # Kings move one square in any direction
            return abs(dx) <= 1 and abs(dy) <= 1
        elif piece.type == PieceKind.KNIGHT:
            # Knights move in L-shape
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        return True  # Default to allowing the move
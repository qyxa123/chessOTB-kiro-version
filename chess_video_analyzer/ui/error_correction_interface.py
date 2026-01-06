"""
Error highlighting and correction interface for the Chess Video Analyzer.

This module provides functionality for:
- Highlighting problematic moves in the UI
- Manual correction capabilities for detected errors
- User interaction for reviewing and fixing flagged moves

Requirements: 8.4, 8.5
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional, Callable, Dict, Any
import logging

from ..core.data_models import Move, Position, PieceType, Color, PieceKind, GameState


class MoveEditDialog:
    """Dialog for editing individual moves."""
    
    def __init__(self, parent, move: Move, move_index: int):
        """
        Initialize the move edit dialog.
        
        Args:
            parent: Parent window
            move: The move to edit
            move_index: Index of the move in the game
        """
        self.parent = parent
        self.original_move = move
        self.move_index = move_index
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Edit Move {move_index + 1}")
        self.dialog.geometry("400x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (500 // 2)
        self.dialog.geometry(f"400x500+{x}+{y}")
        
        self._setup_ui()
        self._populate_fields()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text=f"Edit Move {self.move_index + 1}", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Original move info
        original_frame = ttk.LabelFrame(main_frame, text="Original Move", padding="5")
        original_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.original_info = tk.Text(original_frame, height=3, width=50, state=tk.DISABLED)
        self.original_info.pack(fill=tk.X)
        
        # Edit fields
        edit_frame = ttk.LabelFrame(main_frame, text="Edit Move", padding="5")
        edit_frame.pack(fill=tk.X, pady=(0, 10))
        
        # From square
        ttk.Label(edit_frame, text="From Square:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.from_square_var = tk.StringVar()
        self.from_square_entry = ttk.Entry(edit_frame, textvariable=self.from_square_var, width=10)
        self.from_square_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # To square
        ttk.Label(edit_frame, text="To Square:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.to_square_var = tk.StringVar()
        self.to_square_entry = ttk.Entry(edit_frame, textvariable=self.to_square_var, width=10)
        self.to_square_entry.grid(row=0, column=3, sticky=tk.W)
        
        # Piece type
        ttk.Label(edit_frame, text="Piece:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.piece_type_var = tk.StringVar()
        piece_combo = ttk.Combobox(edit_frame, textvariable=self.piece_type_var, 
                                  values=["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"],
                                  state="readonly", width=10)
        piece_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        
        # Piece color
        ttk.Label(edit_frame, text="Color:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.piece_color_var = tk.StringVar()
        color_combo = ttk.Combobox(edit_frame, textvariable=self.piece_color_var,
                                  values=["White", "Black"], state="readonly", width=10)
        color_combo.grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
        
        # Captured piece (optional)
        ttk.Label(edit_frame, text="Captured Piece:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.captured_piece_var = tk.StringVar()
        captured_combo = ttk.Combobox(edit_frame, textvariable=self.captured_piece_var,
                                     values=["None", "Pawn", "Rook", "Knight", "Bishop", "Queen"],
                                     state="readonly", width=10)
        captured_combo.grid(row=2, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        
        # Special move
        ttk.Label(edit_frame, text="Special Move:").grid(row=2, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.special_move_var = tk.StringVar()
        special_combo = ttk.Combobox(edit_frame, textvariable=self.special_move_var,
                                    values=["None", "Castling Kingside", "Castling Queenside", 
                                           "En Passant", "Promotion"],
                                    state="readonly", width=15)
        special_combo.grid(row=2, column=3, sticky=tk.W, pady=(5, 0))
        
        # Flag status
        flag_frame = ttk.LabelFrame(main_frame, text="Flag Status", padding="5")
        flag_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.is_flagged_var = tk.BooleanVar()
        flagged_check = ttk.Checkbutton(flag_frame, text="Mark as flagged", 
                                       variable=self.is_flagged_var)
        flagged_check.pack(anchor=tk.W)
        
        ttk.Label(flag_frame, text="Flag Reason:").pack(anchor=tk.W, pady=(5, 0))
        self.flag_reason_var = tk.StringVar()
        self.flag_reason_entry = ttk.Entry(flag_frame, textvariable=self.flag_reason_var, width=50)
        self.flag_reason_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Save", command=self._save_move).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Delete Move", command=self._delete_move).pack(side=tk.LEFT)
    
    def _populate_fields(self):
        """Populate the dialog fields with current move data."""
        move = self.original_move
        
        # Show original move info
        self.original_info.config(state=tk.NORMAL)
        self.original_info.delete(1.0, tk.END)
        
        from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
        to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
        piece_name = f"{move.piece.color.value.title()} {move.piece.type.value.title()}"
        
        info_text = f"Move: {piece_name} from {from_square} to {to_square}"
        if move.captured_piece:
            info_text += f"\nCaptures: {move.captured_piece.type.value.title()}"
        if move.special_move:
            info_text += f"\nSpecial: {move.special_move.value}"
        
        self.original_info.insert(1.0, info_text)
        self.original_info.config(state=tk.DISABLED)
        
        # Populate edit fields
        self.from_square_var.set(from_square)
        self.to_square_var.set(to_square)
        self.piece_type_var.set(move.piece.type.value.title())
        self.piece_color_var.set(move.piece.color.value.title())
        
        if move.captured_piece:
            self.captured_piece_var.set(move.captured_piece.type.value.title())
        else:
            self.captured_piece_var.set("None")
        
        if move.special_move:
            special_map = {
                "O-O": "Castling Kingside",
                "O-O-O": "Castling Queenside",
                "en_passant": "En Passant",
                "promotion": "Promotion"
            }
            self.special_move_var.set(special_map.get(move.special_move.value, "None"))
        else:
            self.special_move_var.set("None")
        
        self.is_flagged_var.set(move.is_flagged)
        self.flag_reason_var.set(move.flag_reason or "")
    
    def _parse_square(self, square_str: str) -> Optional[Position]:
        """Parse algebraic notation to Position."""
        if len(square_str) != 2:
            return None
        
        file_char = square_str[0].lower()
        rank_char = square_str[1]
        
        if file_char not in 'abcdefgh' or rank_char not in '12345678':
            return None
        
        x = ord(file_char) - ord('a')
        y = 8 - int(rank_char)
        
        try:
            return Position(x, y)
        except ValueError:
            return None
    
    def _save_move(self):
        """Save the edited move."""
        try:
            # Parse squares
            from_square = self._parse_square(self.from_square_var.get())
            to_square = self._parse_square(self.to_square_var.get())
            
            if not from_square or not to_square:
                messagebox.showerror("Invalid Input", "Please enter valid square coordinates (e.g., e2, e4)")
                return
            
            # Parse piece
            piece_type_str = self.piece_type_var.get().lower()
            piece_color_str = self.piece_color_var.get().lower()
            
            if not piece_type_str or not piece_color_str:
                messagebox.showerror("Invalid Input", "Please select piece type and color")
                return
            
            piece_type = PieceKind(piece_type_str)
            piece_color = Color(piece_color_str)
            piece = PieceType(piece_color, piece_type)
            
            # Parse captured piece
            captured_piece = None
            captured_str = self.captured_piece_var.get()
            if captured_str and captured_str != "None":
                captured_type = PieceKind(captured_str.lower())
                # Assume captured piece is opposite color
                captured_color = Color.BLACK if piece_color == Color.WHITE else Color.WHITE
                captured_piece = PieceType(captured_color, captured_type)
            
            # Parse special move
            special_move = None
            special_str = self.special_move_var.get()
            if special_str and special_str != "None":
                special_map = {
                    "Castling Kingside": "O-O",
                    "Castling Queenside": "O-O-O",
                    "En Passant": "en_passant",
                    "Promotion": "promotion"
                }
                from ..core.data_models import SpecialMoveType
                special_move = SpecialMoveType(special_map[special_str])
            
            # Create edited move
            edited_move = Move(
                from_square=from_square,
                to_square=to_square,
                piece=piece,
                captured_piece=captured_piece,
                special_move=special_move,
                is_flagged=self.is_flagged_var.get(),
                flag_reason=self.flag_reason_var.get() if self.is_flagged_var.get() else None
            )
            
            self.result = ('save', edited_move)
            self.dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error parsing move data: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
    
    def _delete_move(self):
        """Delete the move."""
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete move {self.move_index + 1}?"):
            self.result = ('delete', None)
            self.dialog.destroy()
    
    def _cancel(self):
        """Cancel editing."""
        self.result = ('cancel', None)
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result


class ErrorCorrectionInterface:
    """
    Interface for highlighting and correcting errors in chess game analysis.
    
    Provides functionality for:
    - Highlighting problematic moves
    - Manual correction of detected errors
    - User review and approval of flagged moves
    
    Requirements: 8.4, 8.5
    """
    
    def __init__(self, parent_widget, on_move_corrected: Optional[Callable] = None):
        """
        Initialize the error correction interface.
        
        Args:
            parent_widget: Parent widget to contain the interface
            on_move_corrected: Callback function called when a move is corrected
        """
        self.parent = parent_widget
        self.on_move_corrected = on_move_corrected
        self.game_state: Optional[GameState] = None
        self.flagged_moves: List[Move] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the error correction interface."""
        # Main frame
        self.main_frame = ttk.LabelFrame(self.parent, text="Error Correction", padding="5")
        self.main_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Error summary
        summary_frame = ttk.Frame(self.main_frame)
        summary_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.error_count_var = tk.StringVar(value="No errors detected")
        self.error_count_label = ttk.Label(summary_frame, textvariable=self.error_count_var,
                                          foreground="green")
        self.error_count_label.pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = ttk.Frame(summary_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.review_all_button = ttk.Button(button_frame, text="Review All Errors",
                                           command=self._review_all_errors, state="disabled")
        self.review_all_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_flags_button = ttk.Button(button_frame, text="Clear All Flags",
                                            command=self._clear_all_flags, state="disabled")
        self.clear_flags_button.pack(side=tk.LEFT)
        
        # Error list
        list_frame = ttk.Frame(self.main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for error list
        columns = ("Move", "Type", "Reason", "Status")
        self.error_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)
        
        # Configure columns
        self.error_tree.heading("Move", text="Move #")
        self.error_tree.heading("Type", text="Move")
        self.error_tree.heading("Reason", text="Issue")
        self.error_tree.heading("Status", text="Status")
        
        self.error_tree.column("Move", width=60, anchor=tk.CENTER)
        self.error_tree.column("Type", width=150)
        self.error_tree.column("Reason", width=200)
        self.error_tree.column("Status", width=80, anchor=tk.CENTER)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.error_tree.yview)
        self.error_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.error_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to edit
        self.error_tree.bind("<Double-1>", self._on_error_double_click)
        
        # Context menu
        self.context_menu = tk.Menu(self.error_tree, tearoff=0)
        self.context_menu.add_command(label="Edit Move", command=self._edit_selected_move)
        self.context_menu.add_command(label="Mark as Resolved", command=self._mark_resolved)
        self.context_menu.add_command(label="Delete Move", command=self._delete_selected_move)
        
        self.error_tree.bind("<Button-2>", self._show_context_menu)  # Right-click on Mac
        self.error_tree.bind("<Button-3>", self._show_context_menu)  # Right-click on Windows/Linux
    
    def update_game_state(self, game_state: GameState):
        """
        Update the interface with a new game state.
        
        Args:
            game_state: The current game state
        """
        self.game_state = game_state
        self.flagged_moves = [move for move in game_state.move_history if move.is_flagged]
        self._refresh_error_display()
    
    def _refresh_error_display(self):
        """Refresh the error display with current flagged moves."""
        # Clear existing items
        for item in self.error_tree.get_children():
            self.error_tree.delete(item)
        
        if not self.flagged_moves:
            self.error_count_var.set("No errors detected")
            self.error_count_label.config(foreground="green")
            self.review_all_button.config(state="disabled")
            self.clear_flags_button.config(state="disabled")
            return
        
        # Update error count
        error_count = len(self.flagged_moves)
        self.error_count_var.set(f"{error_count} error{'s' if error_count != 1 else ''} detected")
        self.error_count_label.config(foreground="red")
        self.review_all_button.config(state="normal")
        self.clear_flags_button.config(state="normal")
        
        # Populate error list
        for i, move in enumerate(self.game_state.move_history):
            if move.is_flagged:
                move_num = (i // 2) + 1
                color = "White" if i % 2 == 0 else "Black"
                
                # Format move description
                from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
                to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
                move_desc = f"{move_num}.{color[0]} {move.piece.type.value.title()} {from_square}-{to_square}"
                
                # Add to tree
                item_id = self.error_tree.insert("", tk.END, values=(
                    f"{move_num}.{color[0]}",
                    move_desc,
                    move.flag_reason or "Unknown issue",
                    "Flagged"
                ))
                
                # Store move index in item for later reference
                self.error_tree.set(item_id, "move_index", i)
    
    def _on_error_double_click(self, event):
        """Handle double-click on error item."""
        self._edit_selected_move()
    
    def _show_context_menu(self, event):
        """Show context menu for error item."""
        item = self.error_tree.identify_row(event.y)
        if item:
            self.error_tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def _edit_selected_move(self):
        """Edit the selected move."""
        selection = self.error_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a move to edit.")
            return
        
        item = selection[0]
        move_index = int(self.error_tree.set(item, "move_index"))
        move = self.game_state.move_history[move_index]
        
        # Show edit dialog
        dialog = MoveEditDialog(self.parent, move, move_index)
        result = dialog.show()
        
        if result and result[0] == 'save':
            # Update the move
            self.game_state.move_history[move_index] = result[1]
            self._refresh_error_display()
            
            if self.on_move_corrected:
                self.on_move_corrected(move_index, result[1])
            
            self.logger.info(f"Move {move_index + 1} corrected by user")
            
        elif result and result[0] == 'delete':
            # Delete the move
            del self.game_state.move_history[move_index]
            self._refresh_error_display()
            
            if self.on_move_corrected:
                self.on_move_corrected(move_index, None)  # None indicates deletion
            
            self.logger.info(f"Move {move_index + 1} deleted by user")
    
    def _mark_resolved(self):
        """Mark the selected move as resolved (unflag it)."""
        selection = self.error_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a move to mark as resolved.")
            return
        
        item = selection[0]
        move_index = int(self.error_tree.set(item, "move_index"))
        move = self.game_state.move_history[move_index]
        
        # Unflag the move
        move.is_flagged = False
        move.flag_reason = None
        
        self._refresh_error_display()
        
        if self.on_move_corrected:
            self.on_move_corrected(move_index, move)
        
        self.logger.info(f"Move {move_index + 1} marked as resolved")
    
    def _delete_selected_move(self):
        """Delete the selected move."""
        selection = self.error_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a move to delete.")
            return
        
        item = selection[0]
        move_index = int(self.error_tree.set(item, "move_index"))
        
        move_num = (move_index // 2) + 1
        color = "White" if move_index % 2 == 0 else "Black"
        
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete move {move_num} ({color})?"):
            del self.game_state.move_history[move_index]
            self._refresh_error_display()
            
            if self.on_move_corrected:
                self.on_move_corrected(move_index, None)
            
            self.logger.info(f"Move {move_index + 1} deleted by user")
    
    def _review_all_errors(self):
        """Review all flagged moves one by one."""
        if not self.flagged_moves:
            return
        
        flagged_indices = [i for i, move in enumerate(self.game_state.move_history) if move.is_flagged]
        
        for move_index in flagged_indices:
            if move_index >= len(self.game_state.move_history):
                continue  # Move may have been deleted
            
            move = self.game_state.move_history[move_index]
            if not move.is_flagged:
                continue  # Move may have been unflagged
            
            # Show edit dialog for each flagged move
            dialog = MoveEditDialog(self.parent, move, move_index)
            result = dialog.show()
            
            if result and result[0] == 'save':
                self.game_state.move_history[move_index] = result[1]
                if self.on_move_corrected:
                    self.on_move_corrected(move_index, result[1])
            elif result and result[0] == 'delete':
                del self.game_state.move_history[move_index]
                if self.on_move_corrected:
                    self.on_move_corrected(move_index, None)
                # Adjust indices for remaining moves
                flagged_indices = [idx - 1 if idx > move_index else idx for idx in flagged_indices]
            elif result and result[0] == 'cancel':
                break  # User cancelled review
        
        self._refresh_error_display()
        messagebox.showinfo("Review Complete", "Finished reviewing all flagged moves.")
    
    def _clear_all_flags(self):
        """Clear all flags from moves."""
        if not self.flagged_moves:
            return
        
        if messagebox.askyesno("Confirm Clear Flags", 
                              "Are you sure you want to clear all error flags? "
                              "This will mark all moves as correct."):
            for move in self.game_state.move_history:
                if move.is_flagged:
                    move.is_flagged = False
                    move.flag_reason = None
            
            self._refresh_error_display()
            
            if self.on_move_corrected:
                self.on_move_corrected(-1, None)  # -1 indicates bulk operation
            
            self.logger.info("All error flags cleared by user")
    
    def add_manual_flag(self, move_index: int, reason: str):
        """
        Manually flag a move with a reason.
        
        Args:
            move_index: Index of the move to flag
            reason: Reason for flagging the move
        """
        if 0 <= move_index < len(self.game_state.move_history):
            move = self.game_state.move_history[move_index]
            move.is_flagged = True
            move.flag_reason = reason
            self._refresh_error_display()
            
            self.logger.info(f"Move {move_index + 1} manually flagged: {reason}")
    
    def get_correction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of corrections made.
        
        Returns:
            Dictionary with correction statistics
        """
        if not self.game_state:
            return {"total_moves": 0, "flagged_moves": 0, "correction_rate": 0.0}
        
        total_moves = len(self.game_state.move_history)
        flagged_moves = len([move for move in self.game_state.move_history if move.is_flagged])
        
        return {
            "total_moves": total_moves,
            "flagged_moves": flagged_moves,
            "correction_rate": (total_moves - flagged_moves) / total_moves if total_moves > 0 else 1.0,
            "accuracy_percentage": ((total_moves - flagged_moves) / total_moves * 100) if total_moves > 0 else 100.0
        }
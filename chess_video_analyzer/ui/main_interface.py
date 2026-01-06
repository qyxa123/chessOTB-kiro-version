"""
Main application interface for the Chess Video Analyzer.

This module provides the primary user interface components including:
- Video file import interface (drag-drop and file dialog)
- Progress indicators for processing operations
- Results display for generated moves

Requirements: 8.2, 8.3, 8.4
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import logging
from datetime import datetime

from ..video.processor import VideoProcessor, VideoProcessingError
from ..core.data_models import GameState, Move, VideoMetadata, GameMetadata
from ..notation.pgn_generator import PGNGenerator
from ..notation.fen_generator import FENGenerator
from .error_correction_interface import ErrorCorrectionInterface


class ProgressCallback:
    """Callback interface for progress updates."""
    
    def __init__(self, callback_func: Callable[[str, float], None]):
        """
        Initialize progress callback.
        
        Args:
            callback_func: Function that takes (status_message, progress_percentage)
        """
        self.callback_func = callback_func
    
    def update(self, message: str, progress: float):
        """Update progress with message and percentage (0.0 to 1.0)."""
        self.callback_func(message, progress)


class ProcessingResults:
    """Container for processing results."""
    
    def __init__(self):
        self.game_state: Optional[GameState] = None
        self.video_metadata: Optional[VideoMetadata] = None
        self.game_metadata: Optional[GameMetadata] = None
        self.pgn_content: Optional[str] = None
        self.fen_sequence: Optional[List[str]] = None
        self.processing_time: Optional[float] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []


class MainInterface:
    """
    Main application interface for Chess Video Analyzer.
    
    Provides video import, processing controls, progress tracking,
    and results display functionality.
    
    Requirements: 8.2, 8.3, 8.4
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the main interface.
        
        Args:
            root: The root Tkinter window
        """
        self.root = root
        self.root.title("Chess Video Analyzer")
        self.root.geometry("800x600")
        
        # Initialize components
        self.video_processor = VideoProcessor()
        self.pgn_generator = PGNGenerator()
        self.fen_generator = FENGenerator()
        
        # State variables
        self.current_video_path: Optional[str] = None
        self.processing_results = ProcessingResults()
        self.is_processing = False
        
        # Setup UI components
        self._setup_ui()
        self._setup_drag_drop()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(5, weight=1)  # For error correction section
        
        # Title
        title_label = ttk.Label(main_frame, text="Chess Video Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Video import section
        self._setup_import_section(main_frame)
        
        # Processing controls section
        self._setup_processing_section(main_frame)
        
        # Progress section
        self._setup_progress_section(main_frame)
        
        # Results section
        self._setup_results_section(main_frame)
        
        # Error correction section
        self._setup_error_correction_section(main_frame)
    
    def _setup_import_section(self, parent):
        """Set up the video import section."""
        # Import frame
        import_frame = ttk.LabelFrame(parent, text="Video Import", padding="10")
        import_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        import_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(import_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(import_frame, textvariable=self.file_path_var, 
                                        state="readonly")
        self.file_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_button = ttk.Button(import_frame, text="Browse...", 
                                       command=self._browse_file)
        self.browse_button.grid(row=0, column=2)
        
        # Drag and drop area
        self.drop_frame = tk.Frame(import_frame, bg="lightgray", height=80, 
                                  relief="sunken", bd=2)
        self.drop_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                            pady=(10, 0))
        self.drop_frame.grid_propagate(False)
        
        drop_label = tk.Label(self.drop_frame, text="Drag and drop video file here\n(MP4, AVI, MOV)", 
                             bg="lightgray", fg="gray")
        drop_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Video info display
        self.video_info_var = tk.StringVar()
        self.video_info_label = ttk.Label(import_frame, textvariable=self.video_info_var, 
                                         foreground="blue")
        self.video_info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
    
    def _setup_processing_section(self, parent):
        """Set up the processing controls section."""
        # Processing frame
        processing_frame = ttk.LabelFrame(parent, text="Processing", padding="10")
        processing_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Processing button
        self.process_button = ttk.Button(processing_frame, text="Analyze Video", 
                                        command=self._start_processing, state="disabled")
        self.process_button.grid(row=0, column=0, padx=(0, 10))
        
        # Cancel button
        self.cancel_button = ttk.Button(processing_frame, text="Cancel", 
                                       command=self._cancel_processing, state="disabled")
        self.cancel_button.grid(row=0, column=1)
    
    def _setup_progress_section(self, parent):
        """Set up the progress indicators section."""
        # Progress frame
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Estimated time label
        self.time_var = tk.StringVar()
        self.time_label = ttk.Label(progress_frame, textvariable=self.time_var)
        self.time_label.grid(row=2, column=0, sticky=tk.W)
    
    def _setup_results_section(self, parent):
        """Set up the results display section."""
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results notebook for tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Moves tab
        self.moves_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.moves_frame, text="Moves")
        
        self.moves_text = ScrolledText(self.moves_frame, height=10, width=60)
        self.moves_text.pack(fill=tk.BOTH, expand=True)
        
        # PGN tab
        self.pgn_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.pgn_frame, text="PGN")
        
        self.pgn_text = ScrolledText(self.pgn_frame, height=10, width=60)
        self.pgn_text.pack(fill=tk.BOTH, expand=True)
        
        # FEN tab
        self.fen_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.fen_frame, text="FEN")
        
        self.fen_text = ScrolledText(self.fen_frame, height=10, width=60)
        self.fen_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.grid(row=1, column=0, sticky=tk.W)
        
        self.export_pgn_button = ttk.Button(export_frame, text="Export PGN", 
                                           command=self._export_pgn, state="disabled")
        self.export_pgn_button.grid(row=0, column=0, padx=(0, 10))
        
        self.export_fen_button = ttk.Button(export_frame, text="Export FEN", 
                                           command=self._export_fen, state="disabled")
        self.export_fen_button.grid(row=0, column=1)
    
    def _setup_error_correction_section(self, parent):
        """Set up the error correction section."""
        # Error correction will be added to a separate frame
        error_frame = ttk.Frame(parent)
        error_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        error_frame.columnconfigure(0, weight=1)
        error_frame.rowconfigure(0, weight=1)
        
        # Initialize error correction interface
        self.error_correction = ErrorCorrectionInterface(
            error_frame, 
            on_move_corrected=self._on_move_corrected
        )
    
    def _setup_drag_drop(self):
        """Set up drag and drop functionality."""
        # Note: Full drag-drop implementation would require additional libraries
        # like tkinterdnd2. For now, we'll set up the visual feedback.
        
        def on_drop_enter(event):
            self.drop_frame.config(bg="lightblue")
        
        def on_drop_leave(event):
            self.drop_frame.config(bg="lightgray")
        
        def on_drop(event):
            self.drop_frame.config(bg="lightgray")
            # In a full implementation, this would handle the dropped file
            # For now, we'll show a message
            messagebox.showinfo("Drag & Drop", 
                              "Drag & Drop functionality requires additional setup.\n"
                              "Please use the Browse button to select files.")
        
        # Bind events (basic visual feedback only)
        self.drop_frame.bind("<Enter>", on_drop_enter)
        self.drop_frame.bind("<Leave>", on_drop_leave)
        self.drop_frame.bind("<Button-1>", lambda e: self._browse_file())
    
    def _browse_file(self):
        """Open file dialog to browse for video files."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Chess Game Video",
            filetypes=filetypes
        )
        
        if filename:
            self._load_video_file(filename)
    
    def _load_video_file(self, file_path: str):
        """
        Load a video file and display its information.
        
        Args:
            file_path: Path to the video file
        """
        try:
            # Load video and get metadata
            metadata = self.video_processor.load_video(file_path)
            
            # Update UI
            self.current_video_path = file_path
            self.file_path_var.set(file_path)
            
            # Display video information
            duration_str = f"{metadata.duration:.1f}s"
            resolution_str = f"{metadata.resolution[0]}x{metadata.resolution[1]}"
            info_text = (f"Duration: {duration_str}, "
                        f"FPS: {metadata.fps:.1f}, "
                        f"Resolution: {resolution_str}, "
                        f"Format: {metadata.format}")
            self.video_info_var.set(info_text)
            
            # Enable processing button
            self.process_button.config(state="normal")
            
            # Clear previous results
            self._clear_results()
            
            self.logger.info(f"Loaded video: {Path(file_path).name}")
            
        except Exception as e:
            self.logger.error(f"Error loading video: {str(e)}")
            messagebox.showerror("Error", f"Failed to load video file:\n{str(e)}")
            self._clear_video_info()
    
    def _clear_video_info(self):
        """Clear video information and disable processing."""
        self.current_video_path = None
        self.file_path_var.set("")
        self.video_info_var.set("")
        self.process_button.config(state="disabled")
        self.video_processor.close()
    
    def _clear_results(self):
        """Clear all results displays."""
        self.moves_text.delete(1.0, tk.END)
        self.pgn_text.delete(1.0, tk.END)
        self.fen_text.delete(1.0, tk.END)
        self.export_pgn_button.config(state="disabled")
        self.export_fen_button.config(state="disabled")
        self.processing_results = ProcessingResults()
    
    def _start_processing(self):
        """Start video processing in a separate thread."""
        if not self.current_video_path or self.is_processing:
            return
        
        self.is_processing = True
        self.process_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.browse_button.config(state="disabled")
        
        # Create progress callback
        progress_callback = ProgressCallback(self._update_progress)
        
        # Start processing in separate thread
        processing_thread = threading.Thread(
            target=self._process_video_thread,
            args=(self.current_video_path, progress_callback),
            daemon=True
        )
        processing_thread.start()
    
    def _process_video_thread(self, video_path: str, progress_callback: ProgressCallback):
        """
        Process video in a separate thread.
        
        Args:
            video_path: Path to the video file
            progress_callback: Callback for progress updates
        """
        try:
            start_time = datetime.now()
            
            # This is a placeholder for the actual processing pipeline
            # In a complete implementation, this would orchestrate all components
            
            progress_callback.update("Initializing processing...", 0.0)
            
            # Simulate processing steps
            import time
            
            progress_callback.update("Extracting frames...", 0.1)
            time.sleep(0.5)  # Simulate work
            
            progress_callback.update("Detecting chess board...", 0.2)
            time.sleep(0.5)
            
            progress_callback.update("Recognizing pieces...", 0.4)
            time.sleep(0.5)
            
            progress_callback.update("Tracking moves...", 0.6)
            time.sleep(0.5)
            
            progress_callback.update("Validating game state...", 0.8)
            time.sleep(0.5)
            
            progress_callback.update("Generating notation...", 0.9)
            time.sleep(0.5)
            
            # Create mock results for demonstration
            self._create_mock_results()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_results.processing_time = processing_time
            
            progress_callback.update("Processing complete!", 1.0)
            
            # Update UI on main thread
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            error_msg = f"Processing failed: {str(e)}"
            self.root.after(0, lambda: self._processing_error(error_msg))
    
    def _create_mock_results(self):
        """Create mock results for demonstration purposes."""
        # This would be replaced with actual processing results
        from ..core.data_models import Move, Position, PieceType, Color, PieceKind, GameState, BoardState
        
        # Create some mock moves
        mock_moves = [
            Move(
                from_square=Position(4, 6),  # e2
                to_square=Position(4, 4),    # e4
                piece=PieceType(Color.WHITE, PieceKind.PAWN)
            ),
            Move(
                from_square=Position(4, 1),  # e7
                to_square=Position(4, 3),    # e5
                piece=PieceType(Color.BLACK, PieceKind.PAWN),
                is_flagged=True,
                flag_reason="Low confidence piece recognition"
            )
        ]
        
        # Create mock game state
        mock_board_state = BoardState(
            squares={},
            timestamp=0.0,
            confidence=0.95
        )
        
        mock_game_state = GameState(
            current_position=mock_board_state,
            move_history=mock_moves
        )
        
        # Create mock metadata
        mock_game_metadata = GameMetadata(
            event="Casual Game",
            site="Home",
            date=datetime.now().strftime("%Y.%m.%d"),
            white_player="White",
            black_player="Black",
            result="*"
        )
        
        # Generate PGN and FEN
        pgn_content = self.pgn_generator.generate_pgn(mock_game_state, mock_game_metadata)
        fen_sequence = ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                       "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"]
        
        # Store results
        self.processing_results.game_state = mock_game_state
        self.processing_results.game_metadata = mock_game_metadata
        self.processing_results.pgn_content = pgn_content
        self.processing_results.fen_sequence = fen_sequence
    
    def _update_progress(self, message: str, progress: float):
        """
        Update progress indicators (called from processing thread).
        
        Args:
            message: Status message
            progress: Progress value (0.0 to 1.0)
        """
        # Schedule UI update on main thread
        self.root.after(0, lambda: self._update_progress_ui(message, progress))
    
    def _update_progress_ui(self, message: str, progress: float):
        """
        Update progress UI elements (called on main thread).
        
        Args:
            message: Status message
            progress: Progress value (0.0 to 1.0)
        """
        self.status_var.set(message)
        self.progress_var.set(progress * 100)
        
        # Update estimated time (simplified calculation)
        if progress > 0.1 and hasattr(self, '_processing_start_time'):
            elapsed = (datetime.now() - self._processing_start_time).total_seconds()
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            if remaining > 0:
                self.time_var.set(f"Estimated time remaining: {remaining:.1f}s")
        elif progress < 0.1:
            self._processing_start_time = datetime.now()
            self.time_var.set("Calculating estimated time...")
    
    def _processing_complete(self):
        """Handle successful processing completion."""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.browse_button.config(state="normal")
        
        # Display results
        self._display_results()
        
        # Update error correction interface
        if self.processing_results.game_state:
            self.error_correction.update_game_state(self.processing_results.game_state)
        
        # Show completion message
        processing_time = self.processing_results.processing_time or 0
        messagebox.showinfo("Processing Complete", 
                          f"Video analysis completed successfully!\n"
                          f"Processing time: {processing_time:.1f} seconds")
    
    def _processing_error(self, error_message: str):
        """
        Handle processing error.
        
        Args:
            error_message: Error message to display
        """
        self.is_processing = False
        self.process_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.browse_button.config(state="normal")
        
        self.status_var.set("Processing failed")
        self.progress_var.set(0)
        self.time_var.set("")
        
        messagebox.showerror("Processing Error", error_message)
    
    def _cancel_processing(self):
        """Cancel ongoing processing."""
        # In a full implementation, this would signal the processing thread to stop
        self.is_processing = False
        self.process_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.browse_button.config(state="normal")
        
        self.status_var.set("Processing cancelled")
        self.progress_var.set(0)
        self.time_var.set("")
    
    def _display_results(self):
        """Display processing results in the UI."""
        if not self.processing_results.game_state:
            return
        
        # Display moves
        moves_text = self._format_moves_display(self.processing_results.game_state.move_history)
        self.moves_text.delete(1.0, tk.END)
        self.moves_text.insert(1.0, moves_text)
        
        # Display PGN
        if self.processing_results.pgn_content:
            self.pgn_text.delete(1.0, tk.END)
            self.pgn_text.insert(1.0, self.processing_results.pgn_content)
        
        # Display FEN
        if self.processing_results.fen_sequence:
            fen_text = "\n".join([f"Move {i}: {fen}" 
                                 for i, fen in enumerate(self.processing_results.fen_sequence)])
            self.fen_text.delete(1.0, tk.END)
            self.fen_text.insert(1.0, fen_text)
        
        # Enable export buttons
        self.export_pgn_button.config(state="normal")
        self.export_fen_button.config(state="normal")
        
        # Highlight problematic moves
        self.highlight_problematic_moves()
    
    def _format_moves_display(self, moves: List[Move]) -> str:
        """
        Format moves for display in the moves tab.
        
        Args:
            moves: List of moves to format
            
        Returns:
            Formatted moves string
        """
        if not moves:
            return "No moves detected."
        
        formatted_moves = []
        for i, move in enumerate(moves):
            move_num = (i // 2) + 1
            color = "White" if i % 2 == 0 else "Black"
            
            # Format move notation (simplified)
            from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
            to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
            piece_name = move.piece.type.value.capitalize()
            
            move_text = f"{move_num}. {color}: {piece_name} {from_square} -> {to_square}"
            
            if move.captured_piece:
                move_text += f" (captures {move.captured_piece.type.value})"
            
            if move.is_flagged:
                move_text += f" [FLAGGED: {move.flag_reason}]"
            
            formatted_moves.append(move_text)
        
        return "\n".join(formatted_moves)
    
    def _export_pgn(self):
        """Export PGN content to file."""
        if not self.processing_results.pgn_content:
            messagebox.showwarning("No Data", "No PGN data available to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export PGN File",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.processing_results.pgn_content)
                messagebox.showinfo("Export Successful", f"PGN exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export PGN:\n{str(e)}")
    
    def _export_fen(self):
        """Export FEN sequence to file."""
        if not self.processing_results.fen_sequence:
            messagebox.showwarning("No Data", "No FEN data available to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export FEN File",
            defaultextension=".fen",
            filetypes=[("FEN files", "*.fen"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for i, fen in enumerate(self.processing_results.fen_sequence):
                        f.write(f"Move {i}: {fen}\n")
                messagebox.showinfo("Export Successful", f"FEN sequence exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export FEN:\n{str(e)}")
    
    def _on_move_corrected(self, move_index: int, corrected_move: Optional[Move]):
        """
        Handle move correction from the error correction interface.
        
        Args:
            move_index: Index of the corrected move (-1 for bulk operations)
            corrected_move: The corrected move (None if deleted)
        """
        if move_index == -1:
            # Bulk operation (e.g., clear all flags)
            self.logger.info("Bulk move correction operation performed")
        elif corrected_move is None:
            # Move was deleted
            self.logger.info(f"Move {move_index + 1} was deleted")
        else:
            # Move was corrected
            self.logger.info(f"Move {move_index + 1} was corrected")
        
        # Refresh the results display
        self._display_results()
        
        # Regenerate PGN and FEN with corrected moves
        if self.processing_results.game_state and self.processing_results.game_metadata:
            try:
                # Regenerate PGN
                self.processing_results.pgn_content = self.pgn_generator.generate_pgn(
                    self.processing_results.game_state, 
                    self.processing_results.game_metadata
                )
                
                # Regenerate FEN sequence (simplified - would need full game state sequence)
                current_fen = self.fen_generator.generate_fen(self.processing_results.game_state)
                self.processing_results.fen_sequence = [current_fen]
                
                # Update displays
                self._display_results()
                
            except Exception as e:
                self.logger.error(f"Error regenerating notation after correction: {str(e)}")
                messagebox.showwarning("Regeneration Error", 
                                     f"Could not regenerate notation after correction:\n{str(e)}")
    
    def highlight_problematic_moves(self):
        """
        Highlight problematic moves in the moves display.
        
        This method adds visual highlighting to flagged moves in the moves text widget.
        """
        if not self.processing_results.game_state:
            return
        
        # Configure text tags for highlighting
        self.moves_text.tag_configure("flagged", background="yellow", foreground="red")
        self.moves_text.tag_configure("normal", background="white", foreground="black")
        
        # Clear existing tags
        self.moves_text.tag_remove("flagged", 1.0, tk.END)
        self.moves_text.tag_remove("normal", 1.0, tk.END)
        
        # Get current text content
        content = self.moves_text.get(1.0, tk.END)
        lines = content.split('\n')
        
        # Highlight flagged moves
        for i, move in enumerate(self.processing_results.game_state.move_history):
            if move.is_flagged and i < len(lines):
                line_start = f"{i + 1}.0"
                line_end = f"{i + 1}.end"
                self.moves_text.tag_add("flagged", line_start, line_end)
    
    def add_export_options_menu(self):
        """Add advanced export options menu."""
        # Create menu bar if it doesn't exist
        if not hasattr(self.root, 'menubar'):
            self.root.menubar = tk.Menu(self.root)
            self.root.config(menu=self.root.menubar)
        
        # Add export menu
        export_menu = tk.Menu(self.root.menubar, tearoff=0)
        self.root.menubar.add_cascade(label="Export", menu=export_menu)
        
        export_menu.add_command(label="Export PGN...", command=self._export_pgn)
        export_menu.add_command(label="Export FEN...", command=self._export_fen)
        export_menu.add_separator()
        export_menu.add_command(label="Export Corrected Moves Only", command=self._export_corrected_moves)
        export_menu.add_command(label="Export Error Report", command=self._export_error_report)
    
    def _export_corrected_moves(self):
        """Export only the corrected (non-flagged) moves."""
        if not self.processing_results.game_state:
            messagebox.showwarning("No Data", "No game data available to export.")
            return
        
        # Filter out flagged moves
        corrected_moves = [move for move in self.processing_results.game_state.move_history 
                          if not move.is_flagged]
        
        if not corrected_moves:
            messagebox.showwarning("No Corrected Moves", "No corrected moves available to export.")
            return
        
        # Create a temporary game state with only corrected moves
        temp_game_state = GameState(
            current_position=self.processing_results.game_state.current_position,
            move_history=corrected_moves
        )
        
        # Generate PGN for corrected moves
        pgn_content = self.pgn_generator.generate_pgn(
            temp_game_state, 
            self.processing_results.game_metadata
        )
        
        # Save to file
        filename = filedialog.asksaveasfilename(
            title="Export Corrected Moves (PGN)",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(pgn_content)
                messagebox.showinfo("Export Successful", 
                                  f"Corrected moves exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export corrected moves:\n{str(e)}")
    
    def _export_error_report(self):
        """Export a detailed error report."""
        if not self.processing_results.game_state:
            messagebox.showwarning("No Data", "No game data available to export.")
            return
        
        # Generate error report
        report_lines = []
        report_lines.append("Chess Video Analyzer - Error Report")
        report_lines.append("=" * 40)
        report_lines.append("")
        
        # Summary
        correction_summary = self.error_correction.get_correction_summary()
        report_lines.append("SUMMARY:")
        report_lines.append(f"Total Moves: {correction_summary['total_moves']}")
        report_lines.append(f"Flagged Moves: {correction_summary['flagged_moves']}")
        report_lines.append(f"Accuracy: {correction_summary['accuracy_percentage']:.1f}%")
        report_lines.append("")
        
        # Detailed errors
        flagged_moves = [move for move in self.processing_results.game_state.move_history if move.is_flagged]
        if flagged_moves:
            report_lines.append("FLAGGED MOVES:")
            for i, move in enumerate(self.processing_results.game_state.move_history):
                if move.is_flagged:
                    move_num = (i // 2) + 1
                    color = "White" if i % 2 == 0 else "Black"
                    from_square = f"{chr(ord('a') + move.from_square.x)}{8 - move.from_square.y}"
                    to_square = f"{chr(ord('a') + move.to_square.x)}{8 - move.to_square.y}"
                    
                    report_lines.append(f"Move {move_num} ({color}): {move.piece.type.value.title()} {from_square}-{to_square}")
                    report_lines.append(f"  Issue: {move.flag_reason or 'Unknown'}")
                    report_lines.append("")
        else:
            report_lines.append("No flagged moves found.")
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        filename = filedialog.asksaveasfilename(
            title="Export Error Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                messagebox.showinfo("Export Successful", f"Error report exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export error report:\n{str(e)}")
    
    def run(self):
        """Start the main application loop."""
        self.root.mainloop()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = MainInterface(root)
    app.run()


if __name__ == "__main__":
    main()
"""
User interface module for the chess video analyzer.

This module provides the main application interface including:
- Video file import (drag-drop and file dialog)
- Progress indicators for processing operations
- Results display for generated moves
- Export functionality for PGN and FEN files

Requirements: 8.2, 8.3, 8.4, 8.5
"""

from .main_interface import MainInterface, ProgressCallback, ProcessingResults
from .error_correction_interface import ErrorCorrectionInterface, MoveEditDialog
from .app_launcher import launch_application

__all__ = [
    'MainInterface',
    'ProgressCallback', 
    'ProcessingResults',
    'ErrorCorrectionInterface',
    'MoveEditDialog',
    'launch_application'
]
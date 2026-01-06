"""
Application launcher for Chess Video Analyzer.

This module provides a simple way to launch the main application interface.
"""

import sys
import tkinter as tk
from tkinter import messagebox
import logging

try:
    from .main_interface import MainInterface
except ImportError:
    # Handle case where module is run directly
    from main_interface import MainInterface


def check_dependencies():
    """
    Check if required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available
    """
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        error_msg = (
            "Missing required dependencies:\n" +
            "\n".join(f"- {dep}" for dep in missing_deps) +
            "\n\nPlease install them using:\n" +
            f"pip install {' '.join(missing_deps)}"
        )
        messagebox.showerror("Missing Dependencies", error_msg)
        return False
    
    return True


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chess_video_analyzer.log')
        ]
    )


def launch_application():
    """Launch the Chess Video Analyzer application."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Check dependencies
        if not check_dependencies():
            return
        
        # Create and configure root window
        root = tk.Tk()
        
        # Set window icon (if available)
        try:
            # This would set an icon if we had one
            # root.iconbitmap('icon.ico')
            pass
        except:
            pass
        
        # Create and run application
        app = MainInterface(root)
        logger.info("Starting Chess Video Analyzer application")
        app.run()
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        messagebox.showerror("Startup Error", 
                           f"Failed to start Chess Video Analyzer:\n{str(e)}")


if __name__ == "__main__":
    launch_application()
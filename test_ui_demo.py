#!/usr/bin/env python3
"""
Simple demo script to test the Chess Video Analyzer UI.
"""

import tkinter as tk
from chess_video_analyzer.ui import launch_application

if __name__ == "__main__":
    print("Starting Chess Video Analyzer UI Demo...")
    try:
        launch_application()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
# Chess Video Analyzer

A comprehensive Python system for analyzing chess game videos recorded from above the board and automatically generating standard chess notation files (PGN and FEN).

## 🎯 Overview

Chess Video Analyzer uses computer vision and machine learning techniques to process video recordings of chess games and extract move sequences. The system can detect the chessboard, recognize pieces, track movements, and generate standard chess notation that can be imported into popular chess platforms like Chess.com and Lichess.

## ✨ Features

- **Automatic Board Detection**: Identifies chess board position and orientation in video frames
- **Piece Recognition**: Uses computer vision to identify chess pieces and their positions
- **Move Tracking**: Detects piece movements between frames and validates chess rules
- **Special Move Support**: Handles castling, en passant, and pawn promotion
- **PGN Generation**: Creates Portable Game Notation files compatible with chess software
- **FEN Generation**: Generates Forsyth-Edwards Notation for each position
- **Quality Control**: Confidence-based quality assessment and error flagging
- **User Interface**: Intuitive GUI for video processing and result review
- **Error Correction**: Manual correction interface for problematic moves

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-video-analyzer.git
cd chess-video-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from chess_video_analyzer.main import ChessVideoAnalyzer

# Initialize the analyzer
analyzer = ChessVideoAnalyzer()

# Load and process a video
analyzer.load_video("path/to/your/chess_game.mp4")
results = analyzer.process_video()

# Export results
analyzer.export_pgn("game.pgn")
analyzer.export_fen("positions.fen")
```

### GUI Usage

```bash
# Launch the graphical interface
python -m chess_video_analyzer.main
```

## 🏗️ Architecture

The system follows a modular pipeline architecture:

```
Video Input → Frame Extraction → Board Detection → Piece Recognition → 
Move Tracking → Game State Management → Notation Generation → Export
```

### Core Components

- **VideoProcessor**: Handles video file input and frame extraction
- **BoardDetector**: Locates and tracks the chessboard in video frames
- **PieceRecognizer**: Identifies chess pieces using computer vision
- **MoveTracker**: Detects moves by comparing consecutive board states
- **GameStateManager**: Maintains game state and validates moves
- **PGNGenerator**: Creates standard PGN notation files
- **FENGenerator**: Generates FEN strings for each position
- **QualityController**: Assesses confidence and flags uncertain moves

## 📁 Project Structure

```
chess_video_analyzer/
├── core/                   # Core data models and enumerations
│   ├── data_models.py     # Position, Move, GameState, etc.
│   └── __init__.py
├── video/                  # Video processing components
│   ├── processor.py       # Video file handling and frame extraction
│   └── __init__.py
├── detection/              # Computer vision components
│   ├── board_detector.py  # Chess board detection
│   ├── piece_recognizer.py # Piece recognition
│   └── __init__.py
├── tracking/               # Move tracking and analysis
│   ├── move_tracker.py    # Move detection between frames
│   └── __init__.py
├── notation/               # Chess notation generation
│   ├── game_state_manager.py # Game state and rule validation
│   ├── pgn_generator.py   # PGN file generation
│   ├── fen_generator.py   # FEN notation generation
│   └── __init__.py
├── quality/                # Quality control and error handling
│   ├── quality_controller.py # Confidence assessment
│   └── __init__.py
├── ui/                     # User interface components
│   ├── main_interface.py  # Main GUI application
│   ├── error_correction_interface.py # Manual correction tools
│   ├── app_launcher.py    # Application launcher
│   └── __init__.py
└── main.py                 # Main application entry point

tests/                      # Comprehensive test suite
├── test_*.py              # Unit and integration tests
└── __init__.py

.kiro/specs/               # Project specifications
├── requirements.md        # Detailed requirements
├── design.md             # System design document
└── tasks.md              # Implementation tasks
```

## 🧪 Testing

The project includes comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chess_video_analyzer

# Run property-based tests only
pytest -m property

# Run specific test file
pytest tests/test_move_tracking.py -v
```

### Test Coverage

- **175+ test cases** covering all major components
- **Property-based testing** using Hypothesis for comprehensive validation
- **Integration tests** for end-to-end functionality
- **UI responsiveness tests** for user interface components

## 📋 Requirements

### System Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- python-chess 1.999+

### Development Requirements

- pytest 7.4+ (for testing)
- hypothesis 6.82+ (for property-based testing)
- tkinter (for GUI, usually included with Python)

### Supported Video Formats

- MP4
- AVI
- MOV

## 🎮 Usage Examples

### Command Line Processing

```bash
# Process a video file
python -m chess_video_analyzer.main video.mp4 --output-dir ./results

# Set confidence threshold
python -m chess_video_analyzer.main video.mp4 --confidence 0.8

# Process without GUI
python -m chess_video_analyzer.main video.mp4 --no-ui
```

### Programmatic Usage

```python
from chess_video_analyzer.main import ChessVideoAnalyzer
from chess_video_analyzer.core.data_models import GameMetadata

# Initialize with custom settings
analyzer = ChessVideoAnalyzer(
    confidence_threshold=0.8,
    enable_quality_control=True
)

# Set up game metadata
metadata = GameMetadata(
    event="Club Championship",
    site="Local Chess Club",
    white_player="Alice",
    black_player="Bob"
)

# Process video
analyzer.load_video("game.mp4")
results = analyzer.process_video(game_metadata=metadata)

# Check quality report
quality_report = analyzer.get_quality_report()
if quality_report:
    print(f"Overall confidence: {quality_report.overall_confidence:.2f}")
    print(f"Issues found: {len(quality_report.issues)}")

# Get flagged moves for review
flagged_moves = analyzer.get_flagged_moves()
for move in flagged_moves:
    print(f"Flagged: {move.flag_reason}")
```

## 🔧 Configuration

The system can be configured through various parameters:

- **Confidence Threshold**: Minimum confidence for accepting piece recognition (default: 0.7)
- **Frame Interval**: Time between processed frames in seconds (default: 1.0)
- **Quality Control**: Enable/disable quality assessment features
- **UI Mode**: Enable/disable graphical user interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- python-chess library for chess logic validation
- Hypothesis library for property-based testing
- Chess.com and Lichess for PGN format standards

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/chess-video-analyzer/issues) page
2. Create a new issue with detailed information
3. Include sample video files if possible (ensure no personal information)

## 🗺️ Roadmap

- [ ] Deep learning model for improved piece recognition
- [ ] Support for different chess set styles
- [ ] Real-time video processing
- [ ] Mobile app integration
- [ ] Cloud processing capabilities
- [ ] Multi-language support
# Chess Video Analyzer

A Python system for analyzing chess game videos recorded from above the board and generating standard chess notation files (PGN and FEN).

## Features

- Automatic chess board detection in video frames
- Piece recognition and position tracking
- Move detection and validation
- PGN (Portable Game Notation) file generation
- FEN (Forsyth-Edwards Notation) sequence generation
- Error handling and quality assurance

## Installation

```bash
pip install -r requirements.txt
```

## Development

### Running Tests

```bash
pytest
```

### Running Property-Based Tests

```bash
pytest -m property
```

## Project Structure

```
chess_video_analyzer/
├── core/           # Core data models and enumerations
├── video/          # Video processing components
├── detection/      # Board and piece detection
├── tracking/       # Move tracking and game state management
├── notation/       # PGN and FEN generation
└── ui/            # User interface components
```

## Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- python-chess 1.999+
- pytest 7.4+ (for testing)
- hypothesis 6.82+ (for property-based testing)
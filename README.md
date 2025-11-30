# Face Recognition System

Lightweight GUI and CLI tools for age, gender, race, and emotion recognition using DeepFace and OpenCV.

Table of contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
  - [GUI](#gui)
  - [CLI](#cli)
- [Project layout](#project-layout)
- [Data handling & data/raw note](#data-handling--dataraw-note)
- [Development & Contributing](#development--contributing)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This repository provides a PyQt5-based GUI and several command-line utilities that perform face analysis (age, gender, race, emotion) using the DeepFace library. It supports single-image analysis, video processing (preview + optional save), and webcam modes.

## Features
- Analyze single images with DeepFace.
- Process videos frame-by-frame with live preview and optional saving.
- Live webcam modes (regular and a faster mode).
- Background threads for non-blocking GUI operations.

## Requirements
- Python 3.8+
- See `requirements.txt` for the exact list (typically: deepface, opencv-python, PyQt5, numpy, tensorflow or pytorch backend as required).

## Installation
From project root:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running

### GUI
From project root:
```bash
python src/gui/main_app.py
```
UI actions:
- Load Image — opens file dialog and analyzes the selected image.
- Load Video — open video file, preview and optionally save processed output.
- Start Webcam — launches the live webcam script.

### CLI
Analyze a single image:
```bash
python scripts/analyze_image.py /path/to/image.jpg
```

Process a video and save output:
```bash
python scripts/analyze_video.py input.mp4 -o output.mp4
```

Preview a video (no save):
```bash
python scripts/analyze_video.py input.mp4 --play
```

Run live webcam:
```bash
python scripts/live_webcam.py
python scripts/live_webcam_fast.py
```

## Project layout
- src/gui/main_app.py — PyQt5 GUI and background threads.
- scripts/analyze_image.py — CLI single-image analyzer.
- scripts/analyze_video.py — CLI video processing.
- scripts/live_webcam.py, scripts/live_webcam_fast.py — webcam modes.
- requirements.txt — Python dependencies.
- data/ — optional sample data and metadata (may contain a `raw/` folder).

## Data handling & data/raw note
The codebase uses user-supplied image/video paths (file dialogs or CLI args). There are no mandatory hard-coded dependencies on `data/raw`. If you want to remove the `data/raw` folder:
- Move or archive first:
  ```bash
  mv data/raw data/raw.backup
  ```
- Or remove:
  ```bash
  rm -rf data/raw
  ```
If files under `data/raw` are referenced in external notebooks or CI, update those references before deleting.

## Development & Contributing
The previous CONTRIBUTING and License sections were placeholders. Use the guidance below as a minimal, accurate contributing workflow to get started.

Basic contribution workflow
1. Fork the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/short-description
   ```
3. Make changes, keep commits small and focused.
4. Run tests and linters locally (see [Testing](#testing)).
5. Push and open a pull request (describe the change, testing performed, and any backward-incompatible impacts).

Coding style
- Follow PEP8 for Python code.
- Use 4-space indentation.
- Prefer clear, small functions; write docstrings for public functions and classes.

Testing
- Add unit tests under a `tests/` directory (pytest recommended).
- Ensure tests are runnable with:
  ```bash
  pytest
  ```

Commits and PRs
- Use descriptive commit messages (imperative present tense).
- Include a short description and link to any relevant issue.
- Provide screenshots or short notes for UI changes.

Local development tips
- Use a virtual environment.
- Install dev dependencies (linters, formatters) as needed.
- Consider using pre-commit hooks to enforce formatting.


## Testing
- Unit tests: add tests under `tests/` and run with `pytest`.
- Manual UI testing: run `src/gui/main_app.py` and exercise all GUI actions.
- Video/webcam testing: test on a machine with a camera and display server (X/Wayland).

## Troubleshooting
- GUI fails to open on Linux/WSL: ensure an X server / display is available.
- DeepFace model downloads: first run may take extra time to download model weights.
- GPU usage: ensure a compatible TensorFlow/PyTorch + CUDA setup if using GPU.

## License
This repository is licensed under the MIT License. A copy of the license is provided below — add a `LICENSE` file with the same text if you accept and want to publish under MIT.

MIT License
Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.



## Acknowledgements
- DeepFace project (https://github.com/serengil/deepface)
- OpenCV (cv2)
- PyQt5

# Face Recognition System

Lightweight GUI and CLI tools for age, gender, race, and emotion recognition using DeepFace.

## Quick links (workspace)
- GUI app: `src/gui/main_app.py` — the PyQt5 GUI and main entry point.
- CLI scripts:
  - `scripts/analyze_image.py` — single image analysis.
  - `scripts/analyze_video.py` — video processing and preview/save.
  - `scripts/live_webcam.py` / `scripts/live_webcam_fast.py` — live webcam modes.
- Dependencies: `requirements.txt`


## Summary
This project provides a small GUI and several command-line scripts that use the DeepFace library to analyze faces in images and videos. The GUI allows loading an image, loading a video (with preview and optional save), and starting a webcam stream.

## Requirements
- Python 3.8+
- See `requirements.txt` for exact packages. Typical dependencies include `deepface`, `opencv-python`, and `PyQt5`.

## Install
From the project root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

## Run the GUI
From the project root run:

```bash
python src/gui/main_app.py
```

The GUI exposes:
- Load Image — analyzes a single image using a background thread.
- Load Video — processes video frames with live preview and optional saving.
- Start Webcam — starts a webcam-based analyzer (calls the live webcam script).

## CLI usage

Analyze a single image:
```bash
python scripts/analyze_image.py /path/to/image.jpg
```

Process a video (save output):
```bash
python scripts/analyze_video.py input.mp4 -o output.mp4
```

Play/preview a video without saving:
```bash
python scripts/analyze_video.py input.mp4 --play
```

Run live webcam (detailed or fast):
```bash
python scripts/live_webcam.py
python scripts/live_webcam_fast.py
```

## Troubleshooting
- On Linux (or WSL) ensure a display server/X server is available for PyQt windows.
- The first run may be slow while DeepFace downloads models.

## Contributing
Open issues or pull requests for bug fixes and improvements. Add a license file if you plan to publish this repository.

## License
Add your license of choice here (e.g., MIT, Apache-2.0).

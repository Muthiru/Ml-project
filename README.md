# Face Recognition System

Deep learning-based face analysis for age, gender, race, and emotion recognition using DeepFace and OpenCV.

## Overview

Command-line tools for facial attribute recognition supporting single images, video files, and live webcam feeds. Includes comprehensive model validation system with automated dataset setup.

## Features

- **Single Image Analysis** - Analyze age, gender, race, and emotion from photos
- **Video Processing** - Process video files with face detection and annotation
- **Live Webcam** - Real-time face recognition (standard and fast modes)
- **Model Validation** - Comprehensive validation system with automated dataset setup
- **Multiple Detectors** - Support for OpenCV, RetinaFace, MTCNN, and more

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 5GB disk space
- Webcam ( for live detection)

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Image Analysis

```bash
python scripts/analyze_image.py <image_path>
```

Example:
```bash
python scripts/analyze_image.py photo.jpg
```

Output: Age, gender, race, and emotion predictions with confidence scores.

---

### Video Processing

```bash
python scripts/analyze_video.py <input.mp4> <output.mp4> [options]
```

Options:
- `--detector opencv|retinaface|mtcnn` - Face detector (default: opencv)
- `--interval N` - Process every N frames (default: 30)
- `--play` - Play video after processing

Example:
```bash
python scripts/analyze_video.py video.mp4 output.mp4 --detector retinaface
```

---

### Live Webcam

**Standard Mode** (2-second updates, detailed info):
```bash
python scripts/live_webcam.py
```

**Fast Mode** (3-second updates, countdown timer):
```bash
python scripts/live_webcam_fast.py
```

Controls:
- Press 's' to save screenshot
- Press 'q' to quit

---

## Model Validation

Comprehensive validation system for testing model performance.

### 1. Setup Validation Datasets

```bash
# Download all datasets
python scripts/setup_validation_datasets.py

# Specific dataset only
python scripts/setup_validation_datasets.py --datasets emotion
```

**Kaggle Setup** (first time):
1. Get API key from https://www.kaggle.com/settings
2. Download `kaggle.json`
3. Place in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows)
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 2. Run Validation

```bash
# Validate all attributes
python scripts/validate_model.py

# Specific attributes
python scripts/validate_model.py --attributes age emotion

# Different detector
python scripts/validate_model.py --detector retinaface
```

**Output:** CSV results, confusion matrices, and summary reports in `validation_results/`

---

## Project Structure

```
Ml project/
├── scripts/
│   ├── analyze_image.py           # Single image analysis
│   ├── analyze_video.py           # Video processing
│   ├── live_webcam.py             # Real-time detection
│   ├── live_webcam_fast.py        # Optimized real-time
│   ├── validate_model.py          # Model validation
│   └── setup_validation_datasets.py  # Dataset setup
├── data/
│   └── validation/                # Validation datasets
│       ├── age/
│       ├── emotion/
│       ├── gender/
│       └── race/
├── validation_results/            # Validation output
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## Validation Datasets

| Attribute | Classes | Images | Source |
|-----------|---------|--------|--------|
| Age | 8 brackets | 240 | UTKFace |
| Gender | 2 classes | 40 | UTKFace |
| Race | 5 classes | 75 | UTKFace |
| Emotion | 4 emotions | 60 | FER-2013 |

---

## Troubleshooting

**"No module named 'deepface'"**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**"Unable to download from Kaggle"**
- Setup Kaggle credentials (see Model Validation section)

**"No face detected"**
- Ensure clear, front-facing face in good lighting
- Try different detector: `--detector retinaface`

**Webcam not opening**
- Check camera permissions
- Close other apps using camera
- Try different camera index (edit `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

---

## Performance

| Operation | Processing Time | Accuracy |
|-----------|----------------|----------|
| Single Image | ~1s (OpenCV) | Varies |
| Video (per frame) | 0.5-5s | By detector |
| Emotion Recognition | 58% | 60 images |
| Age (generous ±1 bracket) | 50-65% | 240 images |
| Gender | 95-100% | 40 images |
| Race | 46-88% | 75 images |

---

## Technical Details

**Models:**
- DeepFace framework with VGG-Face, FaceNet, ArcFace
- Pre-trained on VGGFace2, MS-Celeb-1M, FER-2013

**Detectors:**
- OpenCV Haar Cascades (fast, ~0.5s)
- RetinaFace (accurate, ~5s)
- MTCNN (balanced, ~2s)

---



## Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) - Face recognition framework
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/) - Age, gender, race data
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) - Emotion recognition data

---

## Author

**Danylo**  
December 2025

```text
data/validation/
├── age/
│   ├── 0-10/       (15 images per bracket)
│   ├── 11-20/
│   ├── 21-30/
│   ├── 31-40/
│   ├── 41-50/
│   ├── 51-60/
│   ├── 61-70/
│   └── 71-80/
├── gender/
│   ├── Male/       (20 images)
│   └── Female/     (20 images)
├── race/
│   ├── White/      (15 images per category)
│   ├── Black/
│   ├── Asian/
│   ├── Indian/
│   └── Latino_Hispanic/
└── emotion/
    ├── happy/      (15 images per emotion)
    ├── sad/
    ├── angry/
    └── neutral/
```

**Total:** ~200 validation images across all attributes

### Validation Output

The validation script generates:

1. **Console Report**: Real-time progress and accuracy metrics
2. **CSV Files**: Detailed prediction results for each image
   - Format: `{attribute}_results_{detector}_{timestamp}.csv`
3. **Confusion Matrices**: Visual heatmaps showing prediction performance
   - Format: `{attribute}_confusion_matrix_{detector}_{timestamp}.png`
4. **Summary Report**: Text file with overall validation results
   - Format: `validation_summary_{detector}_{timestamp}.txt`

All outputs are saved to `validation_results/` directory (or custom path with `--output`)

### Understanding Results

**Accuracy Metrics:**
- **Strict Accuracy**: Exact match between prediction and ground truth
- **Generous Accuracy** (age only): Allows ±1 age bracket error (e.g., predicting 25 for someone in 21-30 bracket is correct, predicting 35 gets credit for being close)

**Confusion Matrix:**
- Diagonal cells: Correct predictions
- Off-diagonal cells: Misclassifications
- Darker blue: Higher count

**Good Performance Indicators:**
- Age: >70% strict, >85% generous
- Gender: >90% strict
- Emotion: >65% strict (emotions are subjective)
- Race: >75% strict

### Manual Dataset Setup (Alternative)

If you prefer manual setup or don't have Kaggle access, create the folder structure above and add your own labeled images (10-15 per class recommended).

## Development & Contributing


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

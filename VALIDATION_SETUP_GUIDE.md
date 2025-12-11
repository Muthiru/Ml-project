# Validation Setup Guide

This guide explains how to set up validation datasets for testing model performance on age, gender, race, and emotion predictions.

## Quick Setup (Automated)

The easiest way to set up all validation datasets is using the automated setup script:

```bash
# Setup all datasets (age, gender, race, emotion)
python scripts/setup_validation_datasets.py

# Or setup specific datasets only
python scripts/setup_validation_datasets.py --datasets age emotion
```

## Prerequisites

### 1. Install Kaggle CLI
```bash
pip install kaggle
```

### 2. Get Kaggle API Key
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json`
5. Move it to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Dataset Structure

After setup, you'll have the following structure:

```
data/validation/
├── age/
│   ├── 0-10/       (15 images)
│   ├── 11-20/      (15 images)
│   ├── 21-30/      (15 images)
│   ├── 31-40/      (15 images)
│   ├── 41-50/      (15 images)
│   ├── 51-60/      (15 images)
│   ├── 61-70/      (15 images)
│   └── 71-80/      (15 images)
├── gender/
│   ├── Male/       (20 images)
│   └── Female/     (20 images)
├── race/
│   ├── White/      (15 images)
│   ├── Black/      (15 images)
│   ├── Asian/      (15 images)
│   ├── Indian/     (15 images)
│   └── Latino_Hispanic/ (15 images)
└── emotion/
    ├── happy/      (15 images)
    ├── sad/        (15 images)
    ├── angry/      (15 images)
    └── neutral/    (15 images)
```

**Total:** ~200 validation images across all attributes

## Verify Setup

Check that datasets were created correctly:

```bash
# Check all validation folders
ls -lR data/validation/

# Count images per folder
find data/validation -name "*.jpg" -o -name "*.png" | wc -l
```

---

## Alternative: Manual Download from Google Images

If you don't want to use Kaggle, manually download:

1. **Happy faces**: Google "happy face" → Save 12-13 images → Put in `data/validation/emotion/happy/`
2. **Sad faces**: Google "sad face" → Save 12-13 images → Put in `data/validation/emotion/sad/`
3. **Angry faces**: Google "angry face" → Save 12-13 images → Put in `data/validation/emotion/angry/`
4. **Neutral faces**: Google "neutral face expression" → Save 12-13 images → Put in `data/validation/emotion/neutral/`

**Important**: 
- Use clear, frontal face images
- Avoid cartoons or emojis (use real human faces)
- Ensure faces are well-lit and visible

---

## Run Validation
Once images are in place:

## Running Validation

Once datasets are set up, run validation:

```bash
# Validate all attributes
python scripts/validate_model.py

# Validate specific attributes
python scripts/validate_model.py --attributes age emotion

# Use different detector backend
python scripts/validate_model.py --detector retinaface
```

## Expected Output

For each attribute validated, you'll get:
- **CSV file**: Detailed predictions for each image
  - Format: `{attribute}_results_{detector}_{timestamp}.csv`
- **Confusion matrix**: Visual heatmap of predictions
  - Format: `{attribute}_confusion_matrix_{detector}_{timestamp}.png`
- **Summary report**: Overall accuracy metrics
  - Format: `validation_summary_{detector}_{timestamp}.txt`
- **Terminal output**: Real-time progress and accuracy percentages

All outputs are saved to `validation_results/` directory.

## Manual Setup (Alternative)

If you prefer to manually set up validation data or don't have Kaggle access:

1. Create the folder structure shown above
2. Add 10-15 labeled images per class
3. Use images from public datasets or your own collection
4. Ensure images are named clearly (e.g., `happy_001.jpg`)

## Troubleshooting

**Kaggle credentials not found:**
- Make sure `kaggle.json` is in `~/.kaggle/`
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Dataset download fails:**
- Check internet connection
- Verify Kaggle account is active
- Try manually downloading from Kaggle website

**Not enough images in some classes:**
- The script selects randomly; some classes may have fewer images
- This is normal and won't affect validation significantly
- You can manually add more images if needed

## Dataset Sources

- **Emotion**: FER-2013 dataset (msambare/fer2013)
- **Age/Gender/Race**: UTKFace dataset (jangedoo/utkface-new)

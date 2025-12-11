"""
Unified Validation Dataset Setup Script
Downloads and organizes validation datasets for age, gender, race, and emotion
"""

import os
import sys
import shutil
import random
import subprocess
import zipfile
from pathlib import Path
from tqdm import tqdm

# ==================== CONSTANTS ====================
# Repeated strings
UTKFACE_DATASET = "jangedoo/utkface-new"
UTKFACE_ZIP = "utkface-new.zip"
UTKFACE_SOURCE = "utkface_new"
IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg')
PROCESSING_IMAGES_DESC = "Processing images"

# ==================== CONFIGURATION ====================
VALIDATION_BASE = "data/validation"

DATASETS = {
    "emotion": {
        "kaggle_name": "msambare/fer2013",
        "source_folder": "test",
        "images_per_class": 15,
        "classes": {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "neutral": "neutral"
        },
        "zip_name": "fer2013.zip",
        "temp_dir": "fer2013_temp"
    },
    "age": {
        "kaggle_name": UTKFACE_DATASET,
        "source_folder": UTKFACE_SOURCE,  # or "UTKFace"
        "images_per_class": 15,
        "classes": {
            "0-10": (0, 10),
            "11-20": (11, 20),
            "21-30": (21, 30),
            "31-40": (31, 40),
            "41-50": (41, 50),
            "51-60": (51, 60),
            "61-70": (61, 70),
            "71-80": (71, 80)
        },
        "zip_name": UTKFACE_ZIP,
        "temp_dir": "utkface_temp"
    },
    "gender": {
        "kaggle_name": UTKFACE_DATASET,
        "source_folder": UTKFACE_SOURCE,
        "images_per_class": 20,
        "classes": {
            "Male": "male",
            "Female": "female"
        },
        "zip_name": UTKFACE_ZIP,
        "temp_dir": "utkface_gender_temp"
    },
    "race": {
        "kaggle_name": UTKFACE_DATASET,
        "source_folder": UTKFACE_SOURCE,
        "images_per_class": 15,
        "classes": {
            "White": 0,
            "Black": 1,
            "Asian": 2,
            "Indian": 3,
            "Latino_Hispanic": 4  # underscore for folder name
        },
        "zip_name": UTKFACE_ZIP,
        "temp_dir": "utkface_race_temp"
    }
}
# ====================== END CONFIG ======================


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_section(text):
    """Print formatted section"""
    print(f"\n{'-'*70}")
    print(f"{text}")
    print(f"{'-'*70}")


def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials exist"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def guide_kaggle_setup():
    """Guide user through Kaggle setup"""
    print("\n" + "="*70)
    print("KAGGLE API SETUP REQUIRED")
    print("="*70)
    
    if not check_kaggle_installed():
        print("\n✗ Kaggle CLI not found")
        print("\nPlease install it:")
        print("  pip install kaggle")
        print("\nThen run this script again.")
        return False
    
    if not check_kaggle_credentials():
        print("\n✗ Kaggle credentials not found")
        print("\nSetup steps:")
        print("1. Go to: https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. This downloads 'kaggle.json'")
        print("\nThen run these commands:")
        print("  mkdir -p ~/.kaggle")
        print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  chmod 600 ~/.kaggle/kaggle.json")
        print("\nThen run this script again.")
        return False
    
    return True


def download_dataset(dataset_name, zip_name):
    """Download dataset from Kaggle"""
    print(f"\nDownloading {dataset_name}...")
    print("This may take a few minutes...")
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name],
            check=True,
            capture_output=True
        )
        print(f"✓ Downloaded {zip_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading: {e.stderr.decode()}")
        return False


def extract_dataset(zip_name, temp_dir):
    """Extract downloaded dataset"""
    print(f"Extracting {zip_name}...")
    
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"✓ Extracted to {temp_dir}")
        return True
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False


def setup_emotion_dataset(config):
    """Setup emotion validation dataset"""
    print_section("Setting up EMOTION dataset")
    
    temp_dir = config['temp_dir']
    zip_name = config['zip_name']
    
    # Download and extract
    if not download_dataset(config['kaggle_name'], zip_name):
        return False
    if not extract_dataset(zip_name, temp_dir):
        return False

    source_base = os.path.join(temp_dir, config['source_folder'])
    target_base = os.path.join(VALIDATION_BASE, "emotion")
    
    if not os.path.exists(source_base):
        print(f"✗ Source folder not found: {source_base}")
        return False

    for target_name in config['classes'].values():
        os.makedirs(os.path.join(target_base, target_name), exist_ok=True)

    total_copied = 0
    for source_name, target_name in config['classes'].items():
        source_dir = os.path.join(source_base, source_name)
        target_dir = os.path.join(target_base, target_name)
        
        if not os.path.exists(source_dir):
            print(f"  ⚠ Skipping {source_name}: not found")
            continue
        
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(IMAGE_EXTENSIONS)]

        random.seed(42)
        selected = random.sample(images, min(len(images), config['images_per_class']))
        
        for img in selected:
            shutil.copy2(
                os.path.join(source_dir, img),
                os.path.join(target_dir, img)
            )
        
        total_copied += len(selected)
        print(f"  ✓ {target_name}: {len(selected)} images")
    
    print(f"✓ Emotion dataset ready: {total_copied} images")

    cleanup_files(temp_dir, zip_name)
    return True


def _find_utkface_source_dir(temp_dir):
    """Find UTKFace source directory in temp folder"""
    source_dirs = [
        os.path.join(temp_dir, UTKFACE_SOURCE),
        os.path.join(temp_dir, "UTKFace")
    ]
    return next((d for d in source_dirs if os.path.exists(d)), None)


def _copy_age_image_if_matches(filename, source_dir, target_base, config, group_counts):
    """Copy image to age group folder if matches criteria"""
    try:
        age = int(filename.split('_')[0])
        
        for group_name, (min_age, max_age) in config['classes'].items():
            if min_age <= age <= max_age:
                if group_counts[group_name] < config['images_per_class']:
                    shutil.copy2(
                        os.path.join(source_dir, filename),
                        os.path.join(target_base, group_name, filename)
                    )
                    group_counts[group_name] += 1
                break
    except (ValueError, IndexError):
        pass


def setup_age_dataset(config):
    """Setup age validation dataset"""
    print_section("Setting up AGE dataset")
    
    temp_dir = config['temp_dir']
    zip_name = config['zip_name']
    
    # Download and extract
    if not download_dataset(config['kaggle_name'], zip_name):
        return False
    if not extract_dataset(zip_name, temp_dir):
        return False

    source_dir = _find_utkface_source_dir(temp_dir)
    
    if not source_dir:
        print(f"✗ Source folder not found in {temp_dir}")
        return False

    target_base = os.path.join(VALIDATION_BASE, "age")
    for group_name in config['classes'].keys():
        os.makedirs(os.path.join(target_base, group_name), exist_ok=True)

    group_counts = dict.fromkeys(config['classes'].keys(), 0)
    images = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    random.seed(42)
    random.shuffle(images)
    
    for filename in tqdm(images, desc=PROCESSING_IMAGES_DESC):
        _copy_age_image_if_matches(filename, source_dir, target_base, config, group_counts)
    
    total_copied = sum(group_counts.values())
    for group, count in sorted(group_counts.items()):
        print(f"  ✓ {group}: {count} images")
    
    print(f"✓ Age dataset ready: {total_copied} images")
    
    # Cleanup
    cleanup_files(temp_dir, zip_name)
    return True


def setup_gender_dataset(config):
    """Setup gender validation dataset"""
    print_section("Setting up GENDER dataset")
    
    temp_dir = config['temp_dir']
    zip_name = config['zip_name']
    
    # Download and extract
    if not download_dataset(config['kaggle_name'], zip_name):
        return False
    if not extract_dataset(zip_name, temp_dir):
        return False
    
    # Find source directory
    source_dirs = [
        os.path.join(temp_dir, UTKFACE_SOURCE),
        os.path.join(temp_dir, "UTKFace")
    ]
    source_dir = next((d for d in source_dirs if os.path.exists(d)), None)
    
    if not source_dir:
        print(f"✗ Source folder not found in {temp_dir}")
        return False
    
    # Create target directories
    target_base = os.path.join(VALIDATION_BASE, "gender")
    for folder_name in config['classes'].keys():
        os.makedirs(os.path.join(target_base, folder_name), exist_ok=True)
    
    # Organize images by gender
    gender_counts = dict.fromkeys(config['classes'].keys(), 0)
    images = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    random.seed(42)
    random.shuffle(images)
    
    for filename in tqdm(images, desc=PROCESSING_IMAGES_DESC):
        try:
            gender_code = int(filename.split('_')[1])
            
            folder_name = "Male" if gender_code == 0 else "Female"
            
            if gender_counts[folder_name] < config['images_per_class']:
                shutil.copy2(
                    os.path.join(source_dir, filename),
                    os.path.join(target_base, folder_name, filename)
                )
                gender_counts[folder_name] += 1

            if all(count >= config['images_per_class'] for count in gender_counts.values()):
                break
                
        except (ValueError, IndexError):
            continue
    
    total_copied = sum(gender_counts.values())
    for gender, count in sorted(gender_counts.items()):
        print(f"  ✓ {gender}: {count} images")
    
    print(f"✓ Gender dataset ready: {total_copied} images")
    
    # Cleanup
    cleanup_files(temp_dir, zip_name)
    return True


def setup_race_dataset(config):
    """Setup race validation dataset"""
    print_section("Setting up RACE dataset")
    
    temp_dir = config['temp_dir']
    zip_name = config['zip_name']
    
    # Download and extract
    if not download_dataset(config['kaggle_name'], zip_name):
        return False
    if not extract_dataset(zip_name, temp_dir):
        return False
    
    # Find source directory
    source_dirs = [
        os.path.join(temp_dir, UTKFACE_SOURCE),
        os.path.join(temp_dir, "UTKFace")
    ]
    source_dir = next((d for d in source_dirs if os.path.exists(d)), None)
    
    if not source_dir:
        print(f"✗ Source folder not found in {temp_dir}")
        return False
    
    # Create target directories
    target_base = os.path.join(VALIDATION_BASE, "race")
    for folder_name in config['classes'].keys():
        os.makedirs(os.path.join(target_base, folder_name), exist_ok=True)
    
    # Organize images by race
    race_counts = dict.fromkeys(config['classes'].keys(), 0)
    images = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    random.seed(42)
    random.shuffle(images)

    code_to_folder = {code: name for name, code in config['classes'].items()}
    
    for filename in tqdm(images, desc=PROCESSING_IMAGES_DESC):
        try:
            race_code = int(filename.split('_')[2])
            
            if race_code in code_to_folder:
                folder_name = code_to_folder[race_code]
                
                if race_counts[folder_name] < config['images_per_class']:
                    shutil.copy2(
                        os.path.join(source_dir, filename),
                        os.path.join(target_base, folder_name, filename)
                    )
                    race_counts[folder_name] += 1

            if all(count >= config['images_per_class'] for count in race_counts.values()):
                break
                
        except (ValueError, IndexError):
            continue
    
    total_copied = sum(race_counts.values())
    for race, count in sorted(race_counts.items()):
        print(f"  ✓ {race}: {count} images")
    
    print(f"✓ Race dataset ready: {total_copied} images")
    
    # Cleanup
    cleanup_files(temp_dir, zip_name)
    return True


def cleanup_files(temp_dir, zip_file):
    """Remove temporary files"""
    print("\nCleaning up...")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"  ✓ Removed {temp_dir}/")
    
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"  ✓ Removed {zip_file}")


def verify_datasets():
    """Verify all validation datasets"""
    print_header("VERIFICATION")
    
    all_good = True
    
    for attr in ["emotion", "age", "gender", "race"]:
        attr_path = os.path.join(VALIDATION_BASE, attr)
        
        print(f"\n{attr.upper()}:")
        
        if not os.path.exists(attr_path):
            print(f"  ✗ Directory not found: {attr_path}")
            all_good = False
            continue
        
        classes = [d for d in os.listdir(attr_path) 
                  if os.path.isdir(os.path.join(attr_path, d))]
        
        if not classes:
            print("  ✗ No class folders found")
            all_good = False
            continue
        
        total_images = 0
        for class_name in sorted(classes):
            class_path = os.path.join(attr_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(IMAGE_EXTENSIONS)]
            count = len(images)
            total_images += count
            
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {class_name}: {count} images")
        
        print(f"  Total: {total_images} images")
    
    return all_good


def _setup_dataset_by_name(dataset_name, config):
    """Setup a dataset based on its name"""
    setup_functions = {
        'emotion': setup_emotion_dataset,
        'age': setup_age_dataset,
        'gender': setup_gender_dataset,
        'race': setup_race_dataset
    }
    
    setup_func = setup_functions.get(dataset_name)
    if setup_func:
        return setup_func(config)
    return False


def _print_setup_summary(results):
    """Print summary of dataset setup results"""
    print_header("SETUP COMPLETE")
    
    success_count = sum(1 for v in results.values() if v)
    print(f"\nSuccessfully setup: {success_count}/{len(results)} datasets")
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    return success_count == len(results)


def main():
    """Main entry point"""
    print_header("VALIDATION DATASETS SETUP")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Setup validation datasets for all attributes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['emotion', 'age', 'gender', 'race', 'all'],
        default=['all'],
        help='Which datasets to setup (default: all)'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification step'
    )
    
    args = parser.parse_args()
    
    # Check Kaggle setup
    if not guide_kaggle_setup():
        sys.exit(1)
    
    print("\n✓ Kaggle CLI ready")
    print("✓ Credentials found")
    
    # Determine which datasets to setup
    datasets_to_setup = ['emotion', 'age', 'gender', 'race'] if 'all' in args.datasets else args.datasets
    
    print(f"\nSetting up: {', '.join(datasets_to_setup)}")
    
    # Setup each dataset
    results = {name: _setup_dataset_by_name(name, DATASETS[name]) for name in datasets_to_setup}
    
    # Verify setup
    if not args.skip_verify:
        verify_datasets()
    
    # Summary
    all_success = _print_setup_summary(results)
    
    if all_success:
        print("\n✓ All validation datasets ready!")
        print("\nRun validation with:")
        print("  python scripts/validate_model.py")
    else:
        print("\n⚠ Some datasets failed to setup")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

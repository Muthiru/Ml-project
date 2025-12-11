"""
Comprehensive Model Validation Script
Validates DeepFace model performance on age, gender, race, and emotion predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from tqdm import tqdm
import sys
from pathlib import Path
from datetime import datetime

# ==================== CONFIGURATION ====================
VALIDATION_BASE = "data/validation"

VALIDATION_CONFIG = {
    "age": {
        "path": os.path.join(VALIDATION_BASE, "age"),
        "type": "age",
        "key": "age",
        "has_generous": True
    },
    "gender": {
        "path": os.path.join(VALIDATION_BASE, "gender"),
        "type": "gender",
        "key": "dominant_gender",
        "has_generous": False
    },
    "race": {
        "path": os.path.join(VALIDATION_BASE, "race"),
        "type": "race",
        "key": "dominant_race",
        "has_generous": False
    },
    "emotion": {
        "path": os.path.join(VALIDATION_BASE, "emotion"),
        "type": "emotion",
        "key": "dominant_emotion",
        "has_generous": False
    }
}
# ====================== END CONFIG ======================


class ValidationMetrics:
    """Stores and calculates validation metrics"""
    
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        self.results = []
        
    def add_result(self, image_name, ground_truth, prediction, is_match):
        """Add a single prediction result"""
        self.results.append({
            'image': image_name,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'match': is_match
        })
    
    def get_dataframe(self):
        """Convert results to pandas DataFrame"""
        return pd.DataFrame(self.results)
    
    def calculate_accuracy(self):
        """Calculate strict accuracy"""
        if not self.results:
            return 0.0
        matches = sum(1 for r in self.results if r['match'])
        return (matches / len(self.results)) * 100
    
    def _is_in_adjacent_bracket(self, pred_age, current_idx, all_groups):
        """Check if prediction falls in adjacent age bracket"""
        # Check previous bracket
        if current_idx > 0:
            prev_group = all_groups[current_idx - 1]
            min_age, max_age = map(int, prev_group.split('-'))
            if min_age <= pred_age <= max_age:
                return True
        
        # Check next bracket
        if current_idx < len(all_groups) - 1:
            next_group = all_groups[current_idx + 1]
            min_age, max_age = map(int, next_group.split('-'))
            if min_age <= pred_age <= max_age:
                return True
        
        return False
    
    def calculate_generous_accuracy(self):
        """Calculate generous accuracy for age (±1 bracket error allowed)"""
        if self.attribute_name != 'age' or not self.results:
            return None
        
        df = self.get_dataframe()
        all_groups = sorted(df['ground_truth'].unique(), key=lambda x: int(x.split('-')[0]))
        group_map = {group: i for i, group in enumerate(all_groups)}
        
        generous_matches = 0
        for result in self.results:
            if result['match']:
                generous_matches += 1
                continue

            try:
                current_idx = group_map[result['ground_truth']]
                pred_age = int(result['prediction'])
                
                if self._is_in_adjacent_bracket(pred_age, current_idx, all_groups):
                    generous_matches += 1
            except (ValueError, KeyError, IndexError):
                continue
        
        return (generous_matches / len(self.results)) * 100


class ModelValidator:
    """Main validator class for DeepFace model"""
    
    def __init__(self, detector_backend='opencv', output_dir='validation_results'):
        self.detector = detector_backend
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _is_match(self, prediction, ground_truth, attribute):
        """Check if prediction matches ground truth"""
        if attribute == 'age':
            return self._is_age_in_group(prediction, ground_truth)
        return str(prediction).lower() == str(ground_truth).lower()
    
    def _is_age_in_group(self, predicted_age, age_group):
        """Check if predicted age falls within age group range"""
        try:
            min_age, max_age = map(int, age_group.split('-'))
            pred_age = int(predicted_age)
            return min_age <= pred_age <= max_age
        except (ValueError, TypeError):
            return False
    
    def _analyze_single_image(self, img_path, attribute, ground_truth):
        """Analyze a single image and return result"""
        try:
            result = DeepFace.analyze(
                img_path=img_path,
                actions=[attribute],
                enforce_detection=False,
                detector_backend=self.detector,
                silent=True
            )
            
            if not result or len(result) == 0:
                return None

            config = VALIDATION_CONFIG[attribute]
            prediction = result[0].get(config['key'])
            
            if prediction is None:
                return None
            
            is_match = self._is_match(prediction, ground_truth, attribute)
            
            return {
                'image': os.path.basename(img_path),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'match': is_match
            }
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(img_path)}: {str(e)[:50]}")
            return None
    
    def validate_attribute(self, attribute):
        """Validate model on specific attribute"""
        print(f"\n{'='*70}")
        print(f"Validating {attribute.upper()}")
        print(f"{'='*70}")
        
        config = VALIDATION_CONFIG[attribute]
        data_path = config['path']
        
        if not os.path.exists(data_path):
            print(f"✗ Validation data not found at: {data_path}")
            print("  Please run setup script to create validation dataset first.")
            return None
        
        classes = [d for d in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, d))]
        
        if not classes:
            print(f"✗ No class folders found in {data_path}")
            return None
        
        print(f"Found {len(classes)} classes: {', '.join(sorted(classes))}")
        
        metrics = ValidationMetrics(attribute)

        for class_name in classes:
            class_dir = os.path.join(data_path, class_name)
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\nProcessing {class_name}: {len(images)} images")
            
            for img_file in tqdm(images, desc=f"  {class_name}", leave=False):
                img_path = os.path.join(class_dir, img_file)
                result = self._analyze_single_image(img_path, attribute, class_name)
                
                if result:
                    metrics.add_result(
                        result['image'],
                        result['ground_truth'],
                        result['prediction'],
                        result['match']
                    )

        df = metrics.get_dataframe()
        
        if df.empty:
            print(f"\n✗ No valid results for {attribute}")
            return None
        
        print(f"\n{'─'*70}")
        print(f"Results Summary for {attribute.upper()}")
        print(f"{'─'*70}")
        print(f"Total images processed: {len(df)}")
        print(f"Successful predictions: {len(df[df['prediction'].notna()])}")
        
        strict_acc = metrics.calculate_accuracy()
        print(f"Strict Accuracy: {strict_acc:.2f}%")
        
        generous_acc = None
        if config['has_generous']:
            generous_acc = metrics.calculate_generous_accuracy()
            print(f"Generous Accuracy (±1 bracket): {generous_acc:.2f}%")

        self._save_results(df, attribute)
        self._plot_confusion_matrix(df, attribute)
        
        return {
            'attribute': attribute,
            'strict_accuracy': strict_acc,
            'generous_accuracy': generous_acc,
            'total_images': len(df),
            'dataframe': df
        }
    
    def _save_results(self, df, attribute):
        """Save results to CSV file"""
        filename = f"{attribute}_results_{self.detector}_{self.timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to: {filepath}")
    
    def _plot_confusion_matrix(self, df, attribute):
        """Generate and save confusion matrix"""
        if df.empty:
            return
        
        plt.figure(figsize=(12, 10))

        try:
            if attribute == 'age':
                key_func = lambda x: int(x.split('-')[0])
            else:
                key_func = str
            
            all_labels = sorted(
                set(df['ground_truth'].unique()) | set(df['prediction'].dropna().unique()),
                key=key_func
            )
            
            matrix = pd.crosstab(
                df['ground_truth'],
                df['prediction'],
                rownames=['Actual'],
                colnames=['Predicted']
            )
            matrix = matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)
            
        except Exception:
            # Fallback to simple crosstab
            matrix = pd.crosstab(
                df['ground_truth'],
                df['prediction'],
                rownames=['Actual'],
                colnames=['Predicted']
            )
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix: {attribute.capitalize()} Prediction\n(Detector: {self.detector})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        filename = f"{attribute}_confusion_matrix_{self.detector}_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {filepath}")
    
    def validate_all(self, attributes=None):
        """Validate all specified attributes"""
        if attributes is None:
            attributes = list(VALIDATION_CONFIG.keys())
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE MODEL VALIDATION")
        print(f"Detector: {self.detector}")
        print(f"Attributes: {', '.join(attributes)}")
        print(f"{'='*70}")
        
        results = {}
        
        for attr in attributes:
            if attr not in VALIDATION_CONFIG:
                print(f"\n⚠ Warning: Unknown attribute '{attr}', skipping...")
                continue
            
            result = self.validate_attribute(attr)
            if result:
                results[attr] = result
        
        # Print final summary
        self._print_final_summary(results)
        
        # Save summary report
        self._save_summary_report(results)
        
        return results
    
    def _print_final_summary(self, results):
        """Print final validation summary"""
        print(f"\n{'='*70}")
        print("FINAL VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Detector: {self.detector}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}")
        
        if not results:
            print("No validation results available.")
            print("Please ensure validation datasets are set up correctly.")
        else:
            for attr, result in results.items():
                print(f"\n{attr.upper()}:")
                print(f"  Images processed: {result['total_images']}")
                print(f"  Strict accuracy: {result['strict_accuracy']:.2f}%")
                if result['generous_accuracy'] is not None:
                    print(f"  Generous accuracy: {result['generous_accuracy']:.2f}%")
        
        print(f"\n{'='*70}")
    
    def _save_summary_report(self, results):
        """Save summary report to file"""
        filename = f"validation_summary_{self.detector}_{self.timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DEEPFACE MODEL VALIDATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detector Backend: {self.detector}\n")
            f.write("="*70 + "\n\n")
            
            for attr, result in results.items():
                f.write(f"{attr.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"Total images: {result['total_images']}\n")
                f.write(f"Strict accuracy: {result['strict_accuracy']:.2f}%\n")
                if result['generous_accuracy'] is not None:
                    f.write(f"Generous accuracy: {result['generous_accuracy']:.2f}%\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
        
        print(f"\n✓ Summary report saved to: {filepath}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate DeepFace model on age, gender, race, and emotion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all attributes
  python validate_model.py
  
  # Validate specific attributes
  python validate_model.py --attributes age emotion
  
  # Use different detector
  python validate_model.py --detector retinaface
  
  # Specify output directory
  python validate_model.py --output my_results/
        """
    )
    
    parser.add_argument(
        '--attributes',
        nargs='+',
        choices=['age', 'gender', 'race', 'emotion'],
        default=None,
        help='Specific attributes to validate (default: all)'
    )
    
    parser.add_argument(
        '--detector',
        default='opencv',
        choices=['opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib'],
        help='Face detector backend (default: opencv)'
    )
    
    parser.add_argument(
        '--output',
        default='validation_results',
        help='Output directory for results (default: validation_results)'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = ModelValidator(
        detector_backend=args.detector,
        output_dir=args.output
    )
    
    # Run validation
    attributes = args.attributes if args.attributes else list(VALIDATION_CONFIG.keys())
    results = validator.validate_all(attributes)
    
    if results:
        print("\n✓ Validation complete!")
        print(f"  Results saved to: {args.output}/")
    else:
        print("\n✗ Validation failed or no data available")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

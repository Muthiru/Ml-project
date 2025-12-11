"""
Single Image Face Recognition
Analyzes age, gender, race, and emotion from a single image file
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace
import cv2
import sys


def analyze_image(image_path):
    """Analyze face attributes from image file"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print("="*60)
    print("Face Recognition Analysis")
    print("="*60)
    print(f"Analyzing: {os.path.basename(image_path)}")
    print("Processing...\n")
    
    try:
        result = DeepFace.analyze(
            image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True
        )
        
        if result and len(result) > 0:
            face = result[0]
            
            print("="*60)
            print("RESULTS:")
            print("="*60)
            print(f"Age:      {int(face['age'])} years")
            print(f"Gender:   {face['dominant_gender']} ({face['gender'][face['dominant_gender']]:.1f}%)")
            print(f"Race:     {face['dominant_race']} ({face['race'][face['dominant_race']]:.1f}%)")
            print(f"Emotion:  {face['dominant_emotion']} ({face['emotion'][face['dominant_emotion']]:.1f}%)")
            
            print("\nRace Breakdown:")
            for race_type, probability in sorted(face['race'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {race_type}: {probability:.1f}%")
            
            print("="*60)
            
            img = cv2.imread(image_path)
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            y_text = y - 10
            cv2.putText(img, f"Age: {int(face['age'])}", (x, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"{face['dominant_race']}", (x, y_text - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Face Recognition Result', img)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("Error: No face detected in image")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("="*60)
        print("Single Image Face Recognition")
        print("="*60)
        print("\nUsage:")
        print("  python analyze_image.py <image_path>")
        print("\nExample:")
        print("  python analyze_image.py photo.jpg")
        print("  python analyze_image.py /path/to/image.png")
        print("="*60)
        sys.exit(1)
    
    image_path = sys.argv[1]
    analyze_image(image_path)
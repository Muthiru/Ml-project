"""
Real-time Face Recognition - Detailed Mode
Live webcam feed with age, gender, race, and emotion detection
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from deepface import DeepFace
import time
import sys


def process_frame_prediction(frame, last_prediction_time, prediction_interval):
    """Process frame and return predictions"""
    current_time = time.time()
    predicted_age = None
    predicted_gender = None
    predicted_race = None
    predicted_emotion = None
    face_detected = False
    
    if current_time - last_prediction_time > prediction_interval:
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=True,
                silent=True
            )
            
            if result and len(result) > 0:
                face = result[0]
                predicted_age = int(face['age'])
                predicted_gender = face['dominant_gender']
                predicted_race = face['dominant_race']
                predicted_emotion = face['dominant_emotion']
                face_detected = True
                
        except ValueError:
            face_detected = False
        except Exception:
            face_detected = False
    
    return predicted_age, predicted_gender, predicted_race, predicted_emotion, face_detected, current_time


def draw_predictions_on_frame(frame, face_detected, predicted_age, predicted_gender, 
                              predicted_race, predicted_emotion):
    """Draw prediction overlays on video frame"""
    y_offset = 30
    line_height = 35
    
    cv2.putText(frame, "Face Recognition System", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += line_height
    
    if face_detected and predicted_age:
        cv2.putText(frame, f"Age: {predicted_age} years", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Gender: {predicted_gender}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Race: {predicted_race}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "Face: DETECTED", (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += line_height
        
        cv2.putText(frame, "Position your face in frame", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(frame, "Face: NOT DETECTED", (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press 'q' to quit | 's' to save", 
               (10, frame.shape[0] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    """Main application entry point"""
    print("="*60)
    print("Face Recognition System - Live Webcam (Detailed Mode)")
    print("="*60)
    print("Loading AI models...")
    print("Please wait (first run may take 1-2 minutes)...")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("="*60 + "\n")

    _run_live_webcam()


def _handle_prediction_result(state, new_age, new_gender, new_race, new_emotion, new_face_detected, current_time):
    """Update the shared state based on a new prediction result."""
    if new_face_detected:
        state['predicted_age'] = new_age
        state['predicted_gender'] = new_gender
        state['predicted_race'] = new_race
        state['predicted_emotion'] = new_emotion
        state['face_detected'] = True
        state['last_prediction_time'] = current_time
        state['consecutive_no_face'] = 0

        if state['first_prediction']:
            print("\n" + "="*60)
            print("First prediction complete!")
            print(f"Age:      {state['predicted_age']} years")
            print(f"Gender:   {state['predicted_gender']}")
            print(f"Race:     {state['predicted_race']}")
            print(f"Emotion:  {state['predicted_emotion']}")
            print("="*60 + "\n")
            state['first_prediction'] = False
        else:
            print(f"Updated: Age={state['predicted_age']}, Gender={state['predicted_gender']}, Race={state['predicted_race']}, Emotion={state['predicted_emotion']}")
    else:
        # No face detected; count consecutive failures only if enough time has passed
        if current_time - state['last_prediction_time'] > state['prediction_interval']:
            state['consecutive_no_face'] += 1
            state['last_prediction_time'] = current_time

            if state['consecutive_no_face'] == 1:
                print("\nNo face detected - please position your face in frame")

            if state['consecutive_no_face'] >= 2:
                state['predicted_age'] = None
                state['predicted_gender'] = None
                state['predicted_race'] = None
                state['predicted_emotion'] = None
                state['face_detected'] = False
                print("Predictions cleared - no face visible")


def _run_live_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = {
        'last_prediction_time': 0,
        'prediction_interval': 2.0,
        'predicted_age': None,
        'predicted_gender': None,
        'predicted_race': None,
        'predicted_emotion': None,
        'face_detected': False,
        'screenshot_count': 0,
        'consecutive_no_face': 0,
        'first_prediction': True
    }

    print("Webcam started!")
    print("Show your face to the camera...")
    print("Waiting for first prediction (may take 10-15 seconds)...\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        new_age, new_gender, new_race, new_emotion, new_face_detected, current_time = \
            process_frame_prediction(frame, state['last_prediction_time'], state['prediction_interval'])

        _handle_prediction_result(state, new_age, new_gender, new_race, new_emotion, new_face_detected, current_time)

        draw_predictions_on_frame(frame, state['face_detected'], state['predicted_age'], 
                                 state['predicted_gender'], state['predicted_race'], state['predicted_emotion'])

        cv2.imshow('Face Recognition System - Live Feed', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nClosing webcam...")
            break
        elif key == ord('s') and state['face_detected']:
            state['screenshot_count'] += 1
            filename = f"screenshot_{state['screenshot_count']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")
    print("\n" + "="*60)
    print("Thank you for using Face Recognition System")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        cv2.destroyAllWindows()
        sys.exit(1)
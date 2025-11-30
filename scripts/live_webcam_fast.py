"""
Real-time Face Recognition - Fast Mode
Optimized for performance with face detection validation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from deepface import DeepFace
import time
import sys

# Common UI literals
LOADING = "Loading..."
WAITING = "Please wait..."


def _safe_analyze(frame):
    """Run DeepFace.analyze safely and normalize return values.

    Returns a tuple (status, result) where status is one of:
      - 'ok' -> result is a list of face dicts
      - 'no_face' -> result is an empty list
      - 'error' -> result is error message string
    """
    try:
        result = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True,
            silent=True
        )
        if not result:
            return 'no_face', []
        if isinstance(result, dict):
            result = [result]
        return 'ok', result
    except ValueError:
        # no face detected when enforce_detection=True
        return 'no_face', []
    except Exception as e:
        return 'error', str(e)


def _draw_overlay(frame, age, gender, race, emotion, face_currently_detected, processing, time_until_next):
    """Draw UI overlay on frame and show it. Returns key pressed (int) or -1."""
    if face_currently_detected:
        status_text = f"Next update: {time_until_next:.1f}s | Face: DETECTED"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = f"Next update: {time_until_next:.1f}s | Face: NOT DETECTED"
        status_color = (0, 0, 255)  # Red

    cv2.putText(frame, "Face Recognition System", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if processing:
        cv2.putText(frame, "PROCESSING...", (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y_offset = 70
    line_height = 40

    # Color based on face detection
    if face_currently_detected:
        text_color = (0, 255, 0)  # Green when face detected
    elif processing:
        text_color = (0, 255, 255)  # Yellow when processing
    else:
        text_color = (0, 0, 255)  # Red when no face

    cv2.putText(frame, f"Age: {age}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += line_height

    cv2.putText(frame, f"Gender: {gender}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += line_height

    cv2.putText(frame, f"Race: {race}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += line_height

    cv2.putText(frame, f"Emotion: {emotion}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Status with face detection indicator
    cv2.putText(frame, status_text, (10, frame.shape[0] - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, "Press 'q' to quit | 's' to save", 
               (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Face Recognition - Live Feed (Fast)', frame)
    key = cv2.waitKey(1) & 0xFF
    return key


def _update_state_for_frame(state, frame, frame_count, current_time, prediction_interval):
    """Update mutable state for a frame: handles when to call analyzer and update display texts."""
    time_since_last = current_time - state['last_prediction']

    if not state['first_prediction_done'] and frame_count < 30:
        state['age'] = "Loading models..."
        state['gender'] = WAITING
        state['race'] = "First run takes longer"
        state['emotion'] = "Almost ready..."
        return

    if state['processing']:
        state['age'] = "Processing..."
        state['gender'] = "Analyzing face..."
        state['race'] = WAITING
        state['emotion'] = "Computing..."
        return

    if time_since_last <= prediction_interval:
        # nothing to do this frame
        return

    # Time to attempt a prediction
    state['processing'] = True

    status, result = _safe_analyze(frame)

    if status == 'ok' and result:
        face = result[0]
        state['age'] = f"{int(face.get('age', 0))} years"
        state['gender'] = face.get('dominant_gender', '')
        state['race'] = face.get('dominant_race', '')
        state['emotion'] = face.get('dominant_emotion', '')

        state['face_currently_detected'] = True
        state['frames_without_face'] = 0

        if not state['first_prediction_done']:
            print("\n" + "="*60)
            print("FIRST PREDICTION COMPLETE!")
            print("="*60)
            print(f"Age:      {state['age']}")
            print(f"Gender:   {state['gender']}")
            print(f"Race:     {state['race']}")
            print(f"Emotion:  {state['emotion']}")
            print("="*60)
            print("\nNow updating every 3 seconds...")
            print("Press 'q' to quit | 's' to save screenshot\n")
            state['first_prediction_done'] = True
        else:
            print(f"\nUpdated: Age={state['age']}, Gender={state['gender']}, Race={state['race']}, Emotion={state['emotion']}")

        state['last_prediction'] = current_time

    elif status == 'no_face':
        state['face_currently_detected'] = False
        state['frames_without_face'] += 1
        if state['frames_without_face'] == 1:
            print("\nNo face detected - please position your face in frame")

    else:  # error
        state['face_currently_detected'] = False
        state['frames_without_face'] += 1
        print(f"Error: {result}")

    state['processing'] = False

    # Clear predictions if no face detected for 2 consecutive checks
    if state['frames_without_face'] >= 2 and state['first_prediction_done']:
        state['age'] = "No face detected"
        state['gender'] = "Move closer"
        state['race'] = "Position face in frame"
        state['emotion'] = "Show your face"
        print("\nPredictions cleared - no face visible")


def main():
    """Main application entry point"""
    print("="*60)
    print("Face Recognition System - Live Webcam (Fast Mode)")
    print("="*60)
    print("Initializing camera and AI models...")
    print("First prediction may take 15-30 seconds...")
    print("Please be patient while models load...")
    print("="*60 + "\n")

    _run_live_webcam()


def _run_live_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = {
        'last_prediction': time.time(),
        'prediction_interval': 3.0,
        'age': "Initializing...",
        'gender': LOADING,
        'race': LOADING,
        'emotion': LOADING,
        'screenshot_count': 0,
        'first_prediction_done': False,
        'processing': False,
        'face_currently_detected': False,
        'frames_without_face': 0
    }

    print("Camera started!")
    print("Position your face in the frame...")
    print("\nWaiting for AI models to load...\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        _update_state_for_frame(state, frame, frame_count, current_time, state['prediction_interval'])

        time_since_last = current_time - state['last_prediction']
        time_until_next = max(0, state['prediction_interval'] - time_since_last)

        key = _draw_overlay(frame, state['age'], state['gender'], state['race'], state['emotion'],
                            state['face_currently_detected'], state['processing'], time_until_next)

        if key == ord('q'):
            break
        elif key == ord('s'):
            state['screenshot_count'] += 1
            filename = f"screenshot_{state['screenshot_count']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\nScreenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed!")
    print("Thank you for using Face Recognition System!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
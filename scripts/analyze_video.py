"""
Video File Face Recognition
Processes video files and outputs annotated videos with age predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from deepface import DeepFace
import sys
import argparse
from pathlib import Path


def _open_capture(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    return cap


def _create_writer(output_path, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        return None
    return out


def _get_predictions_for_frame(frame):
    """Run DeepFace.analyze, return list of predictions or empty list on failure"""
    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False,
            silent=True
        )
        if not results:
            return []
        if isinstance(results, dict):
            results = [results]

        preds = []
        for result in results:
            preds.append({
                'age': int(result.get('age', 0)),
                'gender': result.get('dominant_gender', ''),
                'race': result.get('dominant_race', ''),
                'emotion': result.get('dominant_emotion', ''),
                'region': result.get('region', {}),
                'confidence': result.get('age', 0)
            })
        return preds
    except Exception:
        return []


def _annotate_frame(frame, predictions):
    """Draw rectangles and labels for each prediction on the frame."""
    for pred in predictions:
        region = pred.get('region') or {}
        if not region:
            continue
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        age_text = f"Age: {pred.get('age', '?')}"
        gender_text = f"{pred.get('gender', '')}"
        race_text = f"{pred.get('race', '')}"

        text_y = max(10, y - 10)
        cv2.rectangle(frame, (x, text_y - 25), (x + 200, text_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, age_text, (x + 5, text_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"{gender_text}, {race_text}", (x + 5, text_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def _print_video_info(input_path, output_path, frame_width, frame_height, fps, total_frames):
    print("=" * 60)
    print("Video Processing Started")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("=" * 60)


def _process_frames(cap_handle, out_handle, prediction_interval, total_frames):
    """Process frames from cap_handle, annotate and write to out_handle."""
    frame_count = 0
    last_predictions = {}

    # protect against invalid interval
    if prediction_interval <= 0:
        prediction_interval = 30

    try:
        while True:
            ret, frame = cap_handle.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % prediction_interval == 1:
                preds = _get_predictions_for_frame(frame)
                if preds:
                    last_predictions = dict(enumerate(preds))
                    print(f"Frame {frame_count}/{total_frames}: Detected {len(preds)} face(s)")

            if last_predictions:
                _annotate_frame(frame, last_predictions.values())

            out_handle.write(frame)

            chunk = max(1, total_frames // 10)
            if frame_count % chunk == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")


def process_video(input_path, output_path=None, prediction_interval=30):
    """Process video file and add face detection with predictions"""
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_processed{input_file.suffix}")
    
    cap = _open_capture(input_path)
    if cap is None:
        print(f"Error: Could not open video file: {input_path}")
        return False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = _create_writer(output_path, fps, frame_width, frame_height)
    if out is None:
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    _print_video_info(input_path, output_path, frame_width, frame_height, fps, total_frames)
    
    
    try:
        _process_frames(cap, out, prediction_interval, total_frames)

    finally:
        cap.release()
        out.release()

        print("=" * 60)
        print("Processing Complete")
        print(f"Output saved to: {output_path}")
        print("=" * 60)
    
    return True


def play_video(video_path):
    """Play video file using OpenCV"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    print("Playing video. Press 'q' to quit, SPACE to pause/resume")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("Video ended")
                break
            
            cv2.imshow('Video Playback', frame)
        
        key = cv2.waitKey(25) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Process video files for face detection and age prediction'
    )
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file (optional)')
    parser.add_argument('-i', '--interval', type=int, default=30,
                       help='Frame interval for predictions (default: 30)')
    parser.add_argument('-p', '--play', action='store_true',
                       help='Play output video after processing')
    parser.add_argument('--play-only', help='Only play the specified video file')
    
    args = parser.parse_args()
    
    if args.play_only:
        play_video(args.play_only)
        return
    
    success = process_video(
        args.input,
        args.output,
        prediction_interval=args.interval
    )
    
    if success and args.play:
        output_path = args.output if args.output else str(
            Path(args.input).parent / f"{Path(args.input).stem}_processed{Path(args.input).suffix}"
        )
        play_video(output_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python analyze_video.py input.mp4")
        print("  python analyze_video.py input.mp4 -o output.mp4")
        print("  python analyze_video.py input.mp4 --play")
        print("  python analyze_video.py --play-only output.mp4")
        sys.exit(1)
    
    main()
"""
Single Image Face Recognition
Analyzes age, gender, race, and emotion from a single image file
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace
import cv2
import sys
import subprocess

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QGroupBox, QProgressBar, QScrollArea,
                             QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from deepface import DeepFace

# Constants
STYLE_RESULT_SUCCESS = "font-size: 14px; padding: 10px; color: #0a0;"
STYLE_FONT_14PX = "font-size: 14px;"
VIDEO_WINDOW_TITLE = 'Video Processing - Press Q to stop'
MAX_FRAME_DIMENSION = 1920
PREDICTION_INTERVAL = 5
FRAMES_BEFORE_CLEAR = 30
DETECTORS_ORDER = ['retinaface', 'mtcnn', 'ssd', 'opencv']


class PredictionThread(QThread):
    """Background thread for image prediction"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
    
    def run(self):
        try:
            results = DeepFace.analyze(
                self.image_path,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=True,
                detector_backend='retinaface',
                silent=True
            )
            if results:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class VideoProcessingThread(QThread):
    """Background thread for video processing with live preview and better face detection"""
    progress = pyqtSignal(int, str)
    frame_ready = pyqtSignal(object)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, output_path=None, prediction_interval=PREDICTION_INTERVAL, show_preview=True):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.prediction_interval = prediction_interval
        self.show_preview = show_preview
        self.is_cancelled = False
    
    def _open_video_capture(self):
        """Open and validate video capture"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.input_path}")
        return cap
    
    def _calculate_resize_params(self, width, height):
        """Calculate resize parameters if frame is too large"""
        if width <= MAX_FRAME_DIMENSION and height <= MAX_FRAME_DIMENSION:
            return None, None, None, False
        
        if width > height:
            new_width = MAX_FRAME_DIMENSION
            new_height = int(height * (MAX_FRAME_DIMENSION / width))
        else:
            new_height = MAX_FRAME_DIMENSION
            new_width = int(width * (MAX_FRAME_DIMENSION / height))
        
        scale_x = width / new_width
        scale_y = height / new_height
        
        return new_width, new_height, (scale_x, scale_y), True
    
    def _try_detect_with_backend(self, frame, detector):
        """Try to detect faces with a specific backend"""
        try:
            return DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                detector_backend=detector,
                silent=True
            )
        except Exception:
            return None
    
    def _scale_region_coordinates(self, results, scale_factors):
        """Scale region coordinates back to original size"""
        scale_x, scale_y = scale_factors
        for result in results:
            region = result['region']
            result['region']['x'] = int(region['x'] * scale_x)
            result['region']['y'] = int(region['y'] * scale_y)
            result['region']['w'] = int(region['w'] * scale_x)
            result['region']['h'] = int(region['h'] * scale_y)
    
    def _analyze_frame(self, frame):
        """Analyze frame with multi-detector fallback for better face detection"""
        try:
            height, width = frame.shape[:2]
            new_width, new_height, scale_factors, needs_resize = self._calculate_resize_params(width, height)
            
            processed_frame = frame
            if needs_resize:
                processed_frame = cv2.resize(frame, (new_width, new_height))
            
            results = self._try_multiple_detectors(processed_frame)
            
            if not results:
                return {}
            
            if needs_resize and scale_factors:
                self._scale_region_coordinates(results, scale_factors)
            
            return self._convert_to_predictions(results)
            
        except Exception:
            return {}
    
    def _try_multiple_detectors(self, frame):
        """Try multiple face detectors until one succeeds"""
        for detector in DETECTORS_ORDER:
            results = self._try_detect_with_backend(frame, detector)
            if results:
                return results
        return None
    
    def _convert_to_predictions(self, results):
        """Convert DeepFace results to predictions dictionary"""
        predictions = {}
        for idx, result in enumerate(results):
            predictions[idx] = {
                'age': int(result['age']),
                'gender': result['dominant_gender'],
                'race': result['dominant_race'],
                'emotion': result['dominant_emotion'],
                'region': result['region']
            }
        return predictions
    
    def _draw_face_box(self, frame, x, y, w, h, idx):
        """Draw rectangle and number circle for a face"""
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.circle(frame, (x + 25, y + 25), 25, (0, 255, 0), -1)
        cv2.putText(frame, f'{idx+1}', (x + 12, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    def _draw_face_info(self, frame, x, y, h, pred):
        """Draw information labels for a face"""
        cv2.putText(frame, f"Age: {pred['age']}", (x, y-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{pred['gender']}", (x, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"{pred['emotion']}", (x, y+h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"{pred['race']}", (x, y+h+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    
    def _draw_predictions(self, frame, predictions):
        """Draw prediction overlays on frame"""
        for idx, pred in predictions.items():
            region = pred['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            self._draw_face_box(frame, x, y, w, h, idx)
            self._draw_face_info(frame, x, y, h, pred)
        
        return frame
    
    def _update_predictions(self, frame_count, last_predictions, frames_since_detection):
        """Update predictions if it's time to analyze"""
        if frame_count % self.prediction_interval != 1:
            return last_predictions, frames_since_detection
        
        new_predictions = self._analyze_frame(self.current_frame)
        
        if new_predictions:
            return new_predictions, 0
        
        frames_since_detection += 1
        if frames_since_detection > FRAMES_BEFORE_CLEAR:
            return {}, frames_since_detection
        
        return last_predictions, frames_since_detection
    
    def _process_single_frame(self, frame, last_predictions):
        """Process a single frame with predictions"""
        processed_frame = frame.copy()
        if last_predictions:
            processed_frame = self._draw_predictions(processed_frame, last_predictions)
        
        if self.show_preview:
            self.frame_ready.emit(processed_frame)
        
        return processed_frame
    
    def _emit_progress(self, frame_count, total_frames, face_count):
        """Emit progress update"""
        progress_percent = int((frame_count / total_frames) * 100)
        self.progress.emit(progress_percent, 
                         f"Frame {frame_count}/{total_frames} - {face_count} face(s) detected")
    
    def _control_playback_speed(self, fps):
        """Control playback speed for preview"""
        if self.show_preview:
            import time
            time.sleep(1.0 / fps if fps > 0 else 0.03)
    
    def run(self):
        """Process video file with live preview"""
        cap = None
        out = None
        
        try:
            cap = self._open_video_capture()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            out = self._create_video_writer(cap) if self.output_path else None
            
            self._process_video_frames(cap, out, total_frames, fps)
            
            if not self.is_cancelled:
                result_msg = self.output_path if self.output_path else "Processing complete"
                self.finished.emit(result_msg)
        
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
        finally:
            self._cleanup(cap, out)
    
    def _create_video_writer(self, cap):
        """Create video writer for output"""
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
    
    def _process_video_frames(self, cap, out, total_frames, fps):
        """Process all video frames"""
        frame_count = 0
        last_predictions = {}
        frames_since_detection = 0
        
        while not self.is_cancelled:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.current_frame = frame
            
            last_predictions, frames_since_detection = self._update_predictions(
                frame_count, last_predictions, frames_since_detection
            )
            
            processed_frame = self._process_single_frame(frame, last_predictions)
            
            if out:
                out.write(processed_frame)
            
            self._emit_progress(frame_count, total_frames, len(last_predictions))
            self._control_playback_speed(fps)
    
    def _cleanup(self, cap, out):
        """Cleanup video resources"""
        if cap:
            cap.release()
        if out:
            out.release()
    
    def cancel(self):
        """Cancel video processing"""
        self.is_cancelled = True


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_image = None
        self.prediction_thread = None
        self.video_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Face Recognition System')
        self.setGeometry(100, 100, 1000, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        
        main_layout = QVBoxLayout()
        content_widget.setLayout(main_layout)
        
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)
        central_layout.addWidget(scroll)
        
        self._add_title_section(main_layout)
        self._add_image_preview(main_layout)
        self._add_progress_bar(main_layout)
        self._add_buttons(main_layout)
        self._add_results_section(main_layout)
        
        self.statusBar().showMessage('Ready - AI models will auto-load on first use')
    
    def _add_title_section(self, layout):
        """Add title and subtitle to layout"""
        title = QLabel('Face Recognition System')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; font-weight: bold; padding: 20px;')
        layout.addWidget(title)
        
        model_info = QLabel('AI-Powered Age, Gender, Race & Emotion Detection')
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet(STYLE_RESULT_SUCCESS)
        layout.addWidget(model_info)
    
    def _add_image_preview(self, layout):
        """Add image preview section"""
        image_group = QGroupBox('Image/Video Preview')
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet('border: 2px solid #ccc; background-color: #f0f0f0;')
        self.image_label.setText('No image loaded')
        image_layout.addWidget(self.image_label)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
    
    def _add_progress_bar(self, layout):
        """Add progress bar"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    
    def _add_buttons(self, layout):
        """Add control buttons"""
        button_layout = QHBoxLayout()
        
        buttons = [
            ('Load Image', self.load_image),
            ('Load Video', self.load_video),
            ('Analyze Face', self.predict_age),
            ('Start Webcam', self.start_webcam),
            ('Clear', self.clear_image)
        ]
        
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setStyleSheet(STYLE_FONT_14PX)
            button_layout.addWidget(btn)
            
            if text == 'Analyze Face':
                self.predict_btn = btn
                self.predict_btn.setEnabled(False)
            elif text == 'Load Video':
                self.load_video_btn = btn
        
        layout.addLayout(button_layout)
    
    def _add_results_section(self, layout):
        """Add results display section"""
        results_group = QGroupBox('Results')
        results_layout = QVBoxLayout()
        
        self.result_label = QTextEdit()
        self.result_label.setReadOnly(True)
        self.result_label.setStyleSheet(STYLE_FONT_14PX + ' padding: 10px;')
        self.result_label.setPlainText('No prediction yet')
        self.result_label.setMinimumHeight(150)
        results_layout.addWidget(self.result_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
    
    def load_image(self):
        """Load an image file"""
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Image',
            '',
            'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)'
        )
        
        if image_path:
            try:
                self.current_image_path = image_path
                self.current_image = cv2.imread(image_path)
                
                if self.current_image is None:
                    raise ValueError('Could not load image')
                
                self.display_image(self.current_image)
                self.predict_btn.setEnabled(True)
                self.statusBar().showMessage(f'Loaded: {os.path.basename(image_path)}')
                self.result_label.setPlainText('Image loaded. Click "Analyze Face" to begin.')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load image:\n{str(e)}')
    
    def load_video(self):
        """Load and process video file with live preview"""
        video_path = self._get_video_file_path()
        if not video_path:
            return
        
        output_path = self._ask_save_preference(video_path)
        if output_path is False:  # User cancelled
            return
        
        self.process_video(video_path, output_path)
    
    def _get_video_file_path(self):
        """Get video file path from user"""
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Video',
            '',
            'Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)'
        )
        return video_path
    
    def _ask_save_preference(self, video_path):
        """Ask user if they want to save the processed video"""
        reply = QMessageBox.question(
            self,
            'Process Video',
            'Do you want to save the processed video?\n\n'
            'Yes = Process and save\n'
            'No = Just preview without saving',
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Cancel:
            return False
        
        if reply == QMessageBox.No:
            return None
        
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Processed Video',
            video_path.rsplit('.', 1)[0] + '_processed.mp4',
            'Video Files (*.mp4)'
        )
        
        return output_path if output_path else False
    
    def process_video(self, input_path, output_path=None):
        """Process video with live preview"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.load_video_btn.setEnabled(False)
        self.statusBar().showMessage('Processing video with live preview...')
        
        cv2.namedWindow(VIDEO_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VIDEO_WINDOW_TITLE, 960, 720)
        
        self.video_thread = VideoProcessingThread(input_path, output_path, show_preview=True)
        self.video_thread.progress.connect(self.on_video_progress)
        self.video_thread.frame_ready.connect(self.on_video_frame)
        self.video_thread.finished.connect(self.on_video_complete)
        self.video_thread.error.connect(self.on_video_error)
        self.video_thread.start()
    
    def on_video_frame(self, frame):
        """Display video frame in OpenCV window"""
        try:
            cv2.imshow(VIDEO_WINDOW_TITLE, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.video_thread:
                    self.video_thread.cancel()
                cv2.destroyWindow(VIDEO_WINDOW_TITLE)
        except Exception:
            pass
    
    def on_video_progress(self, percent, message):
        """Update progress bar"""
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(message)
    
    def on_video_complete(self, message):
        """Handle video processing completion"""
        self.progress_bar.setVisible(False)
        self.load_video_btn.setEnabled(True)
        
        try:
            cv2.destroyWindow(VIDEO_WINDOW_TITLE)
        except Exception:
            pass
        
        self._show_completion_message(message)
    
    def _show_completion_message(self, message):
        """Show appropriate completion message"""
        if "Processing complete" in message:
            self.statusBar().showMessage('Video processing complete (preview only)')
            QMessageBox.information(
                self,
                'Processing Complete',
                'Video processing complete!\n\nPreview finished.'
            )
        else:
            self.statusBar().showMessage('Video saved successfully')
            QMessageBox.information(
                self,
                'Processing Complete',
                f'Video saved to:\n{message}\n\nProcessing complete!'
            )
    
    def on_video_error(self, error_msg):
        """Handle video processing error"""
        self.progress_bar.setVisible(False)
        self.load_video_btn.setEnabled(True)
        
        try:
            cv2.destroyWindow(VIDEO_WINDOW_TITLE)
        except Exception:
            pass
        
        QMessageBox.critical(self, 'Error', f'Video processing failed:\n{error_msg}')
        self.statusBar().showMessage('Video processing failed')
    
    def display_image(self, image):
        """Display image in GUI"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def predict_age(self):
        """Predict using Face Recognition"""
        if not self.current_image_path:
            QMessageBox.warning(self, 'Warning', 'Please load an image first!')
            return
        
        self.statusBar().showMessage('Analyzing... This may take a few seconds')
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.predict_btn.setEnabled(False)
        
        self.prediction_thread = PredictionThread(self.current_image_path)
        self.prediction_thread.finished.connect(self.on_prediction_complete)
        self.prediction_thread.error.connect(self.on_prediction_error)
        self.prediction_thread.start()
    
    def on_prediction_complete(self, results):
        """Handle prediction results for multiple faces"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        result_image = self._draw_faces_on_image(results)
        result_text = self._format_results_text(results)
        
        self.display_image(result_image)
        self.result_label.setPlainText(result_text)
        self.result_label.setStyleSheet(STYLE_RESULT_SUCCESS + ' font-weight: bold;')
        self.statusBar().showMessage(f'Analysis complete: {len(results)} face(s) detected')
    
    def _draw_faces_on_image(self, results):
        """Draw bounding boxes and labels on image"""
        result_image = self.current_image.copy()
        
        for idx, result in enumerate(results):
            age = int(result['age'])
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            cv2.circle(result_image, (x + 20, y + 20), 20, (0, 255, 0), -1)
            cv2.putText(result_image, f'{idx+1}', (x + 10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            cv2.putText(result_image, f'Age: {age}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image
    
    def _format_results_text(self, results):
        """Format results as text"""
        result_text = f"Detected {len(results)} face(s):\n\n"
        
        for idx, result in enumerate(results):
            result_text += f"Face #{idx+1}:\n"
            result_text += f"  Age: {int(result['age'])} years\n"
            result_text += f"  Gender: {result['dominant_gender']}\n"
            result_text += f"  Race: {result['dominant_race']}\n"
            result_text += f"  Emotion: {result['dominant_emotion']}\n\n"
        
        return result_text
    
    def on_prediction_error(self, error_msg):
        """Handle prediction errors"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        QMessageBox.critical(self, 'Error', f'Analysis failed:\n{error_msg}')
        self.statusBar().showMessage('Analysis failed')
    
    def start_webcam(self):
        """Start webcam in separate window"""
        import subprocess
        try:
            subprocess.Popen(['python', 'scripts/live_webcam.py'])
            self.statusBar().showMessage('Webcam launched in separate window')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not start webcam:\n{str(e)}')
    
    def clear_image(self):
        """Clear image and results"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.clear()
        self.image_label.setText('No image loaded')
        self.result_label.setPlainText('No prediction yet')
        self.result_label.setStyleSheet(STYLE_FONT_14PX + ' padding: 10px;')
        self.predict_btn.setEnabled(False)
        self.statusBar().showMessage('Cleared')


def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
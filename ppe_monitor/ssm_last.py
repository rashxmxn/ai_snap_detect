from ultralytics import YOLO
import cv2
import time
from collections import defaultdict
import logging
from datetime import datetime
import numpy as np
import os
import argparse
from pathlib import Path

# Create directory for violation screenshots if it doesn't exist
VIOLATIONS_DIR = 'violation_screenshots'
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(filename='smart_safety_violations.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

class SmartSafetyMonitor:
    def __init__(self):
        # Required PPE with violation thresholds and confidence settings
        self.REQUIRED_PPE = {
            'helmet': {'violation_time': 3, 'confidence_threshold': 0.6},
            'safety-vest': {'violation_time': 3, 'confidence_threshold': 0.6},
            'safety-suit': {'violation_time': 5, 'confidence_threshold': 0.6},
            'gloves': {'violation_time': 5, 'confidence_threshold': 0.5},
            'glasses': {'violation_time': 5, 'confidence_threshold': 0.5},
            }
        
        # Enhanced violation tracking with screenshot timestamps
        self.violations = defaultdict(lambda: {'start_time': None,
                                               'reported': False,
                                               'missing_items': set(),
                                               'last_screenshot': None
                                               })
        
        # Tracking window for smooth detection
        self.detection_history = defaultdict(lambda: [])
        self.history_window = 3
        
        # Load YOLO model
        self.model = YOLO("/home/beybars/Desktop/beybars/projects/side_hustle/EKTU/yolo9c.pt")
        
        # Initialize visualization colors with high contrast
        self.COLORS = {'person': (0, 0, 255),      # Red
                       'helmet': (255, 0, 255),     # Magenta
                       'safety-vest': (255, 165, 0), # Orange
                       'safety-suit': (0, 255, 0),   # Green
                       'gloves': (255, 255, 0),      # Yellow
                       'glasses': (0, 255, 255),     # Cyan
                       }
        
        # Create output directory for processed videos
        self.output_dir = Path('output_videos')
        self.output_dir.mkdir(exist_ok=True)

    def save_violation_screenshot(self, frame, missing_ppe):
        """Save a screenshot of the violation with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{VIOLATIONS_DIR}/violation_{timestamp}.jpg"
        
        frame_copy = frame.copy()
        cv2.putText(frame_copy, "SAFETY VIOLATION!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y_offset = 100
        for item in missing_ppe:
            cv2.putText(frame_copy, f"Missing: {item}", (50, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
            
        cv2.imwrite(filename, frame_copy)
        logging.info(f"Violation screenshot saved: {filename}")

    def check_ppe_compliance(self, frame, results):
        current_time = time.time()
        persons_detected = False
        
        current_detections = defaultdict(int)
        for box in results.boxes:
            class_name = results.names[int(box.cls[0].item())]
            conf = box.conf[0].item()
            
            if class_name in self.REQUIRED_PPE and conf >= self.REQUIRED_PPE[class_name]['confidence_threshold']:
                current_detections[class_name] += 1
            if class_name == 'person':
                persons_detected = True
        
        if not persons_detected:
            self.violations.clear()
            return frame
        
        missing_ppe = set()
        for ppe in self.REQUIRED_PPE.keys():
            has_ppe = self.smooth_detections(current_detections[ppe] > 0, ppe)
            if not has_ppe:
                missing_ppe.add(ppe)
        
        if missing_ppe:
            if self.violations['current']['start_time'] is None:
                self.violations['current']['start_time'] = current_time
                self.violations['current']['missing_items'] = missing_ppe
            else:
                violation_duration = current_time - self.violations['current']['start_time']
                should_capture = False
                
                for ppe in missing_ppe:
                    if (violation_duration >= self.REQUIRED_PPE[ppe]['violation_time'] and 
                        not self.violations['current']['reported']):
                        should_capture = True
                        
                if should_capture:
                    violation_msg = f"Safety Violation: Missing {', '.join(missing_ppe)}"
                    logging.warning(violation_msg)
                    self.save_violation_screenshot(frame, missing_ppe)
                    self.violations['current']['reported'] = True
                    
                    cv2.putText(frame, "SAFETY VIOLATION!", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_offset = 100
                    for item in missing_ppe:
                        cv2.putText(frame, f"Missing: {item}", (50, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        y_offset += 30
        else:
            self.violations['current']['start_time'] = None
            self.violations['current']['reported'] = False
            self.violations['current']['missing_items'] = set()
        
        return frame

    def smooth_detections(self, current_detection, class_name):
        """Smooth detections over multiple frames"""
        self.detection_history[class_name].append(current_detection)
        if len(self.detection_history[class_name]) > self.history_window:
            self.detection_history[class_name].pop(0)
        
        return sum(self.detection_history[class_name]) > (self.history_window / 2)

    def draw_detections(self, frame, results):
        """Draw detection boxes and labels on the frame"""
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[class_id]
            color = self.COLORS.get(class_name, (255, 255, 255))
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f'{class_name} {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

    def run(self, source=0):
        """Run the monitor with either webcam (source=0) or video file input"""
        cap = cv2.VideoCapture(source)
        
        if isinstance(source, int):  # Webcam
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.output_dir / f'processed_video_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = self.model(frame, verbose=False)[0]
                frame = self.draw_detections(frame, results)
                frame = self.check_ppe_compliance(frame, results)
                
                # Save the processed frame
                out.write(frame)
                
                cv2.imshow('Smart Safety Monitoring System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            logging.info(f"Processed video saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Smart Safety Monitor')
    parser.add_argument('--source', type=str, default='0',
                      help='Source for monitoring (0 for webcam, or path to video file)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    monitor = SmartSafetyMonitor()
    
    # Convert source to integer if it's webcam, otherwise use as path
    source = 0 if args.source == '0' else args.source
    monitor.run(source)
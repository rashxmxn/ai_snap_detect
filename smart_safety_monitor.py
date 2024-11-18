from ultralytics import YOLO
import cv2
import time
from collections import defaultdict
import logging
from datetime import datetime
import numpy as np
import os

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
        
        # Initialize visualization colors (same as original)
        self.COLORS = {'person': (0, 0, 255),
                       'helmet': (128, 0, 128),
                       'safety-vest': (255, 128, 0),
                       'safety-suit': (0, 0, 64),
                       'gloves': (128, 128, 0),
                       'glasses': (0, 0, 128),
                       }

    def save_violation_screenshot(self, frame, missing_ppe):
        """Save a screenshot of the violation with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{VIOLATIONS_DIR}/violation_{timestamp}.jpg"
        
        # Add violation details to the frame
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
        
        # Process current frame detections
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
        
        # Check for missing PPE with smoothing
        missing_ppe = set()
        for ppe in self.REQUIRED_PPE.keys():
            has_ppe = self.smooth_detections(current_detections[ppe] > 0, ppe)
            if not has_ppe:
                missing_ppe.add(ppe)
        
        # Handle violations and screenshots
        if missing_ppe:
            if self.violations['current']['start_time'] is None:
                self.violations['current']['start_time'] = current_time
                self.violations['current']['missing_items'] = missing_ppe
            else:
                # Check violation duration and take action
                violation_duration = current_time - self.violations['current']['start_time']
                should_capture = False
                
                for ppe in missing_ppe:
                    if (violation_duration >= self.REQUIRED_PPE[ppe]['violation_time'] and 
                        not self.violations['current']['reported']):
                        should_capture = True
                        
                if should_capture:
                    # Log violation and capture screenshot
                    violation_msg = f"Safety Violation: Missing {', '.join(missing_ppe)}"
                    logging.warning(violation_msg)
                    self.save_violation_screenshot(frame, missing_ppe)
                    self.violations['current']['reported'] = True
                    
                    # Draw warning on frame
                    cv2.putText(frame, "SAFETY VIOLATION!", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_offset = 100
                    for item in missing_ppe:
                        cv2.putText(frame, f"Missing: {item}", (50, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        y_offset += 30
        else:
            # Reset violation tracking when compliant
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
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class name and confidence
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[class_id]
            
            # Get color for this class (default to white if not in COLORS)
            color = self.COLORS.get(class_name, (255, 255, 255))
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f'{class_name} {conf:.2f}'
            # Create filled rectangle for text background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
            # Add white text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Perform detection
                results = self.model(frame, verbose=False)[0]
                
                # Draw bounding boxes
                frame = self.draw_detections(frame, results)
                
                # Check PPE compliance and update frame
                frame = self.check_ppe_compliance(frame, results)
                
                # Show frame
                cv2.imshow('Smart Safety Monitoring System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = SmartSafetyMonitor()
    monitor.run() 
from ultralytics import YOLO
import cv2
import time
from collections import defaultdict
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    filename='safety_violations.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class SafetyMonitor:
    def __init__(self):
        # Required PPE for each person
        self.REQUIRED_PPE = {
            'helmet': {'violation_time': 5, 'confidence_threshold': 0.6},
            'safety-vest': {'violation_time': 5, 'confidence_threshold': 0.6},
            'safety-suit': {'violation_time': 8, 'confidence_threshold': 0.6},
            'gloves': {'violation_time': 10, 'confidence_threshold': 0.5},
            'glasses': {'violation_time': 8, 'confidence_threshold': 0.5},
            }
        
        # Initialize violation tracking
        self.violations = defaultdict(lambda: {
            'start_time': None,
            'reported': False,
            'missing_items': set()
        })
        
        # Track detections for smoothing
        self.detection_history = defaultdict(lambda: [])
        self.history_window = 5
        
        # Load YOLO model
        self.model = YOLO("/home/beybars/Desktop/beybars/projects/side_hustle/EKTU/yolo9c.pt")
        
        # Initialize colors
        self.COLORS = {
            'person': (0, 0, 255),       # Red
            'ear': (0, 255, 0),         # Green
            'ear-mufs': (255, 0, 0),    # Blue
            'face': (255, 255, 0),      # Cyan
            'face-guard': (255, 0, 255), # Magenta
            'face-mask': (0, 255, 255),  # Yellow
            'foot': (128, 0, 0),        # Dark Blue
            'tool': (0, 128, 0),        # Dark Green
            'glasses': (0, 0, 128),     # Dark Red
            'gloves': (128, 128, 0),    # Dark Cyan
            'helmet': (128, 0, 128),    # Dark Magenta
            'hands': (0, 128, 128),     # Dark Yellow
            'head': (128, 128, 128),    # Gray
            'medical-suit': (64, 0, 0),  # Light Blue
            'shoes': (0, 64, 0),        # Light Green
            'safety-suit': (0, 0, 64),  # Light Red
            'safety-vest': (255, 128, 0) # Orange
        }

    def draw_detections(self, frame, results):
        """Draw detection boxes and labels"""
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class name and confidence
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[class_id]
            
            # Get color for this class
            color = self.COLORS.get(class_name, (0, 255, 0))
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def smooth_detections(self, current_detections, class_name):
        """Smooth detections over multiple frames"""
        self.detection_history[class_name].append(current_detections)
        if len(self.detection_history[class_name]) > self.history_window:
            self.detection_history[class_name].pop(0)
        
        return sum(self.detection_history[class_name]) > (self.history_window / 2)

    def check_ppe_compliance(self, frame, results):
        current_time = time.time()
        persons_detected = False
        
        # Get all detections in current frame
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
        
        # Check for missing PPE
        missing_ppe = set()
        for ppe in self.REQUIRED_PPE.keys():
            has_ppe = self.smooth_detections(current_detections[ppe] > 0, ppe)
            if not has_ppe:
                missing_ppe.add(ppe)
        
        # Update violations
        if missing_ppe:
            if self.violations['current']['start_time'] is None:
                self.violations['current']['start_time'] = current_time
                self.violations['current']['missing_items'] = missing_ppe
            else:
                # Check if violation duration exceeds threshold
                for ppe in missing_ppe:
                    violation_duration = current_time - self.violations['current']['start_time']
                    if (violation_duration >= self.REQUIRED_PPE[ppe]['violation_time'] and 
                        not self.violations['current']['reported']):
                        # Log violation
                        violation_msg = f"Safety Violation: Missing {', '.join(missing_ppe)}"
                        logging.warning(violation_msg)
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
            # Reset violation if PPE is now compliant
            self.violations['current']['start_time'] = None
            self.violations['current']['reported'] = False
            self.violations['current']['missing_items'] = set()
        
        return frame

    def draw_status(self, frame):
        """Draw current system status"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform detection
            results = self.model(frame, verbose=False)[0]  # Added verbose=False to reduce output
            
            # Draw detections
            frame = self.draw_detections(frame, results)
            
            # Check PPE compliance
            frame = self.check_ppe_compliance(frame, results)
            
            # Display status
            self.draw_status(frame)
            
            # Show frame
            cv2.imshow('Safety Monitoring System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = SafetyMonitor()
    monitor.run()
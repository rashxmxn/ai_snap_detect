import cv2
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import streamlit as st

from config.settings import PPE_COLORS, DETECTION_HISTORY_WINDOW

class PPEDetector:
    def __init__(self, model_path):
        self.violations = defaultdict(lambda: {
            'start_time': None,
            'reported': False,
            'missing_items': set(),
            'last_screenshot': None
        })
        
        self.detection_history = defaultdict(list)
        self.history_window = DETECTION_HISTORY_WINDOW
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.COLORS = PPE_COLORS
        
        # Create screenshots directory
        self.screenshots_dir = Path("violation_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
    
    def save_violation_screenshot(self, frame, missing_ppe):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        missing_items = "_".join(missing_ppe)
        filename = f"violation_{missing_items}_{timestamp}.jpg"
        filepath = self.screenshots_dir / filename
        
        # Add violation overlay
        frame_with_overlay = frame.copy()
        cv2.putText(frame_with_overlay, "SAFETY VIOLATION!", (50, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset = 100
        for item in missing_ppe:
            cv2.putText(frame_with_overlay, f"Missing: {item}", (50, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
        
        cv2.imwrite(str(filepath), frame_with_overlay)
        rgb_frame = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
        
        violation_info = {
            'image': rgb_frame,
            'timestamp': datetime.now(),
            'missing_ppe': list(missing_ppe)
        }
        st.session_state.violation_stats['violations_screenshots'].append(violation_info)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        frame = self.draw_detections(frame, results)
        frame = self.check_ppe_compliance(frame, results)
        return frame

    def check_ppe_compliance(self, frame, results):
        current_time = time.time()
        persons_detected = False
        
        current_detections = defaultdict(int)
        for box in results.boxes:
            class_name = results.names[int(box.cls[0].item())]
            conf = box.conf[0].item()
            
            if (class_name in st.session_state.required_ppe and 
                conf >= st.session_state.required_ppe[class_name]['confidence_threshold'] and
                st.session_state.required_ppe[class_name]['enabled']):
                current_detections[class_name] += 1
            if class_name == 'person':
                persons_detected = True
        
        if not persons_detected:
            self.violations.clear()
            return frame

        missing_ppe = self._check_missing_ppe(current_detections)
        frame = self._handle_violations(frame, missing_ppe, current_time)
        return frame

    def _check_missing_ppe(self, current_detections):
        missing_ppe = set()
        for ppe, settings in st.session_state.required_ppe.items():
            if settings['enabled']:
                has_ppe = self.smooth_detections(current_detections[ppe] > 0, ppe)
                if not has_ppe:
                    missing_ppe.add(ppe)
        return missing_ppe

    def _handle_violations(self, frame, missing_ppe, current_time):
        if missing_ppe:
            if self.violations['current']['start_time'] is None:
                self.violations['current']['start_time'] = current_time
                self.violations['current']['missing_items'] = missing_ppe
            else:
                violation_duration = current_time - self.violations['current']['start_time']
                should_capture = self._check_violation_duration(missing_ppe, violation_duration)
                
                if should_capture:
                    self._update_violation_stats(frame, missing_ppe)
                    frame = self._add_violation_overlay(frame, missing_ppe)
        else:
            self.violations['current'] = {
                'start_time': None,
                'reported': False,
                'missing_items': set()
            }
        return frame

    def _check_violation_duration(self, missing_ppe, duration):
        if self.violations['current']['reported']:
            return False
        
        for ppe in missing_ppe:
            if duration >= st.session_state.required_ppe[ppe]['violation_time']:
                return True
        return False

    def _update_violation_stats(self, frame, missing_ppe):
        st.session_state.violation_stats['total_violations'] += 1
        for ppe in missing_ppe:
            st.session_state.violation_stats['violations_by_ppe'][ppe] += 1
        self.violations['current']['reported'] = True
        self.save_violation_screenshot(frame, missing_ppe)

    def _add_violation_overlay(self, frame, missing_ppe):
        cv2.putText(frame, "SAFETY VIOLATION!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset = 100
        for item in missing_ppe:
            cv2.putText(frame, f"Missing: {item}", (50, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
        return frame

    def smooth_detections(self, current_detection, class_name):
        self.detection_history[class_name].append(current_detection)
        if len(self.detection_history[class_name]) > self.history_window:
            self.detection_history[class_name].pop(0)
        return sum(self.detection_history[class_name]) > (self.history_window / 2)

    def draw_detections(self, frame, results):
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[class_id]
            
            if (class_name in st.session_state.required_ppe and 
                not st.session_state.required_ppe[class_name]['enabled']):
                continue
                
            color = self.COLORS.get(class_name, (255, 255, 255))
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f'{class_name} {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
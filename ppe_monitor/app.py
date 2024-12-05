import streamlit as st
from ultralytics import YOLO
import cv2
import time
from collections import defaultdict
import logging
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import tempfile
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Smart Safety Monitor",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .violation-counter {
        font-size: 2rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'required_ppe' not in st.session_state:
    st.session_state.required_ppe = {
        'helmet': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
        'safety-vest': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
        'safety-suit': {'violation_time': 5, 'confidence_threshold': 0.6, 'enabled': True},
        'gloves': {'violation_time': 5, 'confidence_threshold': 0.5, 'enabled': True},
        'glasses': {'violation_time': 5, 'confidence_threshold': 0.5, 'enabled': True},
    }

if 'violation_stats' not in st.session_state:
    st.session_state.violation_stats = {
        'total_violations': 0,
        'violations_by_ppe': defaultdict(int),
        'violation_timestamps': [],
    }

class StreamlitSafetyMonitor:
    def __init__(self):
        
        self.violations = defaultdict(lambda: {
            'start_time': None,
            'reported': False,
            'missing_items': set(),
            'last_screenshot': None
        })
        
        self.detection_history = defaultdict(list)
        self.history_window = 3
        
        # Initialize YOLO model
        model_path = st.session_state.get('model_path', "models/yolo9c.pt")
        self.model = YOLO(model_path)
        
        self.COLORS = {
            'person': (0, 0, 255),      # Red
            'helmet': (255, 0, 255),     # Magenta
            'safety-vest': (255, 165, 0), # Orange
            'safety-suit': (0, 255, 0),   # Green
            'gloves': (255, 255, 0),      # Yellow
            'glasses': (0, 255, 255),     # Cyan
        }

    def update_violation_stats(self, missing_ppe):
        st.session_state.violation_stats['total_violations'] += 1
        for ppe in missing_ppe:
            st.session_state.violation_stats['violations_by_ppe'][ppe] += 1
        st.session_state.violation_stats['violation_timestamps'].append(datetime.now())

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

        missing_ppe = set()
        for ppe, settings in st.session_state.required_ppe.items():
            if settings['enabled']:
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
                    if (violation_duration >= st.session_state.required_ppe[ppe]['violation_time'] and 
                        not self.violations['current']['reported']):
                        should_capture = True
                        
                if should_capture:
                    self.update_violation_stats(missing_ppe)
                    self.violations['current']['reported'] = True
                    
                    # Add violation overlay to frame
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
            
            if class_name in st.session_state.required_ppe and not st.session_state.required_ppe[class_name]['enabled']:
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

def display_violation_metrics():
    st.subheader("Violation Statistics")
    
    # Display total violations
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Violations", st.session_state.violation_stats['total_violations'])
    
    # Violations by PPE type
    if st.session_state.violation_stats['violations_by_ppe']:
        violations_df = pd.DataFrame(
            list(st.session_state.violation_stats['violations_by_ppe'].items()),
            columns=['PPE Type', 'Violations']
        )
        
        fig = px.bar(violations_df, x='PPE Type', y='Violations',
                    title='Violations by PPE Type',
                    color='PPE Type')
        st.plotly_chart(fig)
    
    # Timeline of violations
    if st.session_state.violation_stats['violation_timestamps']:
        timeline_df = pd.DataFrame(
            st.session_state.violation_stats['violation_timestamps'],
            columns=['Timestamp']
        )
        timeline_df['Count'] = 1
        timeline_df = timeline_df.set_index('Timestamp')
        timeline_df = timeline_df.resample('1Min').sum()
        
        fig = px.line(timeline_df, title='Violations Timeline')
        st.plotly_chart(fig)

def main():
    st.title("🎥 Smart Safety Monitor")
    
    # Sidebar configurations
    st.sidebar.header("Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input("Model Path", value="models/yolo9c.pt")
    if 'model_path' not in st.session_state or st.session_state.model_path != model_path:
        st.session_state.model_path = model_path
    
    # PPE Configuration
    st.sidebar.subheader("PPE Settings")
    for ppe, settings in st.session_state.required_ppe.items():
        st.sidebar.markdown(f"### {ppe.title()}")
        enabled = st.sidebar.checkbox("Enable", value=settings['enabled'], key=f"{ppe}_enabled")
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0, settings['confidence_threshold'],
            key=f"{ppe}_conf"
        )
        violation_time = st.sidebar.slider(
            "Violation Time (seconds)",
            1, 10, settings['violation_time'],
            key=f"{ppe}_time"
        )
        
        st.session_state.required_ppe[ppe].update({
            'enabled': enabled,
            'confidence_threshold': conf_threshold,
            'violation_time': violation_time
        })
    
    # Reset button for violation stats
    if st.sidebar.button("Reset Violation Statistics"):
        st.session_state.violation_stats = {
            'total_violations': 0,
            'violations_by_ppe': defaultdict(int),
            'violation_timestamps': [],
        }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        source = st.radio("Select Input Source", ["Webcam", "Video File"])
        
        if source == "Video File":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                source_path = temp_file.name
            else:
                st.warning("Please upload a video file")
                return
        else:
            source_path = 0
        
        # Initialize monitor
        monitor = StreamlitSafetyMonitor()
        
        # Video display placeholder
        video_placeholder = st.empty()
        
        cap = cv2.VideoCapture(source_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = monitor.process_frame(frame)
            
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(rgb_frame)
            
            # Add a small delay to control frame rate
            time.sleep(0.01)
        
        cap.release()
        
        if source == "Video File":
            os.unlink(source_path)
    
    with col2:
        display_violation_metrics()

if __name__ == "__main__":
    main()
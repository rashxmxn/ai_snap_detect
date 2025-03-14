import streamlit as st
import cv2
import os
import tempfile
import time
from pathlib import Path
from ..base_project import BaseProject
from .detector import PPEDetector
from .utils import display_violation_metrics, reset_violation_stats
from config.settings import DEFAULT_PPE_SETTINGS, DEFAULT_VIOLATION_STATS, PPE_CSS, DEFAULT_PPE_DETECTION_MODELS

class PPEDetectionApp(BaseProject):
    def __init__(self):
        super().__init__()
        self.title = "Smart Safety Monitor"
        self.description = """
        Real-time PPE (Personal Protective Equipment) detection system for workplace safety monitoring.
        Monitor and enforce safety compliance with automatic violation detection and reporting.
        """
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables if they don't exist"""
        if 'required_ppe' not in st.session_state:
            st.session_state.required_ppe = DEFAULT_PPE_SETTINGS
        if 'violation_stats' not in st.session_state:
            st.session_state.violation_stats = DEFAULT_VIOLATION_STATS
        self.load_model()
    
    def load_model(self):
        """Initialize the PPE detection model"""
        self.model_path = st.session_state.get('model_path', DEFAULT_PPE_DETECTION_MODELS[0])
        self.detector = PPEDetector(self.model_path)
    
    def setup_sidebar(self):
        """Configure sidebar settings"""
        st.sidebar.header("Configuration")
        
        # Model selection
        model_path = st.sidebar.selectbox("Select PPE Model", DEFAULT_PPE_DETECTION_MODELS)
        if 'model_path' not in st.session_state or st.session_state.model_path != model_path:
            st.session_state.model_path = model_path
        
        # PPE Configurationf
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
        
        # Reset button
        if st.sidebar.button("Reset Violation Statistics"):
            reset_violation_stats()
    
    def process_input(self, source_path):
        """Process video input from file or webcam"""
        cap = cv2.VideoCapture(source_path)
        
        # Video display placeholder
        video_placeholder = st.empty()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Process frame
                processed_frame = self.detector.process_frame(frame, )
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Display the frame
                video_placeholder.image(rgb_frame)
                # Add a small delay to control frame rate
                time.sleep(0.01)
        finally:
            cap.release()
    
    def display_output(self, col2):
        """Display violation metrics and statistics"""
        with col2:
            display_violation_metrics()
    
    def run(self):
        """Main method to run the PPE detection interface"""
        # Apply custom CSS
        st.markdown(PPE_CSS, unsafe_allow_html=True)
        # Show project info
        self.show_project_info(self.title, self.description)
        # Setup sidebar
        self.setup_sidebar()
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            source = st.radio("Select Input Source", ["Webcam", "Video File"])
            
            if source == "Video File":
                uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.read())
                        source_path = temp_file.name
                else:
                    st.warning("Please upload a video file")
                    return
            else:
                source_path = 0
            
            # Process input
            self.process_input(source_path)
            
            # Clean up temporary file if used
            if source == "Video File" and 'source_path' in locals():
                os.unlink(source_path)
        
        # Display output
        self.display_output(col2)
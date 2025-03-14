import streamlit as st
import cv2
import os
import tempfile
import time
from pathlib import Path
import plotly.express as px
import pandas as pd

from ..base_project import BaseProject
from .detector import TrafficSignDetector
from config.settings import TRAFFIC_SIGN_SETTINGS, TRAFFIC_SIGN_CSS

class TrafficSignApp(BaseProject):
    """Streamlit interface for traffic sign detection system"""
    def __init__(self):
        super().__init__()
        self.title = "Traffic Sign Detection System"
        self.description = """
        Real-time traffic sign detection system using computer vision.
        Upload a video file or use webcam feed to detect and classify traffic signs.
        """
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize detection components"""
        if 'detector' not in st.session_state:
            self.load_model()
            
    def load_model(self):
        """Initialize face recognition components - Implementation of abstract method"""
        st.session_state.detector = TrafficSignDetector(
                model_id=TRAFFIC_SIGN_SETTINGS['model_id'],
                confidence_threshold=TRAFFIC_SIGN_SETTINGS['confidence_threshold']
            )
    
    def setup_sidebar(self):
        """Configure sidebar settings"""
        st.sidebar.header("Detection Settings")
        
        # Confidence threshold
        confidence = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0,
            TRAFFIC_SIGN_SETTINGS['confidence_threshold']
        )
        
        # Enable/disable annotations
        enable_annotations = st.sidebar.checkbox(
            "Show Annotations",
            value=TRAFFIC_SIGN_SETTINGS['enable_annotations']
        )
        
        # Reset statistics button
        if st.sidebar.button("Reset Statistics"):
            st.session_state.detector.reset_stats()
    
    def display_output(self):
        pass
    
    def process_input(self, source_path, video_placeholder):
        """Process video input from file or webcam"""
        cap = cv2.VideoCapture(source_path)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = st.session_state.detector.process_frame(frame)
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(rgb_frame)
                
                # Add a small delay to control frame rate
                time.sleep(0.01)
                
        finally:
            cap.release()
    
    def run(self):
        """Main method to run the traffic sign detection interface"""
        # Apply custom CSS
        st.markdown(TRAFFIC_SIGN_CSS, unsafe_allow_html=True)
        
        # Show project info
        self.show_project_info(self.title, self.description)
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            source = "Video File"
            
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov']
            )
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    source_path = temp_file.name
            else:
                st.warning("Please upload a video file")
                return

            
            # Create video placeholder
            video_placeholder = st.empty()
            
            # Process video
            self.process_input(source_path, video_placeholder)
            
            # Clean up temporary file if used
            if source == "Video File" and 'source_path' in locals():
                os.unlink(source_path)
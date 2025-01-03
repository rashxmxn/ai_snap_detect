import streamlit as st
import tempfile
import os
from pathlib import Path
from ..base_project import BaseProject
from .matcher import VideoFrameMatcher
from config.settings import SCENARIO_SEARCH_SETTINGS, SCENARIO_SEARCH_CSS

class ScenarioSearchApp(BaseProject):
    """Streamlit interface for scenario-based video search"""
    def __init__(self):
        super().__init__()
        self.title = "Video Scenario Search"
        self.description = """
        Search for specific scenes in videos using natural language descriptions.
        Simply upload a video and describe what you're looking for!
        """
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize the video frame matcher"""
        if 'matcher' not in st.session_state:
            self.load_model()
            
    def load_model(self):
        """Initialize the video frame matcher - Implementation of abstract method"""
        st.session_state.matcher = VideoFrameMatcher()
        
    def setup_sidebar(self):
        """Configure sidebar settings"""
        st.sidebar.header("Search Settings")
        
        # Similarity threshold
        threshold = st.sidebar.slider(
            "Similarity Threshold",
            0.0, 1.0,
            SCENARIO_SEARCH_SETTINGS['similarity_threshold']
        )
        
        # Sample rate
        sample_rate = st.sidebar.slider(
            "Sample Rate (frames)",
            1, 60,
            SCENARIO_SEARCH_SETTINGS['sample_rate']
        )
        
        return threshold, sample_rate
    
    def process_input(self):
        """Process the input data"""
        pass
    
    def display_output(self, results):
        """Display search results"""
        st.subheader("Search Results")
        
        if not results['frames']:
            st.warning("No matching frames found. Try adjusting the similarity threshold.")
            return
            
        # Display results in columns
        cols = st.columns(min(3, len(results['frames'])))
        for i, (frame, timestamp, score) in enumerate(zip(
            results['frames'],
            results['timestamps'],
            results['scores']
        )):
            with cols[i]:
                st.image(frame, caption=f"Time: {timestamp:.2f}s")
                st.markdown(f"<div class='similarity-score'>Similarity: {score:.2f}%</div>",
                          unsafe_allow_html=True)
    
    def run(self):
        """Main method to run the scenario search interface"""
        # Apply custom CSS
        st.markdown(SCENARIO_SEARCH_CSS, unsafe_allow_html=True)
        
        # Show project info
        self.show_project_info(self.title, self.description)
        
        # Setup sidebar
        threshold, sample_rate = self.setup_sidebar()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov']
        )
        
        # Search query
        text_query = st.text_input(
            "Describe what you're looking for",
            placeholder="E.g., 'a person wearing a red shirt'"
        )
        
        # Process search
        if uploaded_file is not None and text_query:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                
                try:
                    with st.spinner("Searching for matching frames..."):
                        results = st.session_state.matcher.find_relevant_frames(
                            video_path=video_path,
                            text_query=text_query,
                            threshold=threshold,
                            sample_rate=sample_rate
                        )
                        
                    self.display_output(results)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(video_path)
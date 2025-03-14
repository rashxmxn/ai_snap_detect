# import streamlit as st
# import tempfile, time
# import os
# from pathlib import Path
# from ..base_project import BaseProject
# from .matcher import VideoFrameMatcher
# from config.settings import SCENARIO_SEARCH_SETTINGS, SCENARIO_SEARCH_CSS

# class ScenarioSearchApp(BaseProject):
#     """Streamlit interface for scenario-based video search"""
#     def __init__(self):
#         super().__init__()
#         self.title = "Video Scenario Search"
#         self.description = """
#         Search for specific scenes in videos using natural language descriptions.
#         Simply upload a video and describe what you're looking for!
#         """
#         self._initialize_session_state()
    
#     def _initialize_session_state(self):
#         """Initialize the video frame matcher"""
#         if 'matcher' not in st.session_state:
#             self.load_model()
            
#     def load_model(self):
#         """Initialize the video frame matcher - Implementation of abstract method"""
#         st.session_state.matcher = VideoFrameMatcher()
        
#     def setup_sidebar(self):
#         """Configure sidebar settings"""
#         st.sidebar.header("Search Settings")
        
#         # Similarity threshold
#         threshold = st.sidebar.slider(
#             "Similarity Threshold",
#             0.0, 1.0,
#             SCENARIO_SEARCH_SETTINGS['similarity_threshold']
#         )
        
#         # Sample rate
#         sample_rate = st.sidebar.slider(
#             "Sample Rate (frames)",
#             1, 60,
#             SCENARIO_SEARCH_SETTINGS['sample_rate']
#         )
        
#         return threshold, sample_rate
    
#     def process_input(self):
#         """Process the input data"""
#         pass
    
#     def display_output(self, results):
#         """Display search results"""
#         st.subheader("Search Results")
        
#         if not results['frames']:
#             st.warning("No matching frames found. Try adjusting the similarity threshold.")
#             return
            
#         # Display results in columns
#         cols = st.columns(min(3, len(results['frames'])))
#         for i, (frame, timestamp, score) in enumerate(zip(
#             results['frames'],
#             results['timestamps'],
#             results['scores']
#         )):
#             with cols[i]:
#                 st.image(frame, caption=f"Time: {timestamp:.2f}s")
#                 st.markdown(f"<div class='similarity-score'>Similarity: {score:.2f}%</div>",
#                           unsafe_allow_html=True)
    
#     def run(self):
#         """Main method to run the scenario search interface"""
#         # Apply custom CSS
#         st.markdown(SCENARIO_SEARCH_CSS, unsafe_allow_html=True)
        
#         # Show project info
#         self.show_project_info(self.title, self.description)
        
#         # Setup sidebar
#         threshold, sample_rate = self.setup_sidebar()
        
#         # File upload
#         uploaded_file = st.file_uploader(
#             "Upload a video file",
#             type=['mp4', 'avi', 'mov']
#         )
        
#         # Search query
#         text_query = st.text_input(
#             "Describe what you're looking for",
#             placeholder="E.g., 'a person wearing a red shirt'"
#         )
        
#         # Process search
#         if uploaded_file is not None and text_query:
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#                 temp_file.write(uploaded_file.read())
#                 video_path = temp_file.name
                
                
#                 try:
#                     with st.spinner("Searching for matching frames..."):
#                         results = st.session_state.matcher.find_relevant_frames(
#                             video_path=video_path,
#                             text_query=text_query,
#                             threshold=threshold,
#                             sample_rate=sample_rate
#                         )
                        
#                     self.display_output(results)
                    
#                 finally:
#                     # Clean up temporary file
#                     os.unlink(video_path)

#     def run(self):
#         """Main method to run the scenario search interface"""
#         # Apply custom CSS
#         st.markdown(SCENARIO_SEARCH_CSS, unsafe_allow_html=True)
        
#         # Show project info
#         self.show_project_info(self.title, self.description)
        
#         # Setup sidebar
#         threshold, sample_rate = self.setup_sidebar()
        
#         # File upload
#         uploaded_file = st.file_uploader(
#             "Upload a video file",
#             type=['mp4', 'avi', 'mov']
#         )
        
#         # Search query
#         text_query = st.text_input(
#             "Describe what you're looking for",
#             placeholder="E.g., 'a person wearing a red shirt'"
#         )
        
#         # Process search
#         if uploaded_file is not None and text_query:
#             # Create temporary file with a more controlled cleanup approach
#             temp_dir = tempfile.mkdtemp()
#             video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            
#             try:
#                 # Write the file content to disk
#                 with open(video_path, 'wb') as f:
#                     f.write(uploaded_file.read())
                
#                 with st.spinner("Searching for matching frames..."):
#                     try:
#                         results = st.session_state.matcher.find_relevant_frames(
#                             video_path=video_path,
#                             text_query=text_query,
#                             threshold=threshold,
#                             sample_rate=sample_rate
#                         )
#                         self.display_output(results)
#                     except TypeError as e:
#                         st.error(f"Error processing video frames: {str(e)}")
#                         st.info("This may be due to an issue with the similarity calculation. Try a different video or query.")
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#             finally:
#                 # Properly close any resources before deleting
#                 try:
#                     # Make sure all resources are released before deleting
#                     import gc
#                     gc.collect()
                    
#                     # Give Windows a moment to release file locks
#                     time.sleep(0.5)
                    
#                     # Remove the file if it exists
#                     if os.path.exists(video_path):
#                         os.remove(video_path)
                    
#                     # Remove the temporary directory
#                     if os.path.exists(temp_dir):
#                         os.rmdir(temp_dir)
#                 except Exception as cleanup_error:
#                     # Log error but don't crash the app
#                     st.warning(f"Could not clean up temporary files. They will be removed later: {str(cleanup_error)}")

import streamlit as st
import tempfile
import os
import time
import gc
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
            with cols[i % len(cols)]:  # Ensure we don't go out of bounds
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
        
        # Store file path in session state to handle reloads
        if 'video_path' not in st.session_state:
            st.session_state.video_path = None
            st.session_state.temp_dir = None
            
        # Process search
        if uploaded_file is not None and text_query:
            # Create a new temp file only if the file has changed
            if uploaded_file != st.session_state.get('last_uploaded_file'):
                # Clean up previous temp file if it exists
                self._cleanup_temp_files()
                
                # Create temporary file in a dedicated directory
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, "uploaded_video.mp4")
                
                # Save the file paths in session state
                st.session_state.temp_dir = temp_dir
                st.session_state.video_path = video_path
                st.session_state.last_uploaded_file = uploaded_file
                
                # Write the file content to disk
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.read())
            else:
                # Reuse the existing file
                video_path = st.session_state.video_path
                
            try:
                with st.spinner("Searching for matching frames..."):
                    results = st.session_state.matcher.find_relevant_frames(
                        video_path=video_path,
                        text_query=text_query,
                        threshold=threshold,
                        sample_rate=sample_rate
                    )
                    self.display_output(results)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different video or query")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files safely"""
        if st.session_state.get('video_path') and os.path.exists(st.session_state.video_path):
            try:
                # Force garbage collection to release any file handles
                gc.collect()
                
                # Give Windows time to release file locks
                time.sleep(0.5)
                
                # Remove the file
                os.remove(st.session_state.video_path)
                
                # Remove the directory
                if st.session_state.get('temp_dir') and os.path.exists(st.session_state.temp_dir):
                    os.rmdir(st.session_state.temp_dir)
            except Exception as e:
                # Log error but don't crash
                print(f"Could not clean up temporary files: {str(e)}")
                # We'll try again next time
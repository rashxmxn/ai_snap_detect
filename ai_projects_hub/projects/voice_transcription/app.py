# import streamlit as st
# from pathlib import Path
# from ..base_project import BaseProject
# from .transcriber import AudioTranscriber
# from config.settings import VOICE_TRANSCRIPTION_SETTINGS, VOICE_TRANSCRIPTION_CSS

# class VoiceTranscriptionApp(BaseProject):
#     """Streamlit interface for voice transcription"""
#     def __init__(self):
#         super().__init__()
#         self.title = "Voice Transcription"
#         self.description = """
#         Convert speech to text using state-of-the-art Whisper model.
#         Record live audio or upload audio files for transcription.
#         """
#         self._initialize_session_state()
    
#     def _initialize_session_state(self):
#         """Initialize the audio transcriber"""
#         if 'transcriber' not in st.session_state:
#             self.load_model()
    
#     def load_model(self):
#         """Initialize the audio transcriber - Implementation of abstract method"""
#         st.session_state.transcriber = AudioTranscriber()
    
#     def setup_sidebar(self):
#         """Configure sidebar settings"""
#         st.sidebar.header("Transcription Settings")
        
#         # Model selection (if multiple models are supported)
#         model_size = st.sidebar.selectbox(
#             "Model Size",
#             ["tiny", "base", "small", "medium", "large"],
#             index=1  # "base" as default
#         )
        
#         # Recording duration
#         duration = st.sidebar.slider(
#             "Recording Duration (seconds)",
#             10,
#             VOICE_TRANSCRIPTION_SETTINGS['max_duration'],
#             VOICE_TRANSCRIPTION_SETTINGS['default_duration']
#         )
        
#         return model_size, duration
    
#     def process_input(self):
#         """Process the input data"""
#         pass
    
#     def display_output(self, result: dict):
#         """Display transcription results"""
#         if not result:
#             return
            
#         st.subheader("Transcription Results")
        
#         # Display full text
#         with st.expander("Show Full Text", expanded=True):
#             st.markdown(f"<div class='transcription-box'>{result['text']}</div>",
#                       unsafe_allow_html=True)
        
#         # Display segments with timestamps
#         st.subheader("Segments")
#         for segment in result["segments"]:
#             st.markdown(
#                 f"""<div class='segment'>
#                     <span class='timestamp'>
#                         [{segment['start']:.1f}s -> {segment['end']:.1f}s]
#                     </span>
#                     <br>{segment['text']}
#                 </div>""",
#                 unsafe_allow_html=True
#             )
    
#     def run(self):
#         """Main method to run the voice transcription interface"""
#         # Apply custom CSS
#         st.markdown(VOICE_TRANSCRIPTION_CSS, unsafe_allow_html=True)
        
#         # Show project info
#         self.show_project_info(self.title, self.description)
        
#         # Setup sidebar
#         model_size, duration = self.setup_sidebar()
        
#         # Input method selection
#         input_method = st.radio(
#             "Select Input Method",
#             ["Record Live Audio", "Upload Audio File"]
#         )
        
#         if input_method == "Record Live Audio":
#             if st.button("Start Recording"):
#                 result = st.session_state.transcriber.transcribe_live(duration)
#                 self.display_output(result)
                
#         else:  # Upload Audio File
#             uploaded_file = st.file_uploader(
#                 "Upload an audio file",
#                 type=['wav', 'mp3', 'ogg', 'm4a']
#             )
            
#             if uploaded_file:
#                 result = st.session_state.transcriber.transcribe_uploaded_file(
#                     uploaded_file)
#                 self.display_output(result)
import streamlit as st
import os
import tempfile
import sys
from pathlib import Path
from ..base_project import BaseProject
from .transcriber import AudioTranscriber
from config.settings import VOICE_TRANSCRIPTION_SETTINGS, VOICE_TRANSCRIPTION_CSS

class VoiceTranscriptionApp(BaseProject):
    """Streamlit interface for voice transcription"""
    def __init__(self):
        super().__init__()
        self.title = "Voice Transcription"
        self.description = """
        Convert speech to text using state-of-the-art Whisper model.
        Record live audio or upload audio files for transcription.
        """
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize the audio transcriber"""
        if 'transcriber' not in st.session_state:
            try:
                self.load_model()
            except Exception as e:
                st.error(f"Error initializing transcriber: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
        # Initialize temp file tracking
        if 'temp_audio_files' not in st.session_state:
            st.session_state.temp_audio_files = []
    
    def load_model(self):
        """Initialize the audio transcriber - Implementation of abstract method"""
        st.session_state.transcriber = AudioTranscriber()
    
    def setup_sidebar(self):
        """Configure sidebar settings"""
        st.sidebar.header("Transcription Settings")
        
        # Model selection (if multiple models are supported)
        model_size = st.sidebar.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1  # "base" as default
        )
        
        # Recording duration
        duration = st.sidebar.slider(
            "Recording Duration (seconds)",
            10,
            VOICE_TRANSCRIPTION_SETTINGS['max_duration'],
            VOICE_TRANSCRIPTION_SETTINGS['default_duration']
        )
        
        return model_size, duration
    
    def process_input(self):
        """Process the input data"""
        pass
    
    def display_output(self, result: dict):
        """Display transcription results"""
        if not result:
            st.warning("No transcription results to display")
            return
            
        st.subheader("Transcription Results")
        
        # Display full text
        with st.expander("Show Full Text", expanded=True):
            st.markdown(f"<div class='transcription-box'>{result['text']}</div>",
                      unsafe_allow_html=True)
        
        # Display segments with timestamps
        st.subheader("Segments")
        for segment in result["segments"]:
            st.markdown(
                f"""<div class='segment'>
                    <span class='timestamp'>
                        [{segment['start']:.1f}s -> {segment['end']:.1f}s]
                    </span>
                    <br>{segment['text']}
                </div>""",
                unsafe_allow_html=True
            )
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        for file_path in st.session_state.temp_audio_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    st.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                st.warning(f"Could not clean up file {file_path}: {str(e)}")
                
        st.session_state.temp_audio_files = []
    
    def run(self):
        """Main method to run the voice transcription interface"""
        # Apply custom CSS
        st.markdown(VOICE_TRANSCRIPTION_CSS, unsafe_allow_html=True)
        
        # Show project info
        self.show_project_info(self.title, self.description)
        
        # Setup sidebar
        model_size, duration = self.setup_sidebar()
        
        # Input method selection
        input_method = st.radio(
            "Select Input Method",
            ["Record Live Audio", "Upload Audio File"]
        )
        
        if input_method == "Record Live Audio":
            if st.button("Start Recording"):
                with st.spinner("Recording and transcribing..."):
                    # Direct integrated process
                    result = st.session_state.transcriber.transcribe_live(duration)
                    self.display_output(result)
                
        else:  # Upload Audio File
            uploaded_file = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3', 'ogg', 'm4a']
            )
            
            if uploaded_file:
                st.info("File uploaded successfully")
                
                # DIRECT APPROACH: Process with transcriber directly
                with st.spinner("Transcribing uploaded file..."):
                    result = st.session_state.transcriber.transcribe_uploaded_file(uploaded_file)
                    
                if result:
                    self.display_output(result)
                else:
                    st.error("Transcription failed. Please try a different file or check the logs.")
                
        # Cleanup on rerun
        self.cleanup_temp_files()
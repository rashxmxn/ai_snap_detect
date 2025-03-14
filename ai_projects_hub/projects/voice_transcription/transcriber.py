# import torch
# import whisper
# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import streamlit as st
# from config.settings import VOICE_TRANSCRIPTION_SETTINGS

# class AudioTranscriber:
#     """Handles audio recording and transcription using Whisper"""
    
#     def __init__(self):
#         """Initialize the transcriber with specified Whisper model"""
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         st.info(f"Using device: {self.device}")
        
#         # Load model
#         with st.spinner("Loading Whisper model..."):
#             self.model = whisper.load_model(
#                 VOICE_TRANSCRIPTION_SETTINGS['model_size']
#             ).to(self.device)
        
#         # Recording parameters
#         self.sample_rate = VOICE_TRANSCRIPTION_SETTINGS['sample_rate']
#         self.channels = VOICE_TRANSCRIPTION_SETTINGS['channels']
#         self.recordings_dir = Path(VOICE_TRANSCRIPTION_SETTINGS['recordings_dir'])
#         self.recordings_dir.mkdir(exist_ok=True)
    
#     def record_audio(self, duration: int, filename: str = None) -> str:
#         """
#         Record audio for specified duration and save to file
        
#         Args:
#             duration: Recording duration in seconds
#             filename: Optional filename to save the recording
            
#         Returns:
#             Path to the saved audio file
#         """
#         try:
#             # Record audio
#             recording = sd.rec(
#                 int(duration * self.sample_rate),
#                 samplerate=self.sample_rate,
#                 channels=self.channels
#             )
#             sd.wait()
            
#             # Generate filename if not provided
#             if filename is None:
#                 filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
#             filepath = self.recordings_dir / filename
            
#             # Save the recording
#             sf.write(filepath, recording, self.sample_rate)
#             return str(filepath)
            
#         except Exception as e:
#             st.error(f"Error during recording: {str(e)}")
#             return None
    
#     def transcribe_file(self, audio_path: str) -> dict:
#         """
#         Transcribe an audio file to text
        
#         Args:
#             audio_path: Path to the audio file
            
#         Returns:
#             Dictionary containing transcription results
#         """
#         try:
#             result = self.model.transcribe(audio_path)
#             return result
#         except Exception as e:
#             st.error(f"Error during transcription: {str(e)}")
#             return None
    
#     def transcribe_live(self, duration: int) -> dict:
#         """
#         Record and transcribe audio in one go
        
#         Args:
#             duration: Recording duration in seconds
            
#         Returns:
#             Dictionary containing transcription results
#         """
#         with st.spinner("Recording audio..."):
#             audio_path = self.record_audio(duration)
            
#         if audio_path:
#             with st.spinner("Transcribing..."):
#                 return self.transcribe_file(audio_path)
#         return None
    
#     def transcribe_uploaded_file(self, uploaded_file) -> dict:
#         """
#         Transcribe an uploaded audio file
        
#         Args:
#             uploaded_file: Streamlit UploadedFile object
            
#         Returns:
#             Dictionary containing transcription results
#         """
#         try:
#             # Save uploaded file temporarily
#             temp_path = self.recordings_dir / f"temp_{uploaded_file.name}"
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Transcribe
#             with st.spinner("Transcribing uploaded file..."):
#                 result = self.transcribe_file(str(temp_path))
            
#             # Cleanup
#             temp_path.unlink()
#             return result
            
#         except Exception as e:
#             st.error(f"Error processing uploaded file: {str(e)}")
#             return None
import torch
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import streamlit as st
from config.settings import VOICE_TRANSCRIPTION_SETTINGS

class AudioTranscriber:
    """Handles audio recording and transcription using Whisper"""
    
    def __init__(self):
        """Initialize the transcriber with specified Whisper model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {self.device}")
        
        # Check for FFmpeg (required by Whisper)
        self._check_ffmpeg()
        
        # Load model
        with st.spinner("Loading Whisper model..."):
            try:
                self.model = whisper.load_model(
                    VOICE_TRANSCRIPTION_SETTINGS['model_size']
                ).to(self.device)
            except Exception as e:
                st.error(f"Error loading Whisper model: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                raise
        
        # Recording parameters
        self.sample_rate = VOICE_TRANSCRIPTION_SETTINGS['sample_rate']
        self.channels = VOICE_TRANSCRIPTION_SETTINGS['channels']
        
        # Use a guaranteed-to-exist directory for temporary files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a secondary recordings directory for permanent storage if desired
        try:
            # Try absolute path first
            self.recordings_dir = Path(VOICE_TRANSCRIPTION_SETTINGS['recordings_dir']).absolute()
            self.recordings_dir.mkdir(exist_ok=True, parents=True)
            
        except Exception as e:
            st.warning(f"Could not create configured recordings directory: {str(e)}")
            # Fallback to temp directory
            self.recordings_dir = Path(self.temp_dir)
            st.info(f"Fallback to temporary directory: {str(self.recordings_dir)}")
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available and provide instructions if not"""
        import subprocess
        import shutil
        
        ffmpeg_found = False
        
        # Check if ffmpeg is in PATH
        if shutil.which("ffmpeg"):
            st.success("FFmpeg found in PATH.")
            ffmpeg_found = True
        else:
            # Try to find ffmpeg in common Windows locations
            possible_locations = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            ]
            
            for location in possible_locations:
                if os.path.exists(location):
                    # Add to PATH
                    os.environ["PATH"] += os.pathsep + os.path.dirname(location)
                    st.success(f"FFmpeg found at {location} and added to PATH.")
                    ffmpeg_found = True
                    break
        
        if not ffmpeg_found:
            st.error("FFmpeg not found! Whisper requires FFmpeg to process audio files.")
            st.info("""
            Please install FFmpeg:
            
            1. Download from https://ffmpeg.org/download.html
            2. Extract to C:\\ffmpeg
            3. Add C:\\ffmpeg\\bin to your PATH environment variable
            
            Or use this direct installer and restart the application:
            https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
            """)
    
    def _load_audio_direct(self, file_path):
        """
        Load audio without using FFmpeg (direct loading via soundfile)
        
        This is a fallback method when FFmpeg is not available
        """
        try:
            # Load audio file using soundfile
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz if needed (required by Whisper)
            if sample_rate != 16000:
                # Simple resampling - not ideal but works in a pinch
                import scipy.signal
                audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
            
            # Normalize audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / (np.iinfo(audio_data.dtype).max + 1)
                
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
            return audio_data
            
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def record_audio(self, duration: int, filename: str = None) -> str:
        """
        Record audio for specified duration and save to file
        
        Args:
            duration: Recording duration in seconds
            filename: Optional filename to save the recording
            
        Returns:
            Path to the saved audio file
        """
        try:
            # Record audio
            st.info("Starting audio recording...")
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels
            )
            sd.wait()
            st.success("Recording complete!")
            
            # Generate filename if not provided
            if filename is None:
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Use temp directory for temporary storage
            filepath = Path(self.temp_dir) / filename
            
            # Save the recording
            sf.write(filepath, recording, self.sample_rate)
            st.info(f"Audio saved to: {filepath}")
            
            # Verify file exists
            if not os.path.exists(filepath):
                st.error(f"Failed to save audio file at {filepath}")
                return None
                
            return str(filepath)
            
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def transcribe_file(self, audio_path: str) -> dict:
        """
        Transcribe an audio file to text
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results
        """
        try:
                        
            # Verify file exists before transcribing
            if not os.path.exists(audio_path):
                st.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            try:
                result = self.model.transcribe(audio_path)
                st.success("Transcription complete!")
            except FileNotFoundError:
                # FFmpeg not found, try direct audio loading
                st.warning("FFmpeg not found. Falling back to direct audio loading...")
                audio_data = self._load_audio_direct(audio_path)
                
                if audio_data is None:
                    st.error("Failed to load audio directly.")
                    return None
                    
                result = self.model.transcribe(audio_data)
            
            return result
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def transcribe_audio_data(self, audio_data, format="wav"):
        """
        Transcribe audio data directly without saving to disk
        
        Args:
            audio_data: Audio data as bytes or numpy array
            format: Audio format (default: wav)
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # If audio_data is bytes, write directly
                if isinstance(audio_data, bytes):
                    temp_file.write(audio_data)
                else:
                    # Assume it's a numpy array
                    sf.write(temp_path, audio_data, self.sample_rate)
            
            # Process the temporary file
            result = self.transcribe_file(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            st.error(f"Error processing audio data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def transcribe_live(self, duration: int) -> dict:
        """
        Record and transcribe audio in one go
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Dictionary containing transcription results
        """
        with st.spinner("Recording audio..."):
            audio_path = self.record_audio(duration)
            
        if audio_path:
            with st.spinner("Transcribing..."):
                return self.transcribe_file(audio_path)
        return None
    
    def transcribe_uploaded_file(self, uploaded_file) -> dict:
        """
        Transcribe an uploaded audio file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            
            # DIRECT APPROACH: Process the file data directly
            file_bytes = uploaded_file.getvalue()
            
            # Get the file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace(".", "")
            if not file_extension:
                file_extension = "wav"  # Default to wav if no extension
                
            
            # Create a temporary file with the right extension
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_bytes)
                
            
            # Verify file exists and has content
            if not os.path.exists(temp_path):
                st.error(f"Failed to create temporary file: {temp_path}")
                return None
                
            if os.path.getsize(temp_path) == 0:
                st.error(f"Temporary file is empty: {temp_path}")
                return None
                
            
            # Transcribe
            with st.spinner("Transcribing uploaded file..."):
                result = self.transcribe_file(temp_path)
            
            # Cleanup
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as cleanup_error:
                st.warning(f"Could not delete temporary file: {str(cleanup_error)}")
                
            return result
            
        except Exception as e:
            import traceback
            st.error(f"Error processing uploaded file: {str(e)}")
            st.error(traceback.format_exc())
            return None
            
    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass  # Silently ignore cleanup errors
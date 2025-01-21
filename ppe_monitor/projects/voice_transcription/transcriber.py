import torch
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
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
        
        # Load model
        with st.spinner("Loading Whisper model..."):
            self.model = whisper.load_model(
                VOICE_TRANSCRIPTION_SETTINGS['model_size']
            ).to(self.device)
        
        # Recording parameters
        self.sample_rate = VOICE_TRANSCRIPTION_SETTINGS['sample_rate']
        self.channels = VOICE_TRANSCRIPTION_SETTINGS['channels']
        self.recordings_dir = Path(VOICE_TRANSCRIPTION_SETTINGS['recordings_dir'])
        self.recordings_dir.mkdir(exist_ok=True)
    
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
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels
            )
            sd.wait()
            
            # Generate filename if not provided
            if filename is None:
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            filepath = self.recordings_dir / filename
            
            # Save the recording
            sf.write(filepath, recording, self.sample_rate)
            return str(filepath)
            
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
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
            result = self.model.transcribe(audio_path)
            return result
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
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
            # Save uploaded file temporarily
            temp_path = self.recordings_dir / f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Transcribe
            with st.spinner("Transcribing uploaded file..."):
                result = self.transcribe_file(str(temp_path))
            
            # Cleanup
            temp_path.unlink()
            return result
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None
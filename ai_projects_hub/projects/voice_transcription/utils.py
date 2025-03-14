import numpy as np
import soundfile as sf
from pathlib import Path
import sounddevice as sd
from typing import Tuple, Optional
import streamlit as st

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the data and sample rate
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (audio data array, sample rate)
    """
    try:
        data, sample_rate = sf.read(file_path)
        return data, sample_rate
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def save_audio(file_path: str, data: np.ndarray, sample_rate: int) -> bool:
    """
    Save audio data to a file
    
    Args:
        file_path: Path to save the audio file
        data: Audio data array
        sample_rate: Sample rate of the audio
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sf.write(file_path, data, sample_rate)
        return True
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return False

def normalize_audio(data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to [-1, 1] range
    
    Args:
        data: Audio data array
        
    Returns:
        Normalized audio data
    """
    return data / np.max(np.abs(data))

def convert_audio_format(data: np.ndarray, from_channels: int, to_channels: int) -> np.ndarray:
    """
    Convert audio between mono and stereo formats
    
    Args:
        data: Audio data array
        from_channels: Current number of channels
        to_channels: Desired number of channels
        
    Returns:
        Converted audio data
    """
    if from_channels == to_channels:
        return data
        
    if from_channels == 1 and to_channels == 2:
        return np.column_stack((data, data))
    
    if from_channels == 2 and to_channels == 1:
        return np.mean(data, axis=1)
    
    raise ValueError(f"Unsupported channel conversion: {from_channels} to {to_channels}")

def get_audio_duration(data: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the duration of an audio clip
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        
    Returns:
        Duration in seconds
    """
    return len(data) / sample_rate

def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate
    
    Args:
        data: Audio data array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return data
        
    duration = len(data) / orig_sr
    new_length = int(duration * target_sr)
    return np.interp(
        np.linspace(0, duration, new_length),
        np.linspace(0, duration, len(data)),
        data
    )

def split_audio(data: np.ndarray, sample_rate: int, segment_duration: float) -> list:
    """
    Split audio into segments of specified duration
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of audio segments
    """
    samples_per_segment = int(segment_duration * sample_rate)
    segments = []
    
    for start in range(0, len(data), samples_per_segment):
        end = start + samples_per_segment
        segment = data[start:end]
        
        # Pad last segment if needed
        if len(segment) < samples_per_segment:
            segment = np.pad(
                segment,
                (0, samples_per_segment - len(segment)),
                mode='constant'
            )
        
        segments.append(segment)
    
    return segments

def detect_silence(data: np.ndarray, threshold: float = 0.01, min_duration: float = 0.1, 
                  sample_rate: int = 16000) -> list:
    """
    Detect silent segments in audio
    
    Args:
        data: Audio data array
        threshold: Amplitude threshold for silence
        min_duration: Minimum silence duration in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        List of (start, end) tuples indicating silent segments
    """
    amplitude = np.abs(data)
    is_silence = amplitude < threshold
    
    min_samples = int(min_duration * sample_rate)
    silent_segments = []
    
    start = None
    for i in range(len(is_silence)):
        if is_silence[i] and start is None:
            start = i
        elif not is_silence[i] and start is not None:
            if i - start >= min_samples:
                silent_segments.append((start / sample_rate, i / sample_rate))
            start = None
            
    # Handle silence at the end
    if start is not None and len(is_silence) - start >= min_samples:
        silent_segments.append((start / sample_rate, len(is_silence) / sample_rate))
    
    return silent_segments


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, List, Tuple
import librosa
import librosa.display

def plot_waveform(data: np.ndarray, sample_rate: int, title: str = "Waveform") -> None:
    """
    Plot audio waveform using Plotly
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        title: Plot title
    """
    time = np.arange(len(data)) / sample_rate
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=data,
        mode='lines',
        name='waveform'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_spectrogram(data: np.ndarray, sample_rate: int, 
                    title: str = "Spectrogram") -> None:
    """
    Plot spectrogram using librosa and streamlit
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sample_rate, ax=ax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    st.pyplot(fig)
    plt.close()

def plot_mel_spectrogram(data: np.ndarray, sample_rate: int, 
                        title: str = "Mel Spectrogram") -> None:
    """
    Plot mel spectrogram using librosa and streamlit
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    mel_spect = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_db, y_axis='mel', x_axis='time', sr=sample_rate, ax=ax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    st.pyplot(fig)
    plt.close()

def plot_silence_detection(data: np.ndarray, sample_rate: int, 
                         silent_segments: List[Tuple[float, float]], 
                         title: str = "Silence Detection") -> None:
    """
    Plot waveform with highlighted silent segments
    
    Args:
        data: Audio data array
        sample_rate: Sample rate of the audio
        silent_segments: List of (start, end) tuples for silent segments
        title: Plot title
    """
    time = np.arange(len(data)) / sample_rate
    
    fig = go.Figure()
    
    # Plot waveform
    fig.add_trace(go.Scatter(
        x=time,
        y=data,
        mode='lines',
        name='waveform'
    ))
    
    # Add colored regions for silent segments
    for start, end in silent_segments:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="silence",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_transcription_alignment(audio_data: np.ndarray, sample_rate: int, 
                               segments: List[dict], title: str = "Transcription Alignment") -> None:
    """
    Plot waveform with aligned transcription segments
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate of the audio
        segments: List of transcription segments with 'start', 'end', and 'text' keys
        title: Plot title
    """
    time = np.arange(len(audio_data)) / sample_rate
    
    fig = go.Figure()
    
    # Plot waveform
    fig.add_trace(go.Scatter(
        x=time,
        y=audio_data,
        mode='lines',
        name='waveform'
    ))
    
    # Add colored regions and text for transcription segments
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for segment, color in zip(segments, colors):
        rgba_color = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.1)"
        fig.add_vrect(
            x0=segment['start'],
            x1=segment['end'],
            fillcolor=rgba_color,
            layer="below",
            line_width=0,
            annotation_text=segment['text'],
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
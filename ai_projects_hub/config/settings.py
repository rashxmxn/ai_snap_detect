from collections import defaultdict
import torch
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
print(ROBOFLOW_API_KEY)

### PPE Detector ###
DEFAULT_PPE_SETTINGS = {
    'helmet': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
    'safety-vest': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
    'gloves': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
    'glasses': {'violation_time': 3, 'confidence_threshold': 0.6, 'enabled': True},
}

# Color settings for visualization
PPE_COLORS = {
    'person': (0, 0, 255),      # Red
    'helmet': (255, 0, 55),    # Magenta
    'safety-vest': (255, 165, 0),# Orange
    'safety-suit': (0, 255, 0),  # Green
    'gloves': (255, 255, 0),     # Yellow
    'glasses': (0, 255, 255),    # Cyan
}

# Default violation statistics structure
DEFAULT_VIOLATION_STATS = {
    'total_violations': 0,
    'violations_by_ppe': defaultdict(int),
    'violations_screenshots': []
}

# Model settings
DEFAULT_PPE_DETECTION_MODELS = ["models/yolo9c.pt", "models/yolo9e.pt",]

# Detection settings
DETECTION_HISTORY_WINDOW = 3

# Custom CSS
PPE_CSS = """
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
"""



### Face Recognition ###
FACE_RECOGNITION_SETTINGS = {
    'similarity_threshold': 0.85,
    'db_dir': 'face_db',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Face Recognition Custom CSS
FACE_RECOGNITION_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
"""

# Ensure database directory exists
Path(FACE_RECOGNITION_SETTINGS['db_dir']).mkdir(parents=True, exist_ok=True)


### Traffic Sign Detection ###
TRAFFIC_SIGN_SETTINGS = {
    'model_id': "traffic_sign-gv5rp-zklpc/1",
    'confidence_threshold': 0.5,
    'iou_threshold': 0.5,
    'enable_annotations': True
}

# Traffic Sign Detection CSS
TRAFFIC_SIGN_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .detection-stats {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
"""

### Scenario Search Settings ###
SCENARIO_SEARCH_SETTINGS = {
    'model_name': "openai/clip-vit-base-patch32",
    'similarity_threshold': 0.25,
    'sample_rate': 30,
    'max_results': 3,
    'batch_size': 32,
    'temp_dir': 'temp_frames'
}

# Scenario Search CSS
SCENARIO_SEARCH_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .search-results {
        margin-top: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
    }
    .similarity-score {
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
"""



### Voice Transcription Settings ###
VOICE_TRANSCRIPTION_SETTINGS = {
    'model_size': "base",
    'sample_rate': 16000,
    'channels': 1,
    'max_duration': 300,  # 5 minutes
    'default_duration': 30,
    'recordings_dir': 'recordings'
}

# Voice Transcription CSS
VOICE_TRANSCRIPTION_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .transcription-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .segment {
        padding: 0.5rem;
        border-bottom: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
    }
    .timestamp {
        color: #666;
        font-size: 0.9rem;
    }
    .recording-status {
        color: #dc3545;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    </style>
"""
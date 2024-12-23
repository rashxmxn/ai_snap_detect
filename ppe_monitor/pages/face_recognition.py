import streamlit as st
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import os
import pickle
from datetime import datetime
from pathlib import Path
import time

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = "face_db"
ENCODING_FILE = "face_encodings.pkl"

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            keep_all=True,
            device=DEVICE,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Initialize InsightFace for face recognition
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.face_analyzer.prepare(ctx_id=0)
        
        # Create database directory if it doesn't exist
        os.makedirs(DB_PATH, exist_ok=True)
        
        # Load existing face encodings
        self.known_faces = self.load_face_encodings()
        
    def load_face_encodings(self):
        if os.path.exists(os.path.join(DB_PATH, ENCODING_FILE)):
            with open(os.path.join(DB_PATH, ENCODING_FILE), 'rb') as f:
                return pickle.load(f)
        return {}
        
    def save_face_encodings(self):
        with open(os.path.join(DB_PATH, ENCODING_FILE), 'wb') as f:
            pickle.dump(self.known_faces, f)
            
    def add_face(self, frame, name):
        # Detect faces using MTCNN
        boxes, _ = self.mtcnn.detect(frame)
        
        if boxes is not None and len(boxes) > 0:
            # Get face embeddings using InsightFace
            faces = self.face_analyzer.get(frame)
            
            if len(faces) > 0:
                # Get the largest face
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                
                # Save face encoding
                self.known_faces[name] = {
                    'embedding': face.embedding,
                    'timestamp': datetime.now()
                }
                
                # Save encodings to file
                self.save_face_encodings()
                return True
                
        return False
        
    def recognize_faces(self, frame):
        if not self.known_faces:
            return frame, []
            
        # Detect and get embeddings for faces in frame
        faces = self.face_analyzer.get(frame)
        results = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Find closest match
            min_dist = float('inf')
            best_match = None
            
            for name, data in self.known_faces.items():
                dist = np.linalg.norm(embedding - data['embedding'])
                if dist < min_dist:
                    min_dist = dist
                    best_match = name
            
            # Draw bounding box and name if match found
            if min_dist < 1.0:  # Threshold for face recognition
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, best_match, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                results.append({
                    'name': best_match,
                    'confidence': 1 - min_dist,
                    'bbox': bbox
                })
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
        return frame, results

def main():
    st.title("👤 Face Recognition System")
    
    # Initialize face recognition system
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
    
    # Sidebar for adding new faces
    st.sidebar.header("Add New Face")
    new_name = st.sidebar.text_input("Enter name")
    if st.sidebar.button("Capture Face"):
        if not new_name:
            st.sidebar.error("Please enter a name")
        else:
            st.session_state.capturing = True
            st.session_state.capture_name = new_name
    
    # Display registered faces
    st.sidebar.header("Registered Faces")
    for name in st.session_state.face_system.known_faces.keys():
        st.sidebar.text(f"✓ {name}")
    
    if st.sidebar.button("Clear Database"):
        st.session_state.face_system.known_faces.clear()
        st.session_state.face_system.save_face_encodings()
        st.sidebar.success("Database cleared!")
    
    # Main content - Video feed
    st.header("Live Recognition")
    video_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # If capturing new face
            if st.session_state.get('capturing', False):
                success = st.session_state.face_system.add_face(
                    rgb_frame, 
                    st.session_state.capture_name
                )
                if success:
                    st.sidebar.success(f"Face added for {st.session_state.capture_name}")
                    st.session_state.capturing = False
                    del st.session_state.capture_name
            
            # Perform face recognition
            processed_frame, results = st.session_state.face_system.recognize_faces(rgb_frame)
            
            # Display the frame
            video_placeholder.image(processed_frame)
            
            # Control frame rate
            time.sleep(0.01)
            
    finally:
        cap.release()

if __name__ == "__main__":
    main()
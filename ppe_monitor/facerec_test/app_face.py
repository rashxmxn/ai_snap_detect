import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Dict
import io
import os
import warnings
import json
from datetime import datetime
import base64
import shutil

warnings.filterwarnings('ignore', category=FutureWarning)

class FaceModel(nn.Module):
    """Neural network model using InceptionResnetV1 backbone for face recognition"""
    def __init__(self, device='cpu'):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class FaceDetector:
    """Handles face detection in images using MTCNN"""
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False
        )

    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        """Detects faces in a frame and returns their bounding boxes"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            #frame_rgb.to(device=self.device)
            boxes, _ = self.mtcnn.detect(frame_rgb)
            
            if boxes is None:
                return []
            
            height, width = frame.shape[:2]
            valid_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    valid_boxes.append([x1, y1, x2, y2])
            return valid_boxes
        except Exception as e:
            st.error(f"Face detection error: {str(e)}")
            return []

class FaceRecognizer:
    """Handles face recognition using a pre-trained model"""
    def __init__(self, device='cpu'):
        self.device = device
        self.model = FaceModel(device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_embedding(self, face_img: Image.Image) -> Optional[np.ndarray]:
        """Extracts face embedding from an image"""
        try:
            with torch.no_grad():
                face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                embedding = self.model.backbone(face_tensor)
            return embedding.cpu().numpy()[0]
        except Exception as e:
            st.error(f"Face recognition error: {str(e)}")
            return None

class FaceDatabase:
    """Manages storage and retrieval of face embeddings and images"""
    def __init__(self, db_dir="face_db"):
        self.db_dir = db_dir
        self.embeddings_file = os.path.join(db_dir, "embeddings.json")
        self.images_dir = os.path.join(db_dir, "images")
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Creates necessary directories and files"""
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        if not os.path.exists(self.embeddings_file):
            self._save_embeddings({})
            
    def _save_embeddings(self, embeddings_dict: Dict):
        """Saves embeddings dictionary to JSON file"""
        serializable_dict = {}
        for name, data in embeddings_dict.items():
            serializable_dict[name] = {
                'avg_embedding': data['avg_embedding'].tolist(),
                'image_paths': data['image_paths']
            }
        
        with open(self.embeddings_file, 'w') as f:
            json.dump(serializable_dict, f)
            
    def load_embeddings(self) -> Dict:
        """Loads embeddings dictionary from storage"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                data = json.load(f)
            return {
                name: {
                    'avg_embedding': np.array(data[name]['avg_embedding']),
                    'image_paths': data[name]['image_paths']
                }
                for name in data
            }
        return {}
        
    def add_face(self, name: str, face_img: Image.Image, embedding: np.ndarray):
        """Adds a single face image and its embedding to the database"""
        embeddings_dict = self.load_embeddings()
        
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{name}_{timestamp}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        face_img.save(image_path)
        
        # Update or create entry
        if name in embeddings_dict:
            # Load existing data
            existing_paths = embeddings_dict[name]['image_paths']
            existing_embedding = embeddings_dict[name]['avg_embedding']
            
            # Update average embedding
            n = len(existing_paths)
            new_avg_embedding = (n * existing_embedding + embedding) / (n + 1)
            
            embeddings_dict[name] = {
                'avg_embedding': new_avg_embedding,
                'image_paths': existing_paths + [image_filename]
            }
        else:
            embeddings_dict[name] = {
                'avg_embedding': embedding,
                'image_paths': [image_filename]
            }
        
        # Save updated embeddings
        self._save_embeddings(embeddings_dict)
        return len(embeddings_dict[name]['image_paths'])

    def delete_user(self, name: str) -> bool:
        """
        Deletes a user from the database, including their embeddings and images.
        Returns True if successful, False otherwise.
        """
        try:
            # Load current embeddings
            embeddings_dict = self.load_embeddings()
            
            # Check if user exists
            if name not in embeddings_dict:
                return False
            
            # Delete user's images
            for image_path in embeddings_dict[name]['image_paths']:
                full_path = os.path.join(self.images_dir, image_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
            
            # Remove user from embeddings dictionary
            del embeddings_dict[name]
            
            # Save updated embeddings
            self._save_embeddings(embeddings_dict)
            
            return True
        except Exception as e:
            st.error(f"Error deleting user: {str(e)}")
            return False
        

def find_best_match(embedding: np.ndarray, face_db: Dict, threshold: float = 0.85) -> Tuple[str, float]:
    """Finds the best matching face in the database"""
    best_match = "Unknown"
    max_similarity = -float('inf')
    
    for name, data in face_db.items():
        similarity = embedding @ data['avg_embedding'].T
        if similarity > max_similarity and similarity > threshold:
            max_similarity = similarity
            best_match = name
            
    return best_match, max_similarity

def main():
    st.title("Real-time Face Recognition with User Management")
    
    # Initialize components
    if 'detector' not in st.session_state:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.session_state.detector = FaceDetector(device=device)
        st.session_state.recognizer = FaceRecognizer(device=device)
        st.session_state.face_db = FaceDatabase()
        st.session_state.collecting = False
        st.session_state.target_name = ""
    
    # Database management interface
    st.sidebar.header("Face Database Management")
    
    # User Registration Section
    st.sidebar.subheader("Register New User")
    collection_mode = st.sidebar.checkbox("Enable Registration Mode")
    if collection_mode:
        name = st.sidebar.text_input("Person's Name")
        start_collection = st.sidebar.button("Register Face")
        
        if start_collection and name:
            st.session_state.collecting = True
            st.session_state.target_name = name
    
    # User Deletion Section
    st.sidebar.subheader("Delete User")
    embeddings_dict = st.session_state.face_db.load_embeddings()
    if embeddings_dict:
        user_to_delete = st.sidebar.selectbox(
            "Select user to delete",
            options=list(embeddings_dict.keys())
        )
        if st.sidebar.button("Delete Selected User"):
            if st.session_state.face_db.delete_user(user_to_delete):
                st.sidebar.success(f"Successfully deleted {user_to_delete}")
            else:
                st.sidebar.error(f"Failed to delete {user_to_delete}")
    
    # Display database statistics
    st.sidebar.subheader("Database Statistics")
    embeddings_dict = st.session_state.face_db.load_embeddings()  # Refresh after potential deletion
    for name, data in embeddings_dict.items():
        st.sidebar.write(f"{name}: {len(data['image_paths'])} image")
    
    # Real-time recognition
    cap = cv2.VideoCapture(0)
    if cap is None:
        st.error("No camera detected. Please check your camera connection.")
        return
    
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    try:
        while not stop_button and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame")
                break

            faces = st.session_state.detector.detect_faces(frame)

            for box in faces:
                try:
                    face_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(box)
                    embedding = st.session_state.recognizer.get_embedding(face_img)
                    
                    if embedding is not None:
                        # Registration mode - collect single image
                        if st.session_state.collecting:
                            st.session_state.face_db.add_face(
                                st.session_state.target_name, face_img, embedding)
                            st.session_state.collecting = False
                            st.success(f"Successfully registered {st.session_state.target_name}")
                            
                            # Draw registration confirmation
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, 
                                      f"Registered {st.session_state.target_name}",
                                      (box[0], box[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Recognition mode
                        else:
                            best_match, confidence = find_best_match(
                                embedding, st.session_state.face_db.load_embeddings())
                            
                            # Draw recognition results
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, 
                                      f"{best_match} ({confidence:.2f})",
                                      (box[0], box[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    continue
            
            frame_placeholder.image(frame, channels="BGR")
            
    except Exception as e:
        st.error(f"Stream error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()


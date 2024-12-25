# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# import numpy as np
# from typing import Tuple, List, Optional
# import io
# import os
# import warnings
# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# import numpy as np
# from typing import Tuple, List, Optional, Dict
# import io
# import os
# import warnings
# import json
# from datetime import datetime
# import base64

# warnings.filterwarnings('ignore', category=FutureWarning)

# class ArcFaceModel(nn.Module):
#     def __init__(self, embedding_size=512, num_classes=1000):
#         """
#         Initialize ArcFace model with InceptionResnetV1 backbone.
#         """
#         super().__init__()
#         # Suppress warnings during model loading
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
#         self.fc = nn.Linear(512, num_classes)
        
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.fc(x)
#         return x

# class FaceRecognizer:
#     def __init__(self, model_path=None, device='cuda'):
#         """
#         Initialize face recognizer with optional pre-trained weights.
#         """
#         self.device = device
#         self.model = ArcFaceModel().to(device)
#         if model_path and os.path.exists(model_path):
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 self.model.load_state_dict(torch.load(model_path))
#         self.model.eval()
        
#         # Preprocessing transforms for face images
#         self.transform = transforms.Compose([
#             transforms.Resize((160, 160)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
    
#     def get_embedding(self, face_img: Image.Image) -> Optional[np.ndarray]:
#         """
#         Extract face embedding from an image using the model.
#         Returns None if extraction fails.
#         """
#         try:
#             with torch.no_grad():
#                 face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
#                 embedding = self.model.backbone(face_tensor)
#             return embedding.cpu().numpy()
#         except Exception as e:
#             st.error(f"Face recognition error: {str(e)}")
#             return None



# warnings.filterwarnings('ignore', category=FutureWarning)

# class FaceDatabase:
#     """
#     Manages the storage and retrieval of face embeddings and images.
#     Provides functionality to save multiple images per person and their corresponding embeddings.
#     """
#     def __init__(self, db_dir="face_db"):
#         self.db_dir = db_dir
#         self.embeddings_file = os.path.join(db_dir, "embeddings.json")
#         self.images_dir = os.path.join(db_dir, "images")
#         self._initialize_storage()
        
#     def _initialize_storage(self):
#         """Creates necessary directories and files if they don't exist."""
#         os.makedirs(self.db_dir, exist_ok=True)
#         os.makedirs(self.images_dir, exist_ok=True)
#         if not os.path.exists(self.embeddings_file):
#             self._save_embeddings({})
            
#     def _save_embeddings(self, embeddings_dict: Dict):
#         """Saves embeddings dictionary to JSON file."""
#         # Convert numpy arrays to lists for JSON serialization
#         serializable_dict = {}
#         for name, data in embeddings_dict.items():
#             serializable_dict[name] = {
#                 'embeddings': [emb.tolist() for emb in data['embeddings']],
#                 'image_paths': data['image_paths']
#             }
        
#         with open(self.embeddings_file, 'w') as f:
#             json.dump(serializable_dict, f)
            
#     def load_embeddings(self) -> Dict:
#         """Loads and returns embeddings dictionary from storage."""
#         if os.path.exists(self.embeddings_file):
#             with open(self.embeddings_file, 'r') as f:
#                 data = json.load(f)
#             # Convert lists back to numpy arrays
#             return {
#                 name: {
#                     'embeddings': [np.array(emb) for emb in data[name]['embeddings']],
#                     'image_paths': data[name]['image_paths']
#                 }
#                 for name in data
#             }
#         return {}
        
#     def add_face(self, name: str, face_img: Image.Image, embedding: np.ndarray):
#         """Adds a new face image and its embedding to the database."""
#         # Load existing embeddings
#         embeddings_dict = self.load_embeddings()
        
#         # Create entry for new person if doesn't exist
#         if name not in embeddings_dict:
#             embeddings_dict[name] = {'embeddings': [], 'image_paths': []}
        
#         # Save image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         image_filename = f"{name}_{timestamp}.jpg"
#         image_path = os.path.join(self.images_dir, image_filename)
#         face_img.save(image_path)
        
#         # Update embeddings dictionary
#         embeddings_dict[name]['embeddings'].append(embedding)
#         embeddings_dict[name]['image_paths'].append(image_filename)
        
#         # Save updated embeddings
#         self._save_embeddings(embeddings_dict)
        
#         return len(embeddings_dict[name]['embeddings'])

# class FaceDetector:
#     def __init__(self, device='cuda'):
#         self.device = device
#         self.mtcnn = MTCNN(
#             keep_all=True,
#             device=device,
#             post_process=False
#         )

#     def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
#         try:
#             if len(frame.shape) == 3 and frame.shape[2] == 3:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             else:
#                 frame_rgb = frame
                
#             boxes, _ = self.mtcnn.detect(frame_rgb)
#             if boxes is None:
#                 return []
            
#             height, width = frame.shape[:2]
#             valid_boxes = []
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(width, x2), min(height, y2)
#                 if x2 > x1 and y2 > y1:
#                     valid_boxes.append([x1, y1, x2, y2])
#             return valid_boxes
#         except Exception as e:
#             st.error(f"Face detection error: {str(e)}")
#             return []

# class FaceRecognizer:
#     def __init__(self, model_path=None, device='cuda'):
#         self.device = device
#         self.model = ArcFaceModel().to(device)
#         if model_path and os.path.exists(model_path):
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 self.model.load_state_dict(torch.load(model_path))
#         self.model.eval()
        
#         self.transform = transforms.Compose([
#             transforms.Resize((160, 160)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
    
#     def get_embedding(self, face_img: Image.Image) -> Optional[np.ndarray]:
#         try:
#             with torch.no_grad():
#                 face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
#                 embedding = self.model.backbone(face_tensor)
#             return embedding.cpu().numpy()[0]  # Return flattened embedding
#         except Exception as e:
#             st.error(f"Face recognition error: {str(e)}")
#             return None

# def initialize_camera():
#     for i in range(2):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             cap.set(cv2.CAP_PROP_FPS, 30)
#             return cap, i
#     return None, None

# def find_best_match(embedding: np.ndarray, face_db: Dict, threshold: float = 0.6) -> Tuple[str, float]:
#     """
#     Finds the best matching face in the database using average distance to all embeddings of each person.
#     Returns the name and confidence score of the best match.
#     """
#     best_match = "Unknown"
#     min_avg_dist = float('inf')
    
#     for name, data in face_db.items():
#         distances = [np.linalg.norm(embedding - stored_emb) for stored_emb in data['embeddings']]
#         avg_dist = np.mean(distances)
#         if avg_dist < min_avg_dist and avg_dist < threshold:
#             min_avg_dist = avg_dist
#             best_match = name
            
#     return best_match, min_avg_dist

# def main():
#     st.title("Real-time Face Recognition with Database Collection")
    
#     # Initialize components
#     if 'detector' not in st.session_state:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         st.session_state.detector = FaceDetector(device=device)
#         st.session_state.recognizer = FaceRecognizer(device=device)
#         st.session_state.face_db = FaceDatabase()
#         st.session_state.collecting = False
#         st.session_state.collect_count = 0
#         st.session_state.target_name = ""
    
#     # Database collection interface
#     st.sidebar.header("Face Database Management")
    
#     # Registration mode
#     collection_mode = st.sidebar.checkbox("Enable Collection Mode")
#     if collection_mode:
#         name = st.sidebar.text_input("Person's Name")
#         target_images = st.sidebar.number_input("Number of images to collect", 
#                                               min_value=1, max_value=20, value=5)
#         start_collection = st.sidebar.button("Start Collection")
        
#         if start_collection and name:
#             st.session_state.collecting = True
#             st.session_state.collect_count = 0
#             st.session_state.target_count = target_images
#             st.session_state.target_name = name
    
#     # Display database statistics
#     embeddings_dict = st.session_state.face_db.load_embeddings()
#     st.sidebar.markdown("### Database Statistics")
#     for name, data in embeddings_dict.items():
#         st.sidebar.write(f"{name}: {len(data['embeddings'])} images")
    
#     # Real-time recognition
#     cap, camera_index = initialize_camera()
#     if cap is None:
#         st.error("No camera detected. Please check your camera connection.")
#         return
    
#     st.info(f"Using camera index: {camera_index}")
#     frame_placeholder = st.empty()
#     stop_button = st.button("Stop")

#     try:
#         while not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture video frame")
#                 break

#             faces = st.session_state.detector.detect_faces(frame)
            
#             for box in faces:
#                 try:
#                     face_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(box)
#                     embedding = st.session_state.recognizer.get_embedding(face_img)
                    
#                     if embedding is not None:
#                         # Collection mode
#                         if st.session_state.collecting and \
#                            st.session_state.collect_count < st.session_state.target_count:
#                             # Add face to database
#                             count = st.session_state.face_db.add_face(
#                                 st.session_state.target_name, face_img, embedding)
#                             st.session_state.collect_count += 1
                            
#                             # Draw collection progress
#                             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                             cv2.putText(frame, 
#                                       f"Collecting {st.session_state.collect_count}/{st.session_state.target_count}",
#                                       (box[0], box[1]-10),
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
#                             if st.session_state.collect_count >= st.session_state.target_count:
#                                 st.session_state.collecting = False
#                                 st.success(f"Collection complete for {st.session_state.target_name}")
                        
#                         # Recognition mode
#                         else:
#                             # Find best match from database
#                             best_match, confidence = find_best_match(
#                                 embedding, st.session_state.face_db.load_embeddings())
                            
#                             # Draw results
#                             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                             cv2.putText(frame, 
#                                       f"{best_match} ({confidence:.2f})",
#                                       (box[0], box[1]-10),
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 except Exception as e:
#                     continue
            
#             frame_placeholder.image(frame, channels="BGR")
            
#     except Exception as e:
#         st.error(f"Stream error: {str(e)}")
#     finally:
#         cap.release()

# if __name__ == "__main__":
#     main()



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
import shutil  # Added for directory removal

warnings.filterwarnings('ignore', category=FutureWarning)

class ArcFaceModel(nn.Module):
    """Neural network model using InceptionResnetV1 backbone with ArcFace head"""
    def __init__(self, embedding_size=512, num_classes=1000):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

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
                'embeddings': [emb.tolist() for emb in data['embeddings']],
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
                    'embeddings': [np.array(emb) for emb in data[name]['embeddings']],
                    'image_paths': data[name]['image_paths']
                }
                for name in data
            }
        return {}
        
    def add_face(self, name: str, face_img: Image.Image, embedding: np.ndarray):
        """Adds a single face image and its embedding to the database"""
        embeddings_dict = self.load_embeddings()
        
        # Replace existing entry if name exists, otherwise create new
        embeddings_dict[name] = {
            'embeddings': [embedding],  # Store only the latest embedding
            'image_paths': []  # Initialize empty image paths list
        }
        
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{name}_{timestamp}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        face_img.save(image_path)
        
        # Update image paths
        embeddings_dict[name]['image_paths'] = [image_filename]
        
        # Save updated embeddings
        self._save_embeddings(embeddings_dict)
        return 1  # Return 1 since we only store one image now

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

class FaceDetector:
    """Handles face detection in images using MTCNN"""
    def __init__(self, device='cuda'):
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
    """Handles face recognition using ArcFace model"""
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = ArcFaceModel().to(device)
        if model_path and os.path.exists(model_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
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

def initialize_camera():
    """Initializes and configures the camera"""
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap, i
    return None, None

def find_best_match(embedding: np.ndarray, face_db: Dict, threshold: float = 0.6) -> Tuple[str, float]:
    """Finds the best matching face in the database"""
    best_match = "Unknown"
    min_avg_dist = float('inf')
    
    for name, data in face_db.items():
        distances = [np.linalg.norm(embedding - stored_emb) for stored_emb in data['embeddings']]
        avg_dist = np.mean(distances)
        if avg_dist < min_avg_dist and avg_dist < threshold:
            min_avg_dist = avg_dist
            best_match = name
            
    return best_match, min_avg_dist

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
        st.sidebar.write(f"{name}: {len(data['embeddings'])} image")
    
    # Real-time recognition
    cap, camera_index = initialize_camera()
    if cap is None:
        st.error("No camera detected. Please check your camera connection.")
        return
    
    st.info(f"Using camera index: {camera_index}")
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    try:
        while not stop_button:
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
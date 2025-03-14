import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import warnings
import cv2
import numpy as np
from typing import List
from facenet_pytorch import MTCNN
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Tuple, Dict



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

    def get_backbone(self):
        """Returns the backbone model for direct use"""
        return self.backbone
    





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
        """
        Detects faces in a frame and returns their bounding boxes.
        
        Args:
            frame: Input image frame in numpy array format
            
        Returns:
            List of bounding boxes in format [x1, y1, x2, y2]
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) \
                if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
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
        """
        Extracts face embedding from an image.
        
        Args:
            face_img: PIL Image containing a face
            
        Returns:
            numpy array containing face embedding or None if extraction fails
        """
        try:
            with torch.no_grad():
                face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                embedding = self.model.backbone(face_tensor)
            return embedding.cpu().numpy()[0]
        except Exception as e:
            st.error(f"Face recognition error: {str(e)}")
            return None

    def find_best_match(self, embedding: np.ndarray, face_db: Dict, threshold: float = 0.85) -> Tuple[str, float]:
        """
        Finds the best matching face in the database.
        
        Args:
            embedding: Face embedding to match
            face_db: Dictionary containing stored face embeddings
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (best matching name, similarity score)
        """
        best_match = "Unknown"
        max_similarity = -float('inf')
        
        for name, data in face_db.items():
            similarity = embedding @ data['avg_embedding'].T
            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_match = name
                
        return best_match, max_similarity
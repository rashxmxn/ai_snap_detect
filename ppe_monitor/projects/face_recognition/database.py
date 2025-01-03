import os
import json
import numpy as np
from datetime import datetime
from typing import Dict
from PIL import Image
import streamlit as st

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
        """
        Saves embeddings dictionary to JSON file.
        
        Args:
            embeddings_dict: Dictionary containing embeddings data
        """
        serializable_dict = {}
        for name, data in embeddings_dict.items():
            serializable_dict[name] = {
                'avg_embedding': data['avg_embedding'].tolist(),
                'image_paths': data['image_paths']
            }
        
        with open(self.embeddings_file, 'w') as f:
            json.dump(serializable_dict, f)
            
    def load_embeddings(self) -> Dict:
        """
        Loads embeddings dictionary from storage.
        
        Returns:
            Dictionary containing stored embeddings data
        """
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
        
    def add_face(self, name: str, face_img: Image.Image, embedding: np.ndarray) -> int:
        """
        Adds a single face image and its embedding to the database.
        
        Args:
            name: Person's name
            face_img: PIL Image of the face
            embedding: Face embedding array
            
        Returns:
            Number of images stored for this person
        """
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
        
        Args:
            name: Name of the person to delete
            
        Returns:
            True if successful, False otherwise
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
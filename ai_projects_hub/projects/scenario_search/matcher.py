# import cv2
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import streamlit as st
# from typing import List, Dict, Tuple, Any
# from config.settings import SCENARIO_SEARCH_SETTINGS

# class VideoFrameMatcher:
#     """Handles video frame matching using CLIP model"""
    
#     def __init__(self):
#         """Initialize the CLIP model and processor"""
#         self.model_name = SCENARIO_SEARCH_SETTINGS['model_name']
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         with st.spinner("Loading CLIP model..."):
#             self.model = CLIPModel.from_pretrained(self.model_name)
#             self.processor = CLIPProcessor.from_pretrained(self.model_name)
#             self.model.to(self.device)

#     def _extract_frames_impl(self, video_path: str, sample_rate: int = 1) -> Tuple[List[Image.Image], List[float]]:
#         """
#         Internal implementation of frame extraction
#         """
#         frames = []
#         timestamps = []
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = 0
        
#         try:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                    
#                 if frame_count % sample_rate == 0:
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame_pil = Image.fromarray(frame_rgb)
#                     frames.append(frame_pil)
#                     timestamps.append(frame_count / fps)
                    
#                 frame_count += 1
                
#         finally:
#             cap.release()
            
#         return frames, timestamps

#     @st.cache_data
#     def extract_frames(_self, video_path: str, sample_rate: int = 1) -> Tuple[List[Image.Image], List[float]]:
#         """
#         Extract frames from video at given sample rate. This is a cached wrapper around _extract_frames_impl.
        
#         Args:
#             video_path: Path to video file
#             sample_rate: Number of frames to skip between samples
            
#         Returns:
#             Tuple of (list of frames, list of timestamps)
#         """
#         # Create a new instance for thread safety
#         matcher = VideoFrameMatcher()
#         return matcher._extract_frames_impl(video_path, sample_rate)

#     def compute_similarity(self, frames: List[Image.Image], text_query: str) -> np.ndarray:
#         """
#         Compute similarity scores between frames and text query
        
#         Args:
#             frames: List of PIL Image frames
#             text_query: Text query to match against frames
            
#         Returns:
#             Array of similarity scores
#         """
#         similarities = []
#         batch_size = SCENARIO_SEARCH_SETTINGS['batch_size']

#         # Process text input
#         text_inputs = self.processor(
#             text=text_query,
#             return_tensors="pt",
#             padding=True
#         ).to(self.device)

#         # Process frames in batches
#         for i in range(0, len(frames), batch_size):
#             batch_frames = frames[i:i + batch_size]
#             image_inputs = self.processor(
#                 images=batch_frames,
#                 return_tensors="pt",
#                 padding=True
#             ).to(self.device)

#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**image_inputs)
#                 text_features = self.model.get_text_features(**text_inputs)
                
#                 # Normalize features
#                 image_features = image_features / image_features.norm(dim=1, keepdim=True)
#                 text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
#                 # Compute similarity
#                 similarity = (100.0 * image_features @ text_features.T).squeeze()
#                 similarities.extend(similarity.cpu().numpy())

#         return np.array(similarities)

#     def find_relevant_frames(
#         self, 
#         video_path: str, 
#         text_query: str, 
#         threshold: float = None,
#         sample_rate: int = None,
#         max_results: int = None
#     ) -> Dict[str, List]:
#         """
#         Find frames relevant to the text query
        
#         Args:
#             video_path: Path to video file
#             text_query: Text query to match against frames
#             threshold: Minimum similarity score (default from settings)
#             sample_rate: Number of frames to skip (default from settings)
#             max_results: Maximum number of results to return (default from settings)
            
#         Returns:
#             Dictionary containing frames, timestamps, and scores
#         """
#         # Use default settings if not specified
#         threshold = threshold or SCENARIO_SEARCH_SETTINGS['similarity_threshold']
#         sample_rate = sample_rate or SCENARIO_SEARCH_SETTINGS['sample_rate']
#         max_results = max_results or SCENARIO_SEARCH_SETTINGS['max_results']
        
#         # Extract and compute similarities
#         frames, timestamps = self.extract_frames(video_path, sample_rate)
#         similarities = self.compute_similarity(frames, text_query)
        
#         # Find frames above threshold
#         relevant_indices = np.where(similarities > threshold)[0]
#         relevant_frames = [frames[i] for i in relevant_indices]
#         relevant_timestamps = [timestamps[i] for i in relevant_indices]
#         relevant_scores = similarities[relevant_indices]
        
#         # Sort by similarity score and limit results
#         sorted_indices = np.argsort(-relevant_scores)[:max_results]
        
#         return {
#             'frames': [relevant_frames[i] for i in sorted_indices],
#             'timestamps': [relevant_timestamps[i] for i in sorted_indices],
#             'scores': [relevant_scores[i] for i in sorted_indices]
#         }

import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
from pathlib import Path
import streamlit as st
from typing import List, Dict, Tuple, Any
from config.settings import SCENARIO_SEARCH_SETTINGS

class VideoFrameMatcher:
    """Handles video frame matching using CLIP model"""
    
    def __init__(self):
        """Initialize the CLIP model and processor"""
        self.model_name = SCENARIO_SEARCH_SETTINGS['model_name']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner("Loading CLIP model..."):
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)

    def _extract_frames_impl(self, video_path: str, sample_rate: int = 1) -> Tuple[List[Image.Image], List[float]]:
        """
        Internal implementation of frame extraction
        """
        frames = []
        timestamps = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frames.append(frame_pil)
                    timestamps.append(frame_count / fps)
                    
                frame_count += 1
                
        finally:
            cap.release()
            
        return frames, timestamps

    @st.cache_data
    def extract_frames(_self, video_path: str, sample_rate: int = 1) -> Tuple[List[Image.Image], List[float]]:
        """
        Extract frames from video at given sample rate. This is a cached wrapper around _extract_frames_impl.
        
        Args:
            video_path: Path to video file
            sample_rate: Number of frames to skip between samples
            
        Returns:
            Tuple of (list of frames, list of timestamps)
        """
        # Create a new instance for thread safety
        matcher = VideoFrameMatcher()
        return matcher._extract_frames_impl(video_path, sample_rate)

    def compute_similarity(self, frames: List[Image.Image], text_query: str) -> np.ndarray:
        """
        Compute similarity scores between frames and text query
        
        Args:
            frames: List of PIL Image frames
            text_query: Text query to match against frames
            
        Returns:
            Array of similarity scores
        """
        # Handle empty frames case
        if not frames:
            return np.array([])
            
        similarities = []
        batch_size = SCENARIO_SEARCH_SETTINGS['batch_size']

        # Process text input
        text_inputs = self.processor(
            text=text_query,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            image_inputs = self.processor(
                images=batch_frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Compute similarity (handle scalar case properly)
                similarity = (100.0 * image_features @ text_features.T)
                
                # Handle different dimensions properly
                if similarity.dim() == 0:  # It's a scalar tensor (0-d)
                    # Convert scalar to 1-element list
                    similarities.append(similarity.item())
                elif similarity.dim() == 1:  # It's a 1d tensor
                    # Convert to list properly
                    similarities.extend(similarity.cpu().numpy().tolist())
                else:
                    # For 2D tensors (batch × 1), flatten appropriately
                    similarity = similarity.squeeze(-1)  # Be specific about which dim to squeeze
                    similarities.extend(similarity.cpu().numpy().tolist())

        return np.array(similarities)

    def find_relevant_frames(
        self, 
        video_path: str, 
        text_query: str, 
        threshold: float = None,
        sample_rate: int = None,
        max_results: int = None
    ) -> Dict[str, List]:
        """
        Find frames relevant to the text query
        
        Args:
            video_path: Path to video file
            text_query: Text query to match against frames
            threshold: Minimum similarity score (default from settings)
            sample_rate: Number of frames to skip (default from settings)
            max_results: Maximum number of results to return (default from settings)
            
        Returns:
            Dictionary containing frames, timestamps, and scores
        """
        # Use default settings if not specified
        threshold = threshold or SCENARIO_SEARCH_SETTINGS['similarity_threshold']
        sample_rate = sample_rate or SCENARIO_SEARCH_SETTINGS['sample_rate']
        max_results = max_results or SCENARIO_SEARCH_SETTINGS['max_results']
        
        try:
            # Extract and compute similarities
            frames, timestamps = self.extract_frames(video_path, sample_rate)
            
            # Check if we got any frames
            if not frames:
                return {'frames': [], 'timestamps': [], 'scores': []}
                
            similarities = self.compute_similarity(frames, text_query)
            
            # Handle case where similarities could be empty
            if len(similarities) == 0:
                return {'frames': [], 'timestamps': [], 'scores': []}
            
            # Find frames above threshold
            relevant_indices = np.where(similarities > threshold)[0]
            
            # Handle case where no frames match the threshold
            if len(relevant_indices) == 0:
                return {'frames': [], 'timestamps': [], 'scores': []}
                
            relevant_frames = [frames[i] for i in relevant_indices]
            relevant_timestamps = [timestamps[i] for i in relevant_indices]
            relevant_scores = similarities[relevant_indices]
            
            # Sort by similarity score and limit results
            sorted_indices = np.argsort(-relevant_scores)[:max_results]
            
            return {
                'frames': [relevant_frames[i] for i in sorted_indices],
                'timestamps': [relevant_timestamps[i] for i in sorted_indices],
                'scores': [relevant_scores[i] for i in sorted_indices]
            }
        except Exception as e:
            # Add better error handling
            st.error(f"Error in frame matching: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return {'frames': [], 'timestamps': [], 'scores': [], 'error': str(e)}
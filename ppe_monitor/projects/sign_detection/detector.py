import cv2
import supervision as sv
from inference import get_model
import numpy as np
import streamlit as st
from typing import List, Dict, Any

class TrafficSignDetector:
    """Handles traffic sign detection in images and video frames"""
    
    def __init__(self, model_id: str, confidence_threshold: float = 0.5):
        """
        Initialize the traffic sign detector
        
        Args:
            model_id: ID of the model to use for detection
            confidence_threshold: Minimum confidence score for detections
        """
        self.model = get_model(model_id=model_id)
        self.confidence_threshold = confidence_threshold
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Initialize detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'detections_by_class': {}
        }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for traffic sign detection
        
        Args:
            frame: Input frame in numpy array format
            
        Returns:
            Annotated frame with detections
        """
        try:
            # Run inference
            results = self.model.infer(frame)[0]
            
            # Convert results to supervision Detections
            detections = sv.Detections.from_inference(results)
            
            # Filter detections based on confidence
            mask = detections.confidence >= self.confidence_threshold
            detections = detections[mask]
            
            # Update statistics
            self._update_stats(detections)
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, detections)
            
            return annotated_frame
            
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return frame
    
    def _update_stats(self, detections: sv.Detections):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(detections)
        
        # Update per-class statistics
        for label in detections.class_id:
            class_name = str(label)  # Convert class ID to name if needed
            if class_name not in self.detection_stats['detections_by_class']:
                self.detection_stats['detections_by_class'][class_name] = 0
            self.detection_stats['detections_by_class'][class_name] += 1
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Add bounding boxes and labels to the frame"""
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        return annotated_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current detection statistics"""
        return self.detection_stats
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'detections_by_class': {}
        }
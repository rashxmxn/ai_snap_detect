import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
from pathlib import Path

class VideoFrameMatcher:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_frames(self, video_path, sample_rate=1):
        """Extract frames from video at given sample rate"""
        frames = []
        timestamps = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
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
            
        cap.release()
        return frames, timestamps

    def compute_similarity(self, frames, text_query):
        """Compute similarity scores between frames and text query"""
        similarities = []
        batch_size = 32

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
                
                # Compute similarity
                similarity = (100.0 * image_features @ text_features.T).squeeze()
                similarities.extend(similarity.cpu().numpy())

        return np.array(similarities)

    def find_relevant_frames(self, video_path, text_query, threshold=0.25, sample_rate=1):
        """Find frames relevant to the text query"""
        frames, timestamps = self.extract_frames(video_path, sample_rate)
        similarities = self.compute_similarity(frames, text_query)
        
        # Find frames above threshold
        relevant_indices = np.where(similarities > threshold)[0]
        relevant_frames = [frames[i] for i in relevant_indices]
        relevant_timestamps = [timestamps[i] for i in relevant_indices]
        relevant_scores = similarities[relevant_indices]
        
        # Sort by similarity score
        sorted_indices = np.argsort(-relevant_scores)
        return {
            'frames': [relevant_frames[i] for i in sorted_indices],
            'timestamps': [relevant_timestamps[i] for i in sorted_indices],
            'scores': [relevant_scores[i] for i in sorted_indices]
        }

def main():
    # Example usage
    matcher = VideoFrameMatcher()
    video_path = "./videos/memes.mp4"
    text_query = "a green parrot in the cage"
    
    results = matcher.find_relevant_frames(
        video_path=video_path,
        text_query=text_query,
        threshold=0.25,  # Adjust threshold as needed
        sample_rate=30   # Sample every 30 frames
    )
    
    # Save or display results
    for i, (frame, timestamp, score) in enumerate(zip(
        results['frames'], 
        results['timestamps'], 
        results['scores']
    )):
        frame.save(f"./context_aware_screenshots/frame_{i}.jpg")
        #print(f"Frame {i}: timestamp={timestamp:.2f}s, similarity={score:.2f}")

if __name__ == "__main__":
    main()
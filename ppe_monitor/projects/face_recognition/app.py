import streamlit as st
import cv2
from PIL import Image
import torch
from typing import Dict
from ..base_project import BaseProject
from .models import FaceDetector, FaceRecognizer
from .database import FaceDatabase
from config.settings import FACE_RECOGNITION_SETTINGS, FACE_RECOGNITION_CSS

class FaceRecognitionApp(BaseProject):
    """Streamlit interface for face recognition system"""
    def __init__(self):
        super().__init__()
        self.title = "Real-time Face Recognition with User Management"
        self.description = """
        Real-time face recognition system with user registration and management capabilities.
        Register new users, delete existing users, and perform real-time face recognition.
        """
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize face recognition components"""
        if 'detector' not in st.session_state or 'recognizer' not in st.session_state:
            self.load_model()
            
    def load_model(self):
        """Initialize face recognition components - Implementation of abstract method"""
        device = FACE_RECOGNITION_SETTINGS['device']
        st.session_state.detector = FaceDetector(device=device)
        st.session_state.recognizer = FaceRecognizer(device=device)
        st.session_state.face_db = FaceDatabase(FACE_RECOGNITION_SETTINGS['db_dir'])
        st.session_state.collecting = False
        st.session_state.target_name = ""

    def setup_sidebar(self):
        """Configure sidebar for user management"""
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
        embeddings_dict = st.session_state.face_db.load_embeddings()
        for name, data in embeddings_dict.items():
            st.sidebar.write(f"{name}: {len(data['image_paths'])} images")

    def process_input(self, frame):
        """Process a single frame for face detection and recognition"""
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
                        frame = self._draw_box(frame, box, 
                                            f"Registered {st.session_state.target_name}")
                    
                    # Recognition mode
                    else:
                        best_match, confidence = st.session_state.recognizer.find_best_match(
                            embedding, 
                            st.session_state.face_db.load_embeddings(),
                            FACE_RECOGNITION_SETTINGS['similarity_threshold']
                        )
                        
                        # Draw recognition results
                        frame = self._draw_box(frame, box, f"{best_match} ({confidence:.2f})")
            except Exception as e:
                continue
        
        return frame

    def _draw_box(self, frame, box, label):
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def display_output(self, frame_placeholder, processed_frame):
        """Display image output in Streamlit placeholder"""
        frame_placeholder.image(processed_frame, channels="BGR")

    def run(self):
        """Main method to run the face recognition interface"""
        # Apply custom CSS
        st.markdown(FACE_RECOGNITION_CSS, unsafe_allow_html=True)
        # Show project info
        self.show_project_info(self.title, self.description)
        # Setup sidebar
        self.setup_sidebar()
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if cap is None:
            st.error("No camera detected. Please check your camera connection.")
            return
        
        # Create placeholders
        frame_placeholder = st.empty()
        stop_button = st.button("Stop")

        try:
            while not stop_button and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame")
                    break

                # Process and display frame
                processed_frame = self.process_input(frame)
                self.display_output(frame_placeholder, processed_frame)
                
                
        except Exception as e:
            st.error(f"Stream error: {str(e)}")
        finally:
            cap.release()
import streamlit as st
from projects.ppe_detection.app import PPEDetectionApp
from projects.face_recognition.app import FaceRecognitionApp
from projects.sign_detection.app import TrafficSignApp
from projects.scenario_search.app import ScenarioSearchApp

# Set page config - Must be the first Streamlit command
st.set_page_config(
    page_title="CV Projects Hub",
    page_icon="🔍",
    layout="wide"
)

def setup_home_page():
    """Setup the home page layout and content"""
    # Create a container for the home page content
    home_container = st.container()
    
    with home_container:
        # Custom CSS
        st.markdown("""
            <style>
            .stButton > button {
                width: 100%;
                height: 50px;
                margin-top: 20px;
            }
            .stButton > button:hover {
                border-color: rgb(255, 75, 75);
                color: rgb(255, 75, 75);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            h1 {
                color: rgb(60, 60, 60);
            }
            h3 {
                padding-top: 1rem;
                color: rgb(60, 60, 60);
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("Computer Vision Projects Hub")
        st.markdown("""
        ### Welcome to the CV Projects Hub! 👋
        
        This application showcases various computer vision projects. Select a project below to get started.
        """)
        
        # Project cards using columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            ### 🎥 Smart Safety Monitor
            Real-time PPE (Personal Protective Equipment) detection system for workplace safety.
            Features:
            - Real-time PPE detection
            - Violation tracking
            - Automated alerts
            """)
            if st.button("Launch Safety Monitor"):
                st.session_state.current_project = "PPE Detection"
                st.rerun()

        with col2:
            st.markdown("""
            ### 👤 Face Recognition System
            Real-time face recognition system with user management.
            Features:
            - Real-time face detection
            - User registration
            - Identity verification
            """)
            if st.button("Launch Face Recognition"):
                st.session_state.current_project = "Face Recognition"
                st.rerun()
        
        with col3:
            st.markdown("""
            ### 🚦 Traffic Sign Detection
            Real-time traffic sign detection and classification system.
            Features:
            - Sign detection & classification
            - Detection statistics
            - Support for video files and webcam
            """)
            if st.button("Launch Traffic Sign Detection"):
                st.session_state.current_project = "Traffic Sign Detection"
                st.rerun()
        
        with col4:
            st.markdown("""
            ### 👤 Smart Scenario Searcher
            Real-time video scenario search using natural language descriptions.
            Features:
            - Natural language search
            - Scene matching
            - Video annotation
            """)
            if st.button("Launch Scenario Searcher"):
                st.session_state.current_project = "Scenario Searcher"
                st.rerun()

def render_project_page():
    """Render the selected project page"""
    # Create a container for the project content
    project_container = st.container()
    
    # Clear any existing content
    project_container.empty()
    
    with project_container:
        # Add a return to home button in the sidebar
        if st.sidebar.button("← Return to Home"):
            st.session_state.current_project = None
            st.rerun()
        
        # Initialize and run the selected project
        project_classes = {
            "PPE Detection": PPEDetectionApp,
            "Face Recognition": FaceRecognitionApp,
            "Traffic Sign Detection": TrafficSignApp,
            "Scenario Searcher": ScenarioSearchApp
        }
        
        project_class = project_classes.get(st.session_state.current_project)
        if project_class:
            project = project_class()
            project.run()

def main():
    # Initialize session state for project selection if not exists
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    
    # Create main container
    main_container = st.empty()
    
    with main_container:
        # Render either home page or selected project
        if st.session_state.current_project is None:
            setup_home_page()
        else:
            # Clear the main container before rendering project
            main_container.empty()
            render_project_page()

if __name__ == "__main__":
    main()
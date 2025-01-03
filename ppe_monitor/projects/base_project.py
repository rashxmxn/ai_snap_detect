from abc import ABC, abstractmethod
import streamlit as st

class BaseProject(ABC):
    def __init__(self):
        self.model = None
        
    @abstractmethod
    def setup_sidebar(self):
        """Set up project-specific sidebar configuration"""
        pass
        
    @abstractmethod
    def load_model(self):
        """Load the project's model."""
        pass
    
    @abstractmethod
    def process_input(self, input_data):
        """Process the input data."""
        pass
    
    @abstractmethod
    def display_output(self, results):
        """Display the results in the Streamlit interface."""
        pass
    
    def show_project_info(self, title, description):
        """Display project information."""
        st.header(title)
        st.markdown(description)
    
    def clear_sidebar(self):
        """Clear existing sidebar elements"""
        for key in list(st.session_state.keys()):
            if key.startswith(self.__class__.__name__):
                del st.session_state[key]
    
    @abstractmethod
    def run(self):
        """Main method to run the project's Streamlit interface."""
        pass

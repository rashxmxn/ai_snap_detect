FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy the application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV ROBOFLOW_API_KEY=Twdup3qP8sok8DPVIx0T

# Command to run the application
CMD ["streamlit", "run", "main_app.py", "--server.address", "0.0.0.0"]


# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# # Avoid prompts from apt
# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     python3.10-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# # Copy the application code
# COPY . .

# # Expose Streamlit port
# EXPOSE 8501

# # Set environment variables
# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# # Command to run the application
# CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]

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
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ngrok
RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz && \
    tar xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin && \
    rm ngrok-v3-stable-linux-amd64.tgz

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

# Create script to run both ngrok and streamlit
RUN echo '#!/bin/bash\n\
if [ ! -z "$NGROK_AUTHTOKEN" ]; then\n\
    ngrok config add-authtoken $NGROK_AUTHTOKEN\n\
    ngrok http 8501 &\n\
fi\n\
streamlit run app.py --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y     libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev     redis-tools ffmpeg     && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements_v2.txt .
RUN pip install --no-cache-dir -r requirements_v2.txt

# Download YOLOv8-Pose model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"

# Copy app
COPY . .

# Create dirs
RUN mkdir -p uploads static templates logs models

EXPOSE 5000

CMD ["python", "app_v2.py"]

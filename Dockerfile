FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs GPU version)
RUN pip install --no-cache-dir \
    "torch==2.1.0" \
    "torchvision==0.16.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads models

EXPOSE $PORT
CMD ["python", "app.py"]
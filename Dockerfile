FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies (Ubuntu 22.04 has python3.10 by default)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 LTS (for yt-dlp YouTube parsing)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000

CMD ["python3", "scheduler.py", "--config", "streams_config.json", "--web-port", "5000"]

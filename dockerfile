FROM python:3.10-slim

# 필수 시스템 패키지 설치 (🔥 핵심은 libgl1-mesa-glx!)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사 및 실행
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]

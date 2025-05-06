FROM python:3.10-slim

# 시스템 패키지 설치 (libGL 포함)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg git && \
    apt-get clean

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY . .

# HuggingFace 로그인용 환경변수 준비
ENV TRANSFORMERS_CACHE=/app/cache

# 실행
CMD ["python", "app.py"]

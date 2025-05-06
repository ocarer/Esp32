FROM python:3.10-slim

# 시스템 패키지 설치 (libGL 포함!)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 코드 복사
COPY . .

# 가상환경 생성 + 의존성 설치
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 환경변수 설정
ENV PATH="/opt/venv/bin:$PATH"

# 앱 실행
CMD ["python", "app.py"]


FROM python:3.10-slim

# 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 코드 복사
COPY . .

# 가상환경 설정 및 의존성 설치
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 환경 변수 설정
ENV PATH="/opt/venv/bin:$PATH"

# 서버 실행
CMD ["python", "app.py"]

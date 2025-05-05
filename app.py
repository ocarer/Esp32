from flask import Flask, request, jsonify
import base64
import os
from huggingface_hub import login
from transformers import pipeline
from espnet2.bin.tts_inference import Text2Speech
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

app = Flask(__name__)

# .env 파일 로드 (환경변수)
load_dotenv()

# Hugging Face 토큰 로그인
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Hugging Face token not found in environment variables.")

# 모델 로드
mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")

tts_model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

yolo_model = YOLO("yolov5s.pt")

# Railway에서 제공하는 포트를 사용
port = os.getenv('PORT', 5000)

@app.route("/nlp", methods=["POST"])
def handle_nlp():
    text = request.json["text"]
    
    # Mistral-7B로 텍스트 처리 (대화)
    inputs = mistral_tokenizer(text, return_tensors="pt")
    outputs = mistral_model.generate(inputs["input_ids"], max_length=150)
    reply = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 간단한 자연어 명령 분석 (예시: '집어', '가져와' 등)
    action = "none"
    if "집어" in text or "가져와" in text:
        action = "grab"
    
    return jsonify({"reply": reply, "action": action})


@app.route("/stt", methods=["POST"])
def handle_stt():
    audio_base64 = request.json["audio"]
    audio_bytes = base64.b64decode(audio_base64)
    
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    # Whisper 모델을 사용하여 음성 -> 텍스트 변환
    result = stt_pipe("temp.wav")
    return jsonify({"text": result["text"]})


@app.route("/tts", methods=["POST"])
def handle_tts():
    text = request.json["text"]
    
    # ESPnet TTS 모델로 텍스트 -> 음성 변환
    speech, *_ = tts_model(text)
    
    # 음성을 WAV 파일로 저장
    wav_path = "output.wav"
    with open(wav_path, "wb") as f:
        f.write(speech.numpy())

    # 오디오 파일을 Base64로 변환하여 반환
    with open(wav_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    return jsonify({"audio": audio_base64})


@app.route("/image", methods=["POST"])
def handle_image():
    img_base64 = request.json["image"]
    img_bytes = base64.b64decode(img_base64)

    with open("temp.jpg", "wb") as f:
        f.write(img_bytes)
    
    # YOLOv5 모델을 사용하여 이미지에서 객체 탐지
    results = yolo_model("temp.jpg")
    detected_objects = [result["name"] for result in results.pandas().xywh[0].to_dict(orient="records")]

    return jsonify({"objects": detected_objects})

if __name__ == "__main__":
    # Railway에서 제공하는 포트로 Flask 앱 실행
    app.run(host="0.0.0.0", port=port)

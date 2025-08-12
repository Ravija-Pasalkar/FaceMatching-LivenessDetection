from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
import tempfile
import uuid
import subprocess
import difflib
import cv2
from google.generativeai import configure, GenerativeModel
import speech_recognition as sr
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("API_KEY")
configure(api_key=api) 

app = Flask(__name__, template_folder='templates', static_folder='.', static_url_path='')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

current_challenge = ""

def generate_challenge_from_language(language):
    global current_challenge
    model = GenerativeModel("gemini-2.0-flash")
    prompt = (
        f"Generate a short, creative, and fun sentence in {language} for liveness verification. "
        f"Do NOT include phrases like 'Here's a line' or any instructions. "
        f"Only return the sentence in quotes, no other text."
    )
    try:
        response = model.generate_content(prompt)
        challenge_text = response.text.strip().replace('\n', ' ')
        if '"' in challenge_text:
            parts = challenge_text.split('"')
            challenge_text = parts[1] if len(parts) > 1 else parts[0]
        current_challenge = challenge_text
        print(f"Gemini Challenge Generated: \"{current_challenge}\"")
        return challenge_text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Verification prompt unavailable. Please try again."

@app.route('/get-challenge')
def get_challenge():
    language = request.args.get('language', default='English')
    challenge = generate_challenge_from_language(language)
    return jsonify({'challenge': challenge})

def extract_audio_from_video(video_path, audio_path):
    try:
        command = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg Error: {e}")
        return False

def transcribe_audio(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"API error: {e}"

def extract_best_face_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    best_frame = None
    max_faces = 0
    best_frame_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            if result and len(result) > max_faces:
                max_faces = len(result)
                best_frame = frame
        except Exception:
            continue
    cap.release()

    if best_frame is not None:
        best_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{uuid.uuid4()}.jpg")
        cv2.imwrite(best_frame_path, best_frame)
        return best_frame_path
    return None

def verify_face(frame_path, reference_path):
    try:
        result = DeepFace.verify(img1_path=frame_path, img2_path=reference_path, model_name='ArcFace', enforce_detection=False)
        distance = result["distance"]
        print(f"Face Match Distance: {distance}")
        return result["verified"], distance
    except Exception as e:
        print(f"Face Verification Error: {e}")
        return False, None

@app.route('/verify', methods=['POST'])
def verify_audio():
    global current_challenge

    if 'video' not in request.files:
        return jsonify({'error': "No video uploaded."})

    if 'reference_image' not in request.files:
        return jsonify({'error': "No reference image uploaded."})

    video = request.files['video']
    reference_image = request.files['reference_image']

    if video.filename == '' or reference_image.filename == '':
        return jsonify({'error': "File(s) missing or invalid."})

    filename_video = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_video)

    filename_image = f"{uuid.uuid4()}.jpg"
    reference_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_image)

    try:
        video.save(video_path)
        reference_image.save(reference_image_path)
    except Exception as e:
        print(f"Save Error: {e}")
        return jsonify({'error': "Failed to save uploaded files."})

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_path = temp_audio.name

        success = extract_audio_from_video(video_path, audio_path)
        if not success:
            return jsonify({'error': "Failed to extract audio."})

        spoken_text = transcribe_audio(audio_path)

        spoken_clean = ''.join(e for e in spoken_text.lower() if e.isalnum() or e.isspace())
        challenge_clean = ''.join(e for e in current_challenge.lower() if e.isalnum() or e.isspace())

        similarity = difflib.SequenceMatcher(None, spoken_clean, challenge_clean).ratio()
        print(f"Similarity Ratio: {similarity:.2f}")

        voice_passed = similarity >= 0.8

        if voice_passed:
            best_frame_path = extract_best_face_frame(video_path)
            if not best_frame_path:
                return jsonify({'error': "No face detected in video."})

            face_verified, distance = verify_face(best_frame_path, reference_image_path)

            if face_verified:
                print("Verification: Successful")
                return jsonify({'success': True, 'redirect': '/next'})
            else:
                return jsonify({'error': "Face did not match. Please try again."})
        else:
            return jsonify({'error': "Spoken sentence did not match challenge."})

    except Exception as e:
        print(f"[Verification Error] {e}")
        return jsonify({'error': "Server error: " + str(e)})

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/next')
def fourth():
    return render_template('next.html')

if __name__ == '__main__':
    app.run(debug=False)

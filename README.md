# Face Match and Liveness Detection 
This project implements a web-based face verification and liveness detection system. Users are asked to upload a reference image and read a randomly generated sentence aloud while recording a video. The system verifies both identity and liveliness using face matching and audio-visual cues.

## Features
- Reference image upload
- Live video recording with microphone input
- Random challenge sentence for liveness detection
- Gemini API integration for dynamic prompt generation 
- Identity and Live presence verification
- Clean and responsive UI

## Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python (Flask)
- **AI Integration**: Google Gemini API (via `google.generativeai`)
- **Media Handling**: MediaRecorder API

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/Ravija-Pasalkar/FaceMatching-LivenessDetection.git
cd FaceMatching-LivenessDetection
```
### 2. Install Dependencies
```bash
pip install flask flask-cors google-generativeai speechrecognition deepface opencv-python ffmpeg-python
```
### 3. Configure Gemini API Key
- Access Google AI Studio: Go to the Google AI Studio platform.
- Sign in: Sign in with your Google account. 
- Get API Key: Locate the "Get API key" button and click on it.
- Create or Select a Project: You can choose to create a new Google Cloud project or use an existing one.
- Generate the API Key: Click on "Create API key" to generate your API key.
- Copy: Copy the generated API key.
- Update app.py: Replace **your_api_key** in app.py with your key.

### 4. Run the server
```bash
python app.py
```
- This will start the development server on loacalhost and thus can access the frontend via the localhost.
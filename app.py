from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('train_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

def preprocess(frame):
    frame = cv2.resize(frame, (250, 250))
    return frame

def save_video(file):
    filename = 'uploaded_video.avi'
    file.save(filename)
    return filename

@app.route('/detect_accident', methods=['POST'])
def detect_accident():
    video_file = request.files['video']
    if not video_file:
        return jsonify({'error': 'No video file provided'})

    # Save the uploaded video temporarily
    video_path = save_video(video_file)

    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = preprocess(frame)
            frames.append(frame)
        else:
            break
    cap.release()  # Release the video capture object

    if len(frames) == 0:
        os.remove(video_path)
        return jsonify({'error': 'No frames extracted from video'})

    predictions = model.predict(np.array(frames))
    os.remove(video_path)  # Remove the temporary video file
    if np.any(predictions == 1):
        return jsonify({'accidentDetected': True})
    else:
        return jsonify({'accidentDetected': False})

if __name__ == '__main__':
    app.run(debug=True)

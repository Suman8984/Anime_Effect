from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import uuid
from tqdm import tqdm
import cv2
import onnxruntime as ort
import numpy as np
import subprocess

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploaded_videos'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'models/Hayao_modul.onnx'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

progress_tracker = {}

class AnimeGAN:
    def __init__(self, model_path, use_gpu=False):
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def apply_anime_effect(self, input_frame):
        target_height, target_width = 256, 256
        resized_frame = cv2.resize(input_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame / 127.5 - 1.0
        resized_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32)
        output_frame = self.session.run([self.output_name], {self.input_name: resized_frame})[0]
        output_frame = np.squeeze(output_frame, axis=0)
        output_frame = (output_frame + 1.0) * 127.5
        output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        output_frame = cv2.resize(output_frame, (input_frame.shape[1], input_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return output_frame

def process_video(input_video, output_video, model_path, progress_callback, use_gpu=True):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = output_video.replace('.mp4', '__temp.mp4')
    video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    animegan = AnimeGAN(model_path, use_gpu=use_gpu)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break
        anime_frame = animegan.apply_anime_effect(frame)
        video_writer.write(anime_frame)
        progress_callback(i + 1, frame_count)
       

    cap.release()
    video_writer.release()

    # Fix codec and metadata using ffmpeg
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # FFMPEG path
    command = [
        ffmpeg_path,
        "-i", temp_output,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "faststart",
        "-y", output_video
    ]
    subprocess.check_call(command)

    os.remove(temp_output)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{file.filename}")
    file.save(input_path)

    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = 0

    def progress_callback(current, total):
        progress_tracker[task_id] = int((current / total) * 100)

    threading.Thread(target=process_video, args=(input_path, output_path, MODEL_PATH, progress_callback)).start()

    return jsonify({"message": "Processing started", "task_id": task_id, "output_url": f"/static/output/processed_{file.filename}"})

@app.route('/progress/<task_id>', methods=['GET'])
def progress(task_id):
    progress = progress_tracker.get(task_id, None)
    if progress is None:
        return jsonify({"error": "Invalid task ID"}), 404
    return jsonify({"progress": progress})

@app.route('/static/output/<filename>')
def output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

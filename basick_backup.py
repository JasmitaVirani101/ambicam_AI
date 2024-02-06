from flask import Flask, jsonify, request
import cv2

import torch
import subprocess as sp
import threading
from ultralytics import YOLO

app = Flask(__name__)


# Base path where models are stored
MODEL_BASE_PATH = 'prebuilt_model/'
selected_model_name = None  # No default model

stream_processes = {}
@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    data = request.get_json()
    stream_key = data.get('stream_key')

    if not stream_key:
        return jsonify({'error': 'Stream key is required'}), 400

    # Safely stop the stream if it exists
    if stream_key in stream_processes:
        process = stream_processes[stream_key]
        process.terminate()  # Terminate the FFmpeg process
        process.wait()  # Optional: Wait for the process to terminate
        del stream_processes[stream_key]  # Remove the process from the dictionary
        return jsonify({'message': 'Streaming stopped successfully'})
    else:
        return jsonify({'error': 'Stream not found'}), 404



def process_and_stream_frames(model_name, camera_url, stream_key):
    global stream_processes
    rtmp_url = f"{camera_url}_{model_name}"
    model_path = f'{MODEL_BASE_PATH}{model_name}.pt'
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True, device='cpu')
    video_cap = cv2.VideoCapture(camera_url)

    command = ['ffmpeg',
               '-f', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', '{}x{}'.format(int(video_cap.get(3)), int(video_cap.get(4))),
               '-r', '15',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-f', 'flv',
               rtmp_url]
    process = sp.Popen(command, stdin=sp.PIPE)
    stream_processes[stream_key] = process

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            results = model(frame)
            rendered_frame = results.render()[0]
            
            try:
                process.stdin.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR).tobytes())
            except BrokenPipeError:
                print("Broken pipe - FFmpeg process may have terminated unexpectedly.")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure cleanup is performed if process exists
        if process.poll() is None:
            process.terminate()
            process.wait()  # Wait for the process to terminate
        if stream_key in stream_processes:
            del stream_processes[stream_key]
        video_cap.release()

@app.route('/set_model', methods=['POST'])
def set_model_and_stream():
    global stream_process
    data = request.get_json()
    model_name = data.get('model_name')
    camera_url = data.get('camera_url')

    if not model_name or not camera_url:
        return jsonify({'error': 'Both model name and camera URL are required'}), 400

    # Unique key to identify the stream (could be refined based on requirements)
    stream_key = f"{camera_url}_{model_name}"
    
    # Check if a stream with the same key is already running, terminate if so
    if stream_key in stream_processes:
        stream_processes[stream_key].terminate()
        del stream_processes[stream_key]

    # Start a new stream
    thread = threading.Thread(target=process_and_stream_frames, args=(model_name, camera_url, stream_key))
    thread.start()

    return jsonify({'message': 'Streaming started', 'rtmp_url': f"{camera_url}_{model_name}"})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)


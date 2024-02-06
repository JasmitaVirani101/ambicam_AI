from flask import Flask, jsonify, request
import cv2

import torch
import subprocess as sp
import threading
import datetime
import os
import numpy as np
from collections import deque
from Deep_SORT_PyTorch.deep_sort import DeepSort

app = Flask(__name__)


# Base path where models are stored
MODEL_BASE_PATH = 'prebuilt_model/'
selected_model_name = None  # No default model
detected_ids = set() 
stream_processes = {}
frames_since_last_capture = {}
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


# Additional function for creating nested directories
def create_nested_directories(model_name):
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    nested_dir_path = os.path.join(os.getcwd(), "history",today_date, model_name)
    if not os.path.exists(nested_dir_path):
        os.makedirs(nested_dir_path)
    return nested_dir_path

# class BasicTracker:
#     def __init__(self):
#         self.objects = {}
#         self.id_count = 1

#     def update(self, detections):
#         new_objects = {}
#         for detection in detections:
#             x1, y1, x2, y2, conf, cls = detection[:6]
#             centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
#             # Simple tracking: assign new ID for each detection, in a real scenario you would match these
#             new_objects[self.id_count] = {'centroid': centroid, 'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls}
#             self.id_count += 1

#         # Determine new detections (simplified logic)
#         new_ids = set(new_objects.keys()) - set(self.objects.keys())
#         self.objects = new_objects  # Update tracked objects
#         return new_objects, new_ids
class BasicTracker:
    def __init__(self):
        self.objects = {}
        self.id_count = 1

    def update(self, detections):
        new_objects = {}
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection[:6]
            centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            # Simple tracking: assign new ID for each detection, in a real scenario you would match these
            new_objects[self.id_count] = {'centroid': centroid, 'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls}
            self.id_count += 1

        # Determine new detections (simplified logic)
        new_ids = set(new_objects.keys()) - set(self.objects.keys())
        self.objects = new_objects  # Update tracked objects
        return new_objects, new_ids

def process_and_stream_frames(model_name, camera_url, stream_key):
    global stream_processes,frames_since_last_capture
    rtmp_url = f"{camera_url}_{model_name}"
    model_path = f'{MODEL_BASE_PATH}/{model_name}.pt'
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True, device='cpu')
    
    # Set the confidence threshold to 0.7
    model.conf = 0.7
    
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

    tracker = BasicTracker()

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()  # Get detection results

            # Update tracker and draw bounding boxes
            tracked_objects, new_ids = tracker.update(detections)

            # Render frame with tracked objects
            for obj_id, obj in tracked_objects.items():
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{model.names[int(obj['cls'])]}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if obj_id not in frames_since_last_capture:
                    frames_since_last_capture[obj_id] = 0

                # Capture image if new object is detected and enough frames have passed since the last capture
                if obj_id in new_ids or frames_since_last_capture[obj_id] >30:
                    today_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                    image_folder_path = os.path.join(os.getcwd(), "history", today_folder, model_name)
                    if not os.path.exists(image_folder_path):
                        os.makedirs(image_folder_path)
                    image_name = f"{datetime.datetime.now().strftime('%H_%M_%S')}.jpg"
                    img_path = os.path.join(image_folder_path, image_name)
                    cv2.imwrite(img_path, frame)

                    # Reset the frame counter after capturing an image
                    frames_since_last_capture[obj_id] = 0
                else:
                    # Increment the frame counter if no image was captured
                    frames_since_last_capture[obj_id] += 1
                    cv2.imwrite(img_path, frame)
                   


            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Broken pipe - FFmpeg process may have terminated unexpectedly.")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
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


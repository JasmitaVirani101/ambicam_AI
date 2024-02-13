from flask import Flask, jsonify, request
import cv2

import torch
import subprocess as sp
import threading
import datetime
import os
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__)

# Define your email settings
SMTP_SERVER = 'smtp.gmail.com'  # Replace with your SMTP server
SMTP_PORT = 587  # SMTP port number
SMTP_USERNAME = 'jcvirani2000@gmail.com'  # Replace with your SMTP server username
SMTP_PASSWORD = 'nhou pwgh fddd brxj'  # Replace with your SMTP server password
SENDER_EMAIL = 'jcvirani2000@gmail.com'  # Replace with the sender's email address
RECIPIENT_EMAIL = 'jasmita@ambiplatforms.com'  # Replace with the recipient's email address

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
    
def send_email_notification_with_image(subject, body, image_path):
    try:
        # Set up the SMTP server
        server = smtplib.SMTP(host=SMTP_SERVER, port=SMTP_PORT)
        server.starttls()  # Upgrade the connection to secure
        server.login(SMTP_USERNAME, SMTP_PASSWORD)

        # Create the email message
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = RECIPIENT_EMAIL
        message['Subject'] = subject

        # Attach the email body
        message.attach(MIMEText(body, 'plain'))

        # Open the image file in binary mode and attach it to the email
        with open(image_path, 'rb') as file:
            img = MIMEImage(file.read(), name=os.path.basename(image_path))
            message.attach(img)

        # Send the email and close the server connection
        server.send_message(message)
        server.quit()
        print("Email with image sent successfully.")
    except Exception as e:
        print(f"Failed to send email with image: {e}")
email_sent_flag = False
def process_and_stream_frames(model_name, camera_url, stream_key):
    global stream_processes,frames_since_last_capture
    rtmp_url = f"{camera_url}_{model_name}"
    model_path = f'{MODEL_BASE_PATH}/{model_name}.pt'
    # model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True, device='cpu')
    model = model = YOLO("yolov8x.pt")
    
    
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
    time_reference = datetime.datetime.now()
    counter_frame = 0
    processed_fps = 0
    num_people = 0
    FIRE_CLASS_ID = 1
    try:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()  # Get detection results

            # Update tracker and draw bounding boxes
            tracked_objects, new_ids = tracker.update(detections)
            time_now = datetime.datetime.now()
            time_diff = (time_now - time_reference).total_seconds()
            if model_name == 'crowd':
                   
                num_people = 0
                for obj in detections:
                    # Class ID for 'person' is assumed to be 0
                    if int(obj[5]) == 0 and obj[4] >= 0.70:  # Check confidence
                        xmin, ymin, xmax, ymax = map(int, obj[:4])
                        num_people += 1
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.putText(frame, f"person {obj[4]:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Update FPS calculation
               
                if time_diff >= 1:
                    time_reference = time_now
                    processed_fps = counter_frame
                    counter_frame = 0
                else:
                    counter_frame += 1

                # Display the number of people and FPS on the frame
                cv2.putText(frame, f'People: {num_people}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if time_diff >= 10                                                                    :  # Capture an image every 5 minutes (300 seconds)
                    today_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                    image_folder_path = os.path.join(os.getcwd(), "history", today_folder, model_name)
                    if not os.path.exists(image_folder_path):
                        os.makedirs(image_folder_path)
                    image_name = f"{datetime.datetime.now().strftime('%H_%M_%S')}.jpg"
                    img_path = os.path.join(image_folder_path, image_name)
                    cv2.imwrite(img_path, frame)
            if model_name == 'fire':
               
                            # # Optionally, save the frame if fire is detected
                    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                        # Assuming fire class ID is 0, adjust according to your model
                        if cls == 0:
                            label = f'Fire {conf:.2f}'
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                                                                                   
                            today_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                            image_folder_path = os.path.join(os.getcwd(), "history", today_folder, model_name)
                            if not os.path.exists(image_folder_path):
                                os.makedirs(image_folder_path)
                            image_name = f"{datetime.datetime.now().strftime('%H_%M_%S')}.jpg"
                            img_path = os.path.join(image_folder_path, image_name)
                            cv2.imwrite(img_path, frame)


                            email_thread = threading.Thread(target=send_email_notification_with_image,
                                                            args=("Fire Detected!", "A fire has been detected. Please take immediate action.", img_path))
                            email_thread.start()

                            email_sent_flag = True
            elif model_name == 'yolov5s': 
                     
                        # Render frame with tracked objects
                for obj_id, obj in tracked_objects.items():
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{model.names[int(obj['cls'])]}"
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Check if the object ID is not in the frames_since_last_capture and update accordingly
                    if obj_id not in frames_since_last_capture:
                        frames_since_last_capture[obj_id] = 0

                    # Capture image if new object is detected and enough frames have passed since the last capture
                    if obj_id in new_ids or frames_since_last_capture[obj_id] > 30:
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
    app.run(debug=True,host="0.0.0.0",port=6000)


from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import cv2
import requests
import torch
import subprocess as sp
import threading
import datetime
import os
import numpy as np
import re
from flask_cors import CORS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__)
# Allow all origins for all routes
CORS(app)

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
def async_api_call(streamName, customer_id,image_name,cameraId,model_name,imgcount):
    """
    Asynchronously sends data to the API.
    """
    try:
        image_name = image_name
        img_url= f"https://inferenceimage.blob.core.windows.net/inferenceimages/{image_name}"
        send_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            'cameraid': cameraId, 
            'sendtime': send_time, 
            'imgurl': img_url,
            'modelname': model_name, 
            'ImgCount': int(imgcount),
            'customerid': customer_id,
            'streamname':streamName
        }
        api_url="http://142.93.213.150:3000/api/post-analytics"

        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            print("Data sent successfully!")
        else:
            print("Failed to send data! Response Code:", response.status_code)
    except Exception as e:
        print(f"Error sending data to API: {e}")

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
def process_and_stream_frames(model_name, camera_url, stream_key,customer_id,cameraId,streamName):
    global stream_processes,frames_since_last_capture
  
    rtmp_url = stream_key
    model_path = f'{MODEL_BASE_PATH}/{model_name}.pt'
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True, device=0)
    
    # Set the confidence threshold to 0.7
    model.conf = 0.7
    
    video_cap = cv2.VideoCapture(camera_url)
    
    command = ['ffmpeg',
               '-f', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', '{}x{}'.format(int(video_cap.get(3)), int(video_cap.get(4))),
               '-r', '5',
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
    customer_id=customer_id
    cameraId=cameraId
    streamName=streamName
    previous_num_people = 0
    last_capture_time = datetime.datetime.min  # Initialize with a minimum time

    min_interval = datetime.timedelta(seconds=60)  # Minimum time interval between captures
    class_counts = {}
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
                    if int(obj[5]) == 0 and obj[4] >= 0.60:  # Check confidence
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
                if num_people != previous_num_people and (time_now - last_capture_time) > min_interval:
                    # Update previous count
                    previous_num_people = num_people # Capture an image every 5 minutes (300 seconds)
                    last_capture_time = time_now
                    streamName = streamName
                    image_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "_"+streamName +".jpg"
                    image_path = "/home/torqueai/blobdrive/" + image_name 

                    cv2.imwrite(image_path, frame)
                        # Call the API asynchronously
                    threading.Thread(target=async_api_call, args=(streamName, customer_id,image_name,cameraId,model_name,num_people)).start()
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
                            streamName = streamName
                            image_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "_"+streamName +".jpg"
                            image_path = "/home/torqueai/blobdrive/" + image_name 

                            cv2.imwrite(image_path, frame)
                            # Call the API asynchronously
                            threading.Thread(target=async_api_call, args=(streamName, customer_id,image_name,cameraId,model_name,0)).start()
                         


                            email_thread = threading.Thread(target=send_email_notification_with_image,
                                                            args=("Fire Detected!", "A fire has been detected. Please take immediate action.", image_path))
                            email_thread.start()

                            email_sent_flag = True
            else: 
                     
                        # Render frame with tracked objects
                for obj_id, obj in tracked_objects.items():
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    class_id = int(obj['cls'])
                    class_name = model.names[class_id]
                    label = f"{model.names[int(obj['cls'])]}"
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Check if the object ID is not in the frames_since_last_capture and update accordingly
                    if obj_id not in frames_since_last_capture:
                        frames_since_last_capture[obj_id] = 0
                     # Update class counts
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                    # Capture image if new object is detected and enough frames have passed since the last capture
                    if obj_id in new_ids or frames_since_last_capture[obj_id] > 30:
                        streamName = streamName
                        image_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "_"+streamName +".jpg"
                        image_path = "/home/torqueai/blobdrive/" + image_name 

                        cv2.imwrite(image_path, frame)
                         # Call the API asynchronously
                        threading.Thread(target=async_api_call, args=(streamName, customer_id,image_name,cameraId,model_name,len(class_counts))).start()
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
    customer_id = data.get('customer_id')
    cameraId = data.get('cameraId')
    stream_name = data.get('streamName')
    if not model_name or not camera_url:
        return jsonify({'error': 'Both model name and camera URL are required'}), 400
    # Replace "media" with "media5" and "dvr" with digits to "live"
    modified_url = re.sub(r'media\d*', 'media5', camera_url)
    modified_url = re.sub(r'dvr\d+', 'live', modified_url)
    
    # Append the model name at the end of the URL, after a slash
    modified_url_with_model = f"{modified_url}_{model_name}"
    print("mooo",modified_url_with_model)
    
    # Unique key to identify the stream (could be refined based on requirements)
    stream_key = modified_url_with_model
    
    # Check if a stream with the same key is already running, terminate if so
    if stream_key in stream_processes:
        stream_processes[stream_key].terminate()
        del stream_processes[stream_key]

    # Start a new stream
    thread = threading.Thread(target=process_and_stream_frames, args=(model_name, camera_url, stream_key,customer_id,cameraId,stream_name))
    thread.start()

    return jsonify({'message': 'Streaming started', 'rtmp_url':stream_key,'customer_id':customer_id,'cameraId':cameraId,'streamName':stream_name})
########################################################################################################
############# Model upload ,delete,rename ###############
#####upload
ALLOWED_EXTENSIONS = {'pt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file part'}), 400
    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(MODEL_BASE_PATH, filename))
        return jsonify({'message': 'Model uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file type. Only .pt files are allowed'}), 400
    
#############rename
@app.route('/rename_model', methods=['POST'])
def rename_model():
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    old_path = os.path.join(MODEL_BASE_PATH, old_name)
    new_path = os.path.join(MODEL_BASE_PATH, new_name)
    if not os.path.exists(old_path):
        return jsonify({'error': 'Old model does not exist'}), 404
    if os.path.exists(new_path):
        return jsonify({'error': 'New model name already exists'}), 409
    os.rename(old_path, new_path)
    return jsonify({'message': 'Model renamed successfully'}), 200
#########delet
@app.route('/delete_model', methods=['POST'])
def delete_model():
    data = request.get_json()
    model_name = data.get('model_name')
    model_path = os.path.join(MODEL_BASE_PATH, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model does not exist'}), 404
    os.remove(model_path)
    return jsonify({'message': 'Model deleted successfully'}), 200
############################################################################
################# model list
@app.route('/get_models', methods=['GET'])
def get_models():
    models_dir = MODEL_BASE_PATH
    try:
        # List all files in the models directory
        files = os.listdir(models_dir)
        # Filter out files to only include .pt files
        model_files = [file for file in files if file.endswith('.pt')]
        return jsonify({'models': model_files}), 200
    except Exception as e:
        # Handle errors, such as if the directory does not exist
        return jsonify({'error': str(e)}), 500
    
###############################################
##############get the active streams
@app.route('/running_streams', methods=['GET'])
def get_running_streams():
    # Collect all the stream keys representing the RTMP URLs of running streams
    running_streams = list(stream_processes.keys())
    return jsonify({'running_streams': running_streams})
##############################################################################################
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)


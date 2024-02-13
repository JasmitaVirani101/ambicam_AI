import cv2
import torch
import numpy as np

def load_model():
    # Load your YOLOv5 model
    model = torch.hub.load('yolov5', 'custom', path="/home/torque-hq/Documents/ambicam_AI/prebuilt_model/fire.pt", source='local', force_reload=True, device='cpu')
    model.conf = 0.3  # Set confidence threshold
    return model

def process_video(camera_url, model):
    # Initialize video capture
    video_cap = cv2.VideoCapture(camera_url)
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)
        
        # Process detections
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            # Assuming fire class ID is 0, adjust according to your model
            if cls == 0:
                label = f'Fire {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                
        # Display the frame
        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    video_cap.release()
    cv2.destroyAllWindows()

def main():
    model = load_model()
    camera_url = "rtmp://media5.ambicam.com:1938/live/1efa24f9-0cd0-47c5-b604-c7e3ee118302" # Use 0 for webcam, or replace with your video stream URL
    process_video(camera_url, model)

if __name__ == '__main__':
    main()

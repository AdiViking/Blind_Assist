import streamlit as st
from ultralytics import YOLO
import pyttsx3
import cv2
import numpy as np
import threading
import queue
import time

class BlindAssistant:
    def __init__(self, confidence_threshold=0.5, announce_interval=5):
        self.engine = pyttsx3.init()
        self.models = [
            YOLO('yolov8n.pt'),
            YOLO('yolov8n-seg.pt')
        ]
        self.confidence_threshold = confidence_threshold
        self.announce_interval = announce_interval
        self.last_announced = {}
        self.announcement_queue = queue.Queue()
        self.start_announcement_thread()
    
    def start_announcement_thread(self):
        """Voice announcement thread."""
        def worker():
            while True:
                announcement = self.announcement_queue.get()
                if announcement is None:
                    break
                self.engine.say(announcement)
                self.engine.runAndWait()
        
        self.announcement_thread = threading.Thread(target=worker, daemon=True)
        self.announcement_thread.start()
    
    def detect_objects(self, frame):
        """Run YOLO object detection on the frame."""
        all_boxes, all_scores, all_labels = [], [], []
        for model in self.models:
            results = model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(conf)
                        all_labels.append(label)
        return all_boxes, all_scores, all_labels
    
    def should_announce(self, label):
        """Check if a label should be announced."""
        current_time = time.time()
        if label not in self.last_announced or (current_time - self.last_announced[label] > self.announce_interval):
            self.last_announced[label] = current_time
            return True
        return False

assistant = BlindAssistant()

# Streamlit app
st.title("AudioNav Assistance")
st.text("Object detection and navigation aid for visually impaired individuals.")

# Video capture
video = st.camera_input("Capture a frame")

if video:
    # Read frame from Streamlit camera input
    frame = cv2.imdecode(np.frombuffer(video.read(), np.uint8), cv2.IMREAD_COLOR)
    boxes, scores, labels = assistant.detect_objects(frame)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if assistant.should_announce(label):
            assistant.announcement_queue.put(f"Detected {label}")
        st.text(f"Detected {label} with confidence {score:.2f}")

    st.image(frame, channels="BGR")

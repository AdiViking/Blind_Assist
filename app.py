import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import queue
import json

class StreamlitBlindAssistant:
    def __init__(self, confidence_threshold=0.5):
        # Initialize models and settings
        self.initialize_models()
        self.confidence_threshold = confidence_threshold
        
        # Thread-safe queue for detections
        self.detection_queue = queue.Queue()
        
        # Object categories
        self.object_categories = {
            'mobility': [
                'wheelchair', 'walking_stick', 'crutches', 'walker',
                'bicycle', 'motorcycle', 'car', 'bus', 'truck'
            ],
            'urban_infrastructure': [
                'traffic_light', 'crosswalk', 'street_sign', 'road_sign',
                'sidewalk', 'stairs', 'elevator', 'escalator', 'door',
                'bench', 'trash_can', 'fire_hydrant', 'parking_meter'
            ],
            'safety_hazards': [
                'cliff', 'step', 'hole', 'construction_barrier',
                'wet_floor', 'uneven_surface'
            ],
            'public_spaces': [
                'table', 'chair', 'counter', 'ATM', 'phone_booth',
                'drinking_fountain', 'restroom_sign'
            ]
        }
        
        # Tracking last detections
        self.last_detections = {}
        self.last_announced = {}
    
    def initialize_models(self):
        """Initialize YOLO models"""
        try:
            self.models = [
                YOLO('yolov8n.pt'),
                YOLO('yolov8n-seg.pt')
            ]
            st.session_state['models_loaded'] = True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.session_state['models_loaded'] = False
    
    def categorize_object(self, label):
        """Categorize detected objects"""
        for category, objects in self.object_categories.items():
            if label in objects:
                return category
        return 'other'
    
    def calculate_distance(self, box, frame_height):
        """Estimate distance based on bounding box size"""
        box_height = box[3] - box[1]
        distance = max(0.5, min(5, 5 * (1 - (box_height / frame_height))))
        return round(distance, 2)
    
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        if not hasattr(self, 'models'):
            return frame, []
        
        detections = []
        frame_objects = {}
        
        # Collect detections from all models
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
                        
                        # Calculate distance
                        distance = self.calculate_distance(
                            [x1, y1, x2, y2], 
                            frame.shape[0]
                        )
                        
                        # Categorize object
                        category = self.categorize_object(label)
                        
                        # Add to detections
                        detection = {
                            'label': label,
                            'category': category,
                            'distance': distance,
                            'confidence': float(conf),
                            'box': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = (0, 255, 0) if label != 'person' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        text = f"{label} ({distance}m)"
                        cv2.putText(frame, text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, detections

    def callback(self, frame):
        """Callback for WebRTC video processing"""
        img = frame.to_ndarray(format="bgr24")
        processed_frame, detections = self.process_frame(img)
        
        # Add detections to queue for the Streamlit app to process
        self.detection_queue.put(detections)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.set_page_config(page_title="AudioNav Assistance", layout="wide")
    
    st.title("AudioNav Assistance - Web Version")
    st.markdown("""
    This web application provides real-time object detection and assistance for the visually impaired.
    Use your device's camera to detect objects in your surroundings.
    """)
    
    # Initialize the assistant if not already done
    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = StreamlitBlindAssistant()
    
    # WebRTC Configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="blind-assistant",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: st.session_state['assistant'],
        async_processing=True,
    )
    
    # Display detections
    if webrtc_ctx.state.playing:
        detection_placeholder = st.empty()
        
        while True:
            try:
                # Get latest detections from queue
                detections = st.session_state['assistant'].detection_queue.get_nowait()
                
                # Display detections
                if detections:
                    detection_text = ""
                    for det in detections:
                        detection_text += f"Detected {det['label']} ({det['category']}) at {det['distance']}m\n"
                    
                    detection_placeholder.text_area(
                        "Latest Detections:", 
                        detection_text,
                        height=200
                    )
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Error processing detections: {e}")
                break
            
            time.sleep(0.1)

if __name__ == "__main__":
    main()

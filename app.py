import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from PIL import Image
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

    def process_uploaded_image(self, image):
        """Process uploaded images instead of video stream"""
        if isinstance(image, str):
            frame = cv2.imread(image)
        else:
            # Convert PIL Image to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        processed_frame, detections = self.process_frame(frame)
        
        # Convert back to PIL Image for Streamlit
        processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        return processed_image, detections

def main():
    st.set_page_config(page_title="AudioNav Assistance", layout="wide")
    
    st.title("AudioNav Assistance - Web Version")
    st.markdown("""
    This web application provides object detection and assistance for the visually impaired.
    Upload images to detect objects in your surroundings.
    """)
    
    # Initialize the assistant if not already done
    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = StreamlitBlindAssistant()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image and display results
        processed_image, detections = st.session_state['assistant'].process_uploaded_image(image)
        
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, use_column_width=True)
        
        # Display detections
        if detections:
            st.subheader("Detected Objects")
            detection_text = ""
            for det in detections:
                detection_text += f"Detected {det['label']} ({det['category']}) at {det['distance']}m\n"
            st.text_area("Detections:", detection_text, height=200)

if __name__ == "__main__":
    main()

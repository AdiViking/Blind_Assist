import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            # Load YOLOv8n, a smaller and faster model
            self.model = YOLO('yolov8n.pt')
            st.success('Model loaded successfully!')
        except Exception as e:
            st.error(f'Error loading model: {str(e)}')
            
    def process_image(self, image):
        """Process an image and return detections"""
        if self.model is None:
            return None, []
            
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Get predictions
        results = self.model(img_array)
        
        # Process results
        processed_img = img_array.copy()
        detections = []
        
        # Get the first result (assuming single image input)
        result = results[0]
        
        # Draw boxes and collect detections
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls]
            
            if conf > 0.5:  # Confidence threshold
                # Draw rectangle
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                cv2.putText(processed_img, 
                           f'{label} {conf:.2f}',
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 255, 0),
                           2)
                           
                # Add to detections list
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': [x1, y1, x2, y2]
                })
        
        return Image.fromarray(processed_img), detections

def main():
    st.set_page_config(page_title="AudioNav Assistant", layout="wide")
    
    st.title("AudioNav Assistant")
    st.markdown("""
    Upload an image to detect objects. The system will identify and highlight 
    objects in the image and provide descriptions.
    """)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state['detector'] = ObjectDetector()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        processed_image, detections = st.session_state['detector'].process_image(image)
        
        with col2:
            st.subheader("Processed Image")
            if processed_image is not None:
                st.image(processed_image, use_column_width=True)
        
        # Display detections
        if detections:
            st.subheader("Detected Objects")
            for det in detections:
                st.write(f"• {det['label']} (Confidence: {det['confidence']:.2f})")

if __name__ == "__main__":
    main()

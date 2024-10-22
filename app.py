import streamlit as st
import torch
from PIL import Image
import tempfile
import os
import cv2
import requests
from io import BytesIO
import time

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Streamlit app layout
st.title('Underwater Marine Object Detection (Fish,Star Fish)')
st.write('Upload an image or video to detect underwater objects.')

# Image or video file uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

# Function to detect objects in the image using YOLOv5
def detect_and_display(image):
    results = model(image)  # Inference
    results.render()  # Render predictions
    return Image.fromarray(results.ims[0])  # Convert results to PIL Image

# Function to process video frame-by-frame using YOLOv5 and save the output
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    stframe = st.empty()  # Placeholder to display video frames
    progress_bar = st.progress(0)  # Progress bar

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB format (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model(frame_rgb)
        results.render()
        
        # Convert back to BGR for OpenCV to write the video
        frame_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        
        # Write the frame with detections to the output video
        out.write(frame_bgr)
        
        # Display the processed frame in the app
        stframe.image(results.ims[0], use_column_width=True)
        
        # Update progress
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

        # Small sleep to simulate smooth sliding effect
        time.sleep(0.03)  # Adjust this value for smoother or faster transitions

    cap.release()
    out.release()
    progress_bar.empty()

# Handle uploaded image or video
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Process the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        st.write("Detecting objects...")
        result_image = detect_and_display(image)
        
        # Display results
        st.image(result_image, caption='Detection Results', use_column_width=True)
    
    elif uploaded_file.type == 'video/mp4':
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
        
        # Temporary file to store output video
        output_video_path = os.path.join(tempfile.gettempdir(), 'output.mp4')

        # Process video and save the output
        st.write("Processing video for object detection...")
        process_video(video_path, output_video_path)

        # Offer the processed video for download
        with open(output_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.download_button(label="Download Processed Video", data=video_bytes, file_name="detected_video.mp4", mime="video/mp4")

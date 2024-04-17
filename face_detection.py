import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image


mp_face_detection = mp.solutions.face_detection
st.title("Face Detection Streamlit Application")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        if results.detections:
            st.header("Faces detected in the image:")
            for detection in results.detections:
                boundingboxC = detection.location_data.relative_bounding_box
                imageHeight, imageWidth, _ = image_rgb.shape
                bbox = int(boundingboxC.xmin * imageWidth), int(boundingboxC.ymin * imageHeight), int(boundingboxC.width * imageWidth), int(boundingboxC.height * imageHeight)
                cv2.rectangle(image_rgb, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                st.write(f"Face coordinates: {bbox}")
                
            st.image(image_rgb, channels="RGB", use_column_width=True)
            
        else:
            st.error("Face not found in the image. Please upload another image.")

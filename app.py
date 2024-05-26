import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

st.title("YOLOv8 Object Detection")
st.write("Upload an image to perform object detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform object detection
    results = model(image)

    # Annotate the image
    annotated_image = image.copy()

    for result in results:
        boxes = result.boxes  # Get the boxes
        for box in boxes:
            # Extract coordinates, confidence, and class id
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert annotated image back to RGB (OpenCV uses BGR by default)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
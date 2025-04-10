import streamlit as st
import cv2
import numpy as np
import requests
import time
import os

# Roboflow API configuration
API_KEY = "EtEYi37vrlr3MdCwf14a"
MODEL_ENDPOINT = "https://detect.roboflow.com/demofabricdefect/5"

# Roboflow inference function
def roboflow_detect(image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    response = requests.post(
        MODEL_ENDPOINT,
        params={"api_key": API_KEY},
        files={"file": image_data}
    )
    return response.json()

# Shape and severity helpers
def compute_shape_descriptors(x1, y1, x2, y2, image_w, image_h):
    w, h = x2 - x1, y2 - y1
    area = w * h
    rel_area = area / (image_w * image_h)
    aspect_ratio = w / h if h != 0 else 0

    if aspect_ratio < 0.5:
        shape = "Tall & Narrow"
    elif aspect_ratio > 2.0:
        shape = "Wide & Short"
    else:
        shape = "Square-ish"
    return area, rel_area, aspect_ratio, shape

def estimate_severity(rel_area):
    if rel_area < 0.01:
        return 'Minor', (0, 255, 0)
    elif rel_area < 0.03:
        return 'Moderate', (0, 255, 255)
    else:
        return 'Severe', (0, 0, 255)

# Streamlit UI setup
st.set_page_config(layout="centered", page_title="Cloth Defect Detection")
st.title("🧵 Cloth Defect Detection (YOLO + Roboflow)")
st.markdown("Detect fabric defects either in real-time via webcam or from uploaded images.")

# Sidebar mode selector
mode = st.sidebar.radio("Choose Detection Mode", ["Webcam", "Image Upload"])

FRAME_INTERVAL = 1  # seconds

# Webcam Mode
if mode == "Webcam":
    start = st.toggle("▶ Start Webcam Detection")
    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while start:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            h, w = frame.shape[:2]
            tmp_path = "temp_frame.jpg"
            cv2.imwrite(tmp_path, frame)
            time.sleep(0.1)

            try:
                result = roboflow_detect(tmp_path)
            except Exception as e:
                st.error(f"Detection Error: {e}")
                continue

            for pred in result.get("predictions", []):
                class_name = pred["class"]
                x, y = pred["x"], pred["y"]
                width, height = pred["width"], pred["height"]
                confidence = pred["confidence"]

                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                area, rel_area, ar, shape = compute_shape_descriptors(x1, y1, x2, y2, w, h)
                severity, color = estimate_severity(rel_area)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} | {severity} | {shape} | {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)
            time.sleep(FRAME_INTERVAL)

        cap.release()
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")

# Image Upload Mode
elif mode == "Image Upload":
    uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        h, w = frame.shape[:2]
        tmp_path = "uploaded_image.jpg"
        cv2.imwrite(tmp_path, frame)

        with st.spinner("Detecting defects..."):
            try:
                result = roboflow_detect(tmp_path)
            except Exception as e:
                st.error(f"Detection Error: {e}")
                result = {"predictions": []}

        for pred in result.get("predictions", []):
            class_name = pred["class"]
            x, y = pred["x"], pred["y"]
            width, height = pred["width"], pred["height"]
            confidence = pred["confidence"]

            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            area, rel_area, ar, shape = compute_shape_descriptors(x1, y1, x2, y2, w, h)
            severity, color = estimate_severity(rel_area)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} | {severity} | {shape} | {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", caption="🧵 Annotated Defects", use_column_width=True)

        if os.path.exists("uploaded_image.jpg"):
            os.remove("uploaded_image.jpg")

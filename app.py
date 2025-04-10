import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import io
import torch
from utils.model_utils import CombinedClassifier
from utils.infer_utils import predict_and_display

# Paths
YOLO_MODEL_PATH = "models/fabric_defect_yolov8.pt"
CNN_MODEL_PATH = "models/combined_fabric_defect_model.pt"
CSV_PATH = "E:/Fabric Defect Detection/fabric_dataset/fabric_types.csv"

# Load CSV
fabric_df = pd.read_csv(CSV_PATH)
CLASSES = ['hole', 'knot', 'stain']
id_to_fabric = {i: name for i, name in enumerate(sorted(fabric_df["Fabric_Type"].unique()))}

# Load YOLO model
model_yolo = YOLO(YOLO_MODEL_PATH)

# Load CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn = CombinedClassifier(len(CLASSES), len(id_to_fabric))
state_dict = torch.load(CNN_MODEL_PATH, map_location=device)
model_cnn.load_state_dict(state_dict)
model_cnn.to(device).eval()

# Page config
st.set_page_config(page_title="TexSure", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #F6F8D5;
            font-size: 18px;
            color: #333333;
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            color: #7D4F50;
            text-align: center;
            margin-top: 1rem;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #333333;
            margin: 2rem auto 1rem;
            max-width: 600px;
            text-align: center;
        }
        .social-box {
            background-color: #D0EBFF;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem auto;
            width: fit-content;
        }
        .social-box a {
            margin: 0 15px;
            text-decoration: none;
            color: #0077b6;
            font-weight: bold;
            font-size: 1.2rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #e8f8f5  !important;
        }
        .result-box {
            background-color: rgba(255, 255, 240, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 0 10px #ddd;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("TexSure")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Try Our Model"])

# Home page
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='main-title'>TexSure</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='subtitle'>
            TexSure is a Smart Textile Surety and Defect Detection System powered by YOLOv8 and CNN. 
            It enables real-time and image-based detection of fabric defects such as holes, knots, and more, 
            with severity analysis and fabric type classification.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class='social-box'>
                <a href="https://github.com/vrushikakpanchal" target="_blank">🔗 GitHub</a>
                <a href="https://www.linkedin.com/in/vrushikakpanchal" target="_blank">🔗 LinkedIn</a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("E:/Fabric Defect Detection/static/img.jpg", use_container_width=True)

# Model Interaction Page
if page == "🔍 Try Our Model":
    st.markdown("<h3 style='text-align:center;'>Choose Detection Method</h3>", unsafe_allow_html=True)
    mode = st.radio("", ["📸 Real-Time Detection", "📷 Take Photo & Upload", "🖼️ Upload from Gallery"], horizontal=True)

    def analyze_and_display_yolo(image_np, filename="uploaded.jpg"):
        results = model_yolo.predict(image_np, conf=0.5)
        r = results[0]
        img_annotated = r.plot()

        names = model_yolo.names
        defect_list = []
        for box in r.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, class_id = box
            defect_list.append({
                "label": names[int(class_id)],
                "area": int((x2 - x1) * (y2 - y1))
            })

        if not defect_list:
            st.success("✅ No defects detected!")
            return

        defect = defect_list[0]
        count = len(defect_list)
        area = defect["area"]
        severity = "Severe" if area > 8000 else "Moderate" if area > 4000 else "Minor"
        shape = "Square-ish"

        matched = fabric_df[fabric_df["Image"].str.contains(os.path.basename(filename), case=False)]
        fabric_type = matched["Fabric_Type"].values[0] if not matched.empty else "Unknown"

        results_text = f"""
        TexSure Report 🧵

        ✅ Cloth Material: {fabric_type}
        🩹 Defect Type: {defect['label']}
        #️⃣ No. of Defects: {count}
        🔳 Shape: {shape}
        📏 Area: {area} px²
        🔥 Severity: {severity}
        """

        with st.container():
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"**Cloth Material:** `{fabric_type}`")
            st.markdown(f"**Defect Detected:** `{defect['label']}`")
            st.markdown(f"**No. of Defects:** `{count}`")
            st.markdown(f"**Shape:** `{shape}`")
            st.markdown(f"**Area:** `{area} px²`")
            st.markdown(f"**Severity:** `{severity}`")
            st.image(image_np, caption="Original Image", use_container_width=True)
            st.image(img_annotated, caption="Analyzed Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            buffer = io.StringIO()
            buffer.write(results_text)
            st.download_button(
                label="📄 Download Report",
                data=buffer.getvalue(),
                file_name="TexSure_Report.txt",
                mime="text/plain"
            )

    def analyze_with_cnn(image):
        st.subheader("CNN-Based Fabric & Defect Classification")
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        predict_and_display(model_cnn, img_cv, CLASSES, id_to_fabric)

    if mode == "📸 Real-Time Detection":
        st.info("Use your webcam for real-time fabric detection.")
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible.")
                break
            results = model_yolo.predict(frame, conf=0.5)
            FRAME_WINDOW.image(results[0].plot(), channels="BGR")
        cap.release()

    elif mode == "📷 Take Photo & Upload":
        img_file = st.camera_input("Capture Fabric Image")
        if img_file is not None:
            img = Image.open(img_file).convert("RGB")
            img_np = np.array(img)
            st.image(img, caption="Captured Image", use_container_width=True)
            analyze_and_display_yolo(img_np, filename=img_file.name)
            analyze_with_cnn(img)

    elif mode == "🖼️ Upload from Gallery":
        uploaded = st.file_uploader("Upload a Fabric Image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            img_np = np.array(img)
            analyze_and_display_yolo(img_np, filename=uploaded.name)
            analyze_with_cnn(img)
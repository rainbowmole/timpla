import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# --- CONFIGURATION ---
# Update this path to your current champion model
MODEL_PATH = "runs/detect/runs/detect/timpla_full/yolo12n_canonical_e100/weights/best.pt"
CONF_THRESHOLD = 0.35

st.set_page_config(page_title="TIMPLA Ingredient Detector", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights not found at {MODEL_PATH}. Please check the path.")
        return None
    return YOLO(MODEL_PATH)

model = load_model()

st.title("🥘 TIMPLA Ingredient Detector")
st.markdown("Upload an image, take a photo, or start live detection to identify ingredients.")

if model is None:
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Detection Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, CONF_THRESHOLD, 0.05)
img_size = st.sidebar.selectbox("Image Size", [320, 640, 1280], index=1)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📷 Take Photo", "🎥 Live Detection"])

def process_image(img):
    # Convert PIL to OpenCV
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run Inference
    results = model.predict(img_bgr, conf=conf, imgsz=img_size, agnostic_nms=True, max_det=300)[0]

    # Plot results
    res_plotted = results.plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return res_rgb, results

with tab1:
    st.subheader("Upload from Gallery")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        processed_img, res = process_image(image)
        with col2:
            st.image(processed_img, caption="Detections", use_container_width=True)

        # Show detected classes
        if len(res.boxes) > 0:
            detected_classes = [model.names[int(cls)] for cls in res.boxes.cls]
            st.success(f"**Detected:** {', '.join(set(detected_classes))}")
        else:
            st.warning("No ingredients detected. Try lowering the confidence threshold in the sidebar.")

with tab2:
    st.subheader("Capture from Webcam")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        processed_img, res = process_image(image)
        st.image(processed_img, caption="Detection Result", use_container_width=True)
        if len(res.boxes) > 0:
            detected_classes = [model.names[int(cls)] for cls in res.boxes.cls]
            st.success(f"**Detected:** {', '.join(set(detected_classes))}")
        else:
            st.warning("No ingredients detected. Try lowering the confidence threshold in the sidebar.")

with tab3:
    st.subheader("Real-time Detection")
    st.info("Live detection uses your browser's webcam stream via Streamlit's WebRTC logic.")
    
    # Note: True live video in Streamlit often requires streamlit-webrtc.
    # For a simple implementation, we use a loop with camera_input or a placeholder.
    # Since we are in a local environment, a simple CV2 loop is easier if run locally,
    # but for the app, we will use the camera_input as a 'pseudo-live' snap.
    
    run_live = st.checkbox("Enable Live-Capture Mode")
    if run_live:
        img_file_live = st.camera_input("Live Stream")
        if img_file_live is not None:
            image = Image.open(img_file_live).convert("RGB")
            processed_img, res = process_image(image)
            st.image(processed_img, use_container_width=True)
            if len(res.boxes) > 0:
                detected_classes = [model.names[int(cls)] for cls in res.boxes.cls]
                st.success(f"**Detected:** {', '.join(set(detected_classes))}")
            else:
                st.warning("No ingredients detected. Try lowering the confidence threshold in the sidebar.")


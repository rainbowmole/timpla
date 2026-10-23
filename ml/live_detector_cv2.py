import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
MODEL_PATH = "runs/detect/runs/detect/timpla_full/yolo12n_canonical_e100/weights/best.pt"
CONF_THRESHOLD = 0.25  # Lowered for better sensitivity
IOU_THRESHOLD = 0.5     # Allows more overlapping detections

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return
    
    print("Loading Champion Model...")
    model = YOLO(MODEL_PATH)
    
    # 2. Initialize Webcam
    print("Requesting camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera index 0 failed. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("\n❌ ERROR: Could not open webcam.")
            print("Please check:")
            print("1. Is another app (Zoom, Teams, Browser) using the camera?")
            print("2. System Settings > Privacy & Security > Camera -> Is Terminal/VS Code allowed?")
            return

    # Test read a frame to verify access
    ret, frame = cap.read()
    if not ret:
        print("\n❌ ERROR: Camera opened but cannot read frames.")
        cap.release()
        return
    
    print("\n✅ Camera access granted and verified!")
    print("🚀 LIVE DETECTION STARTED")
    print("Press 'q' to quit")
    print(f"Settings: Conf={CONF_THRESHOLD}, IOU={IOU_THRESHOLD}")
    print("-" * 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera connection.")
            break

        # Run Inference
        # agnostic_nms=True and max_det=300 help catch multiple different ingredients
        results = model.predict(
            frame, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD, 
            agnostic_nms=True, 
            max_det=300, 
            verbose=False
        )[0]

        # Plot results on the frame
        annotated_frame = results.plot()

        # Print detected ingredients to terminal
        if len(results.boxes) > 0:
            labels = [model.names[int(cls)] for cls in results.boxes.cls]
            print(f"Detected: {', '.join(set(labels))}", end='\r')

        # Show the video window
        cv2.imshow("TIMPLA Live Detector - Press 'q' to Quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

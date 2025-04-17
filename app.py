import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image as PILImage
from colors import get_dominant_color, rgb_to_name
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Load model once
@st.cache_resource
def load_model():
    return YOLO("FINAL_CODE/model.pt")

model = load_model()

# Class names
class_names = ['truck', 'person', 'traffic sign', 'rider', 'car', 'motorcycle', 'animal',
               'bicycle', 'vehicle fallback', 'caravan', 'autorickshaw', 'train', 'traffic light', 'bus', 'trailer']

# Streamlit UI
st.title("🚗 Vehicle Detection & Suspect Filter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
print("\n\n\nHello wait wait wait:",uploaded_file ,"\n\n\n")

if uploaded_file:
    img_path = "input.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(PILImage.open(img_path), caption="Uploaded Image", use_container_width=True)

    st.markdown("### Running YOLO Detection...")
    results = model.predict(source=img_path, conf=0.7, save=True)

    # Read the saved prediction image
    output_img_path = os.path.join(results[0].save_dir, os.path.basename(img_path))
    image_cv = cv2.imread(output_img_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else np.array([])

    st.image(image_rgb, caption="Detected Vehicles", use_container_width=True)

    st.markdown("### Detected Vehicles & Alerts")
    for det in detections:
        class_index = int(det[5])
        vehicle_type = class_names[class_index] if class_index < len(class_names) else "Unknown"
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        vehicle_image = image_cv[y1:y2, x1:x2]
        vehicle_color = get_dominant_color(vehicle_image)
        color_name = rgb_to_name(vehicle_color)
        st.warning(f"🚨 {vehicle_type.upper()} DETECTED! Confidence: {det[4]:.2f}, Color: {color_name}, "
                   f"Location: ({x1}, {y1}) → ({x2}, {y2})")

    # Filter Section
    st.markdown("### 🔍 Filter Suspect Vehicle")
    search_color = st.text_input("Enter Color of Suspect (e.g., gray, indigo)").lower()
    search_type = st.text_input("Enter Type of Suspect (e.g., car, truck)").lower()

    if st.button("Find Suspect"):
        # Convert uploaded PIL image to OpenCV BGR format for clean bounding box overlay
        uploaded_image_pil = PILImage.open(uploaded_file).convert("RGB")
        image_original = cv2.cvtColor(np.array(uploaded_image_pil), cv2.COLOR_RGB2BGR)
        image_filtered = image_original.copy()

        found = False

        for det in detections:
            x1, y1, x2, y2, conf, class_idx = map(int, det[:6])
            detected_type = class_names[class_idx].lower()
            detected_color = get_dominant_color(image_cv[y1:y2, x1:x2])
            color_name = rgb_to_name(detected_color).lower()

            if (search_color in color_name) and (search_type in detected_type):
                found = True
                label = f"{color_name} {detected_type}"
                cv2.rectangle(image_filtered, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(image_filtered, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        if found:
            st.success("🚘 Matching suspect(s) found!")
            st.image(cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB), caption="Filtered Result", use_container_width=True)
        else:
            st.error(f"No vehicles found matching color '{search_color}' and type '{search_type}'")

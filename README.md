# 🚗 Vehicle Detection & Suspect Filter using YOLO and Streamlit

A real-time web application built using **YOLO (You Only Look Once)** and **Streamlit** to detect vehicles in an image and filter potential suspects based on vehicle **type** and **color**.

---

## 📌 Features

- ✅ Upload any street-view image.
- 🧠 Detects multiple vehicles (car, bus, truck, autorickshaw, etc.) using a custom-trained YOLO model.
- 🎨 Identifies the **dominant color** of each vehicle using image segmentation.
- 🔍 Search and filter **suspect vehicles** by specifying:
  - Vehicle type (e.g., `car`, `truck`, `bus`)
  - Color (e.g., `white`, `gray`, `indigo`)
- 📍 Highlights only the suspect vehicles with bounding boxes on the **original image**.
- ⚡ Fast and interactive interface using Streamlit.

---

## 📸 Sample Workflow

1. **Upload an image** containing street vehicles.
2. YOLO model detects all vehicles and displays them with color & type info.
3. Enter suspect's color and type in the filter input.
4. Matching vehicles will be highlighted in the original image with bounding boxes and labels.

---

## 🧰 Tech Stack

| Tool | Description |
|------|-------------|
| **YOLOv8 (Ultralytics)** | For real-time object detection |
| **Streamlit** | For building the web-based UI |
| **OpenCV** | For image processing and bounding box drawing |
| **Pillow** | Image handling |
| **Custom Color Detection** | Dominant color extraction and name mapping |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/vehicle-sos.git
cd sos

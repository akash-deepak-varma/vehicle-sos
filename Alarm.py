import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil
from IPython.display import Image, display
from colors import get_dominant_color, rgb_to_name

# Define a function to send an immediate alarm message with vehicle details.
def send_alarm(detections, class_names, img_path):
    image = cv2.imread(img_path)
    for det in detections:
        class_index = int(det[5])
        vehicle_type = class_names[class_index] if class_index < len(class_names) else "Unknown"
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        vehicle_image = image[y1:y2, x1:x2]
        vehicle_color = get_dominant_color(vehicle_image)
        color_name = rgb_to_name(vehicle_color)
        message = (f"\nALERT: {vehicle_type} DETECTED!!!\n"
                   f"CONFIDENCE: {det[4]:.2f}\n"
                   f"COLOR: {color_name}\n"
                   f"LOCATION: ({det[0]:.0f}, {det[1]:.0f}), ({det[2]:.0f}, {det[3]:.0f})")
        print(message)

# Function to get the dominant color name from bounding box
def get_dominant_color_name(image, box):
    x1, y1, x2, y2 = box
    vehicle_image = image[y1:y2, x1:x2]
    dominant_color = get_dominant_color(vehicle_image)
    return rgb_to_name(dominant_color)

# Load YOLO model and run detection
model = YOLO(r"FINAL_CODE\model.pt")
img_path = r"images.jpg"
results = model.predict(source=img_path, conf=0.7, save=True)

output_img = os.path.join(results[0].save_dir, os.path.basename(img_path))
image_cv = cv2.imread(output_img)

cv2.imshow("Detected Vehicle", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else np.array([])

class_names = ['truck', 'person', 'traffic sign', 'rider', 'car', 'motorcycle', 'animal',
               'bicycle', 'vehicle fallback', 'caravan', 'autorickshaw', 'train', 'traffic light', 'bus', 'trailer']

original_detections = detections.copy()
tracked_detections = [det[:5] for det in detections]  # using first 5 values, no tracker ID available

if detections.size > 0:
    send_alarm(detections, class_names, img_path)

shutil.copy(output_img, 'detected_vehicle.jpg')
shutil.make_archive('vehicle_detection_output', 'zip', results[0].save_dir)


# ----------------- FILTER SECTION -----------------
img_path = r"FINAL_CODE\FINAL_CODE\images.jpg"
image_cv = cv2.imread(img_path)
search_color = input("\nEnter the color of the Suspect: ").strip().lower()
search_type = input("Enter the Type of the Suspect: ").strip().lower()

image_filtered = image_cv.copy()
matches_found = False

for det in original_detections:
    x1, y1, x2, y2, conf, class_idx = map(int, det[:6])
    detected_type = class_names[class_idx]
    detected_color = get_dominant_color_name(image_cv, [x1, y1, x2, y2])

    if (search_type in detected_type.lower()) and (search_color in detected_color.lower()):
        matches_found = True
        label = f"{detected_color} {detected_type}"
        cv2.rectangle(image_filtered, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image_filtered, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)


# Show filtered result
if matches_found:
    result_path = 'filtered_result.jpg'
    cv2.imwrite(result_path, image_filtered)

    # Display using IPython (works in Jupyter Notebook)
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=result_path))
    except ImportError:
        pass

    # Display using OpenCV (works in local environment)
    cv2.imshow("Filtered Result", image_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No vehicles found matching color '{search_color}' and type '{search_type}'")

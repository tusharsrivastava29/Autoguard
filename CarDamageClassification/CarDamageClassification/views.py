from django.shortcuts import render
import numpy as np
import cv2
import base64
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("CarDamageClassification/yolov8.pt")
labels = {
    "dent": "Dent",
    "scratch": "Scratch",
    "crack": "Crack",
    "glass shatter": "Glass Shatter",
    "lamp broken": "Lamp Broken",
    "tire flat": "Flat Tire"
}

def index(request):
    if request.method == 'POST':
        # Check for file upload
        car_image = request.FILES.get('car_image')
        if car_image:
            print("CAR IMAGE UPLOAD")
            # Convert the uploaded image to a NumPy array
            nparr = np.frombuffer(car_image.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV format

            # Convert BGR (OpenCV default) to RGB as YOLO expects RGB input
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform inference with YOLOv8
            results = model(image_rgb)
            result_image = results[0].plot()  # Get the image with bounding boxes and labels

            # Extract labels and bounding boxes
            class_names = results[0].names  # Dictionary mapping class IDs to class names
            boxes = results[0].boxes  # Detection boxes

            # Collect the labels and their corresponding bounding boxes
            predicted_labels = [labels.get(class_names[int(box.cls)], class_names[int(box.cls)]) for box in boxes] if boxes is not None else []

            # Convert the processed image back to BGR (for displaying with OpenCV)
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            # Convert the processed image to base64
            _, img_encoded = cv2.imencode('.png', result_bgr)
            image_data = base64.b64encode(img_encoded).decode('utf-8')

            # Pass the base64 image and labels to the template for display
            context = {
                'image_data': image_data,
                'predicted_labels': predicted_labels
            }

            return render(request, "index.html", context)
        
        # Check for image data URL from the camera capture
        image_data_url = request.POST.get('image_data')
        if image_data_url:
            print("CAR IMAGE CAPTURED")
            # Convert data URL to a NumPy array
            header, encoded = image_data_url.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV format

            # Convert BGR (OpenCV default) to RGB as YOLO expects RGB input
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform inference with YOLOv8
            results = model(image_rgb)
            result_image = results[0].plot()  # Get the image with bounding boxes and labels

            # Extract labels and bounding boxes
            class_names = results[0].names  # Dictionary mapping class IDs to class names
            boxes = results[0].boxes  # Detection boxes

            # Collect the labels and their corresponding bounding boxes
            predicted_labels = [labels.get(class_names[int(box.cls)], class_names[int(box.cls)]) for box in boxes] if boxes is not None else []

            # Convert the processed image back to BGR (for displaying with OpenCV)
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            # Convert the processed image to base64
            _, img_encoded = cv2.imencode('.png', result_bgr)
            image_data = base64.b64encode(img_encoded).decode('utf-8')

            # Pass the base64 image and labels to the template for display
            context = {
                'image_data': image_data,
                'predicted_labels': predicted_labels
            }

            return render(request, "index.html", context)

    # In case of GET request or no image uploaded
    
    return render(request, "index.html")

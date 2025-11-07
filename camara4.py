import cv2
import numpy as np
import os
from ultralytics import YOLO  # YOLOv8 model for object detection

# Path to resources (update this to your folder path)
resource_dir = r"C:\Users\sivas\Documents\age-gender-detection"  # Replace with the path to your folder containing the model files

# Verify all necessary files exist
required_files = [
    "deploy_age.prototxt",
    "age_net.caffemodel",
    "deploy_gender.prototxt",
    "gender_net.caffemodel",
    "haarcascade_frontalface_default.xml"
]
for file in required_files:
    full_path = os.path.join(resource_dir, file)
    if not os.path.exists(full_path):
        print(f"Error: Missing file {file} in {resource_dir}")
        exit()

# Load the face detection model
face_cascade = cv2.CascadeClassifier(os.path.join(resource_dir, "haarcascade_frontalface_default.xml"))

# Load the age and gender models
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(resource_dir, "deploy_age.prototxt"),
    os.path.join(resource_dir, "age_net.caffemodel")
)
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(resource_dir, "deploy_gender.prototxt"),
    os.path.join(resource_dir, "gender_net.caffemodel")
)

# Labels for age and gender predictions
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your YOLOv8 model path

# Initialize the USB camera (adjust the index as needed for your system)
usb_camera_index = 1  # Change to 0 for default webcam
cap = cv2.VideoCapture(usb_camera_index)

if not cap.isOpened():
    print(f"Error: Unable to access USB camera with index {usb_camera_index}.")
    exit()

print("Press 'q' to exit the program.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame from the USB camera.")
        break

    # Perform YOLO object detection
    results = model(frame)

    # Parse YOLO detection results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
        scores = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert bounding box coordinates to integers
            label = f"{model.names[int(class_id)]}: {score:.2f}"
            color = (0, 255, 0)  # Green color for the rectangle

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label text near the object
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region for predictions
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict the age
        try:
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
        except cv2.error as e:
            print("Error during age prediction:", e)
            age = "Unknown"

        # Predict the gender
        try:
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
        except cv2.error as e:
            print("Error during gender prediction:", e)
            gender = "Unknown"

        # Display the predictions
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the combined output
    cv2.imshow("Object, Age, and Gender Detection (USB Camera)", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the USB camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

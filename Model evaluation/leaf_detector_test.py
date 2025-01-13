from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLOv8 model
model_path = '/Volumes/SanDisk SSD/detect/train32/weights/best.pt'
model = YOLO(model_path)

# Video source
input_video_path = '/Users/joshua.stanley/Desktop/Science Research/Images:Video/Testing video/leaf.mp4'
cap = cv2.VideoCapture(input_video_path)  # Use 0 for webcam

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Set confidence threshold

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    # Perform inference
    results = model(frame)
    
    # Loop through detected boxes
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box
            if confidence < CONFIDENCE_THRESHOLD:
                continue  # Skip boxes with low confidence
            
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # Crop the leaf region
            leaf_region = frame[y1:y2, x1:x2]
            
            # Convert to grayscale and apply thresholding
            gray = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Fill contours on the original frame
            for contour in contours:
                contour = contour + [x1, y1]  # Adjust contour position to match original image coordinates
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), cv2.FILLED)  # Fill with green color
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Display label
            label = f"{model.names[int(class_id)]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the result
    cv2.imshow('Leaf Detection with Filled Outlines', frame)

    # Press 'q' to quit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

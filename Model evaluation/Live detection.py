from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Load YOLO model
model_path = '/Volumes/SanDisk SSD/detect/train32/weights/best.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Object classes
classNames = ["leaf"]

# Suppress verbose logging
model.overrides['verbose'] = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam.")
        break

    # Perform inference
    try:
        results = model(img)
    except Exception as e:
        print(f"Inference error: {e}")
        break

    # Process results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"
            print(f"Detected: {class_name}")

            # Add label to frame
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Display frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

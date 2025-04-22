from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8m model
model = YOLO('yolov8m.pt')  # YOLOv8m is the medium-sized model for balanced performance

# Define a function to filter objects detected on the desk
def filter_desk_objects(results):
    desk_objects = [
        'laptop', 'book', 'pen', 'pencil', 'mobile phone', 'tablet', 'mouse',
        'keyboard', 'notebook', 'charger', 'headphones', 'spectacles',
        'scissors', 'stapler', 'eraser', 'ruler', 'marker', 'highlighter',
        'paper', 'envelope', 'cup', 'bottle', 'calculator', 'sticky notes',
        'folder', 'smartphone', 'ballpoint pen', 'gel pen', 'fountain pen'
    ]
    filtered_results = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Class ID
            label = model.names[cls]  # Get class name
            if label in desk_objects:
                filtered_results.append((label, box.xyxy[0].tolist()))  # Label and bounding box
    return filtered_results

# Function to process webcam frames
def process_frame(frame):
    results = model(frame)  # Run YOLO detection
    filtered = filter_desk_objects(results)  # Filter desk objects

    # Draw bounding boxes and labels on the frame
    for label, bbox in filtered:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


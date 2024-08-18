
import torch
from ultralytics import YOLO
import cv2
import numpy as np

def draw_center_point(image, center, radius=5):
    # Draw white outline
    cv2.circle(image, center, radius + 2, (255, 255, 255), 2)
    # Draw black inner circle
    cv2.circle(image, center, radius, (0, 0, 0), -1)
    # Draw colored cross
    cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 255), 2)
    cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 255, 255), 2)

# Ensure CUDA is available and set to use the NVIDIA GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set to use the first CUDA device (your NVIDIA GPU)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load the YOLO model
model = YOLO("C:/Users/rodri/Downloads/WasteSenseAI-BestModel.pt")

# Optimize for inference
model.fuse()
_ = model.to('cuda')  # Move model to GPU

# Set a smaller batch size and image size to reduce memory usage
batch_size = 80  # Changed to 1 for real-time processing
img_size = 864  # or 416, 512, depending on your memory constraints

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam+

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to match img_size
    frame = cv2.resize(frame, (img_size, img_size))
    
    # Run inference
    results = model(frame, conf=0.5, imgsz=img_size)
    
    for r in results:
        # Get the annotated frame
        annotated_frame = r.plot()
        
        # Draw center points for each detected object
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                draw_center_point(annotated_frame, center)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference InstanceSegmentation WasteSenseAI", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

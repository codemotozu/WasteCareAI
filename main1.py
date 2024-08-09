 VIDEO TUTORIAL: YOLOv9 Instance Segmentation on Live Webcam and YOLOv8 Comparison

# from ultralytics import YOLO

# # Build a YOLOv9c model from pretrained weight
# model = YOLO('yolov9c-seg.pt')

# # Display model information (optional)
# model.info()

# results = model.predict(0, show=True, save=True)

----------------------------------------------------------
 SOLVED PROBLEM "GPU OUT OF MEMORY"

# from ultralytics import YOLO
# import torch
# import cv2

# # Build a YOLOv9c model from pretrained weight
# model = YOLO('yolov9c-seg.pt')

# # Display model information (optional)
# model.info()

# # Try to use GPU, fall back to CPU if out of memory
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Set a smaller input size
# input_size = 320  # You can adjust this value. Smaller size = less memory usage

# # Open video capture
# cap = cv2.VideoCapture(0)  # Use 0 for webcam

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Failed to read frame")
#         break

#     # Resize frame to reduce memory usage
#     frame = cv2.resize(frame, (input_size, input_size))

#     try:
#         # Predict with reduced input size
#         results = model(frame, 
#                         imgsz=input_size,
#                         conf=0.25,  # Confidence threshold
#                         iou=0.45,  # NMS IoU threshold
#                         max_det=100,  # Maximum number of detections per image
#                         device=device)

#         # Process results
#         for r in results:
#             boxes = r.boxes  # Boxes object for bbox outputs
#             masks = r.masks  # Masks object for segment masks outputs
#             probs = r.probs  # Class probabilities for classification outputs
#             print(f"Detected {len(boxes)} objects")

#             # Draw bounding boxes on the frame
#             annotated_frame = r.plot()
            
#             # Display the annotated frame
#             cv2.imshow("YOLOv9 Inference", annotated_frame)

#     except torch.cuda.OutOfMemoryError:
#         print("CUDA out of memory. Switching to CPU.")
#         device = 'cpu'
#         model.to(device)
#         continue
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Clear CUDA cache
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

----------------------------------------------------------


Red point in the center of the object


# from ultralytics import YOLO
# import torch
# import cv2
# import numpy as np

# # Build a YOLOv9c model from pretrained weight
# model = YOLO('yolov9c-seg.pt')

# # Display model information (optional)
# model.info()

# # Try to use GPU, fall back to CPU if out of memory
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Set a smaller input size
# input_size = 320  # You can adjust this value. Smaller size = less memory usage

# # Open video capture
# cap = cv2.VideoCapture(0)  # Use 0 for webcam

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Failed to read frame")
#         break

#     # Resize frame to reduce memory usage
#     frame = cv2.resize(frame, (input_size, input_size))

#     try:
#         # Predict with reduced input size
#         results = model(frame, 
#                         imgsz=input_size,
#                         conf=0.25,  # Confidence threshold
#                         iou=0.45,  # NMS IoU threshold
#                         max_det=100,  # Maximum number of detections per image
#                         device=device)

#         # Process results
#         for r in results:
#             boxes = r.boxes  # Boxes object for bbox outputs
#             masks = r.masks  # Masks object for segment masks outputs
#             probs = r.probs  # Class probabilities for classification outputs
#             print(f"Detected {len(boxes)} objects")

#             # Draw bounding boxes on the frame
#             annotated_frame = r.plot()
            
#             # Draw center points
#             for box in boxes:
#                 # Get the coordinates of the bounding box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 # Calculate the center point
#                 center_x = int((x1 + x2) / 2)
#                 center_y = int((y1 + y2) / 2)
#                 # Draw the center point
#                 cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

#             # Display the annotated frame
#             cv2.imshow("YOLOv9 Inference", annotated_frame)

#     except torch.cuda.OutOfMemoryError:
#         print("CUDA out of memory. Switching to CPU.")
#         device = 'cpu'
#         model.to(device)
#         continue
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Clear CUDA cache
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

----------------------------------------------------------

Yellow cruise in the center of the object

# from ultralytics import YOLO
# import torch
# import cv2
# import numpy as np

# # Build a YOLOv9c model from pretrained weight
# model = YOLO('yolov9c-seg.pt')

# # Display model information (optional)
# model.info()

# # Try to use GPU, fall back to CPU if out of memory
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Set a smaller input size
# input_size = 320  # You can adjust this value. Smaller size = less memory usage

# def draw_center_marker(img, center, size=10, color=(0, 255, 255), thickness=2):
#     x, y = center
#     # Draw a cross
#     cv2.line(img, (x - size, y), (x + size, y), color, thickness)
#     cv2.line(img, (x, y - size), (x, y + size), color, thickness)
#     # Draw a circle around the cross
#     cv2.circle(img, center, size, color, thickness)

# # Open video capture
# cap = cv2.VideoCapture(0)  # Use 0 for webcam

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Failed to read frame")
#         break

#     # Resize frame to reduce memory usage
#     frame = cv2.resize(frame, (input_size, input_size))

#     try:
#         # Predict with reduced input size
#         results = model(frame, 
#                         imgsz=input_size,
#                         conf=0.25,  # Confidence threshold
#                         iou=0.45,  # NMS IoU threshold
#                         max_det=100,  # Maximum number of detections per image
#                         device=device)

#         # Process results
#         for r in results:
#             boxes = r.boxes  # Boxes object for bbox outputs
#             masks = r.masks  # Masks object for segment masks outputs
#             probs = r.probs  # Class probabilities for classification outputs
#             print(f"Detected {len(boxes)} objects")

#             # Draw bounding boxes on the frame
#             annotated_frame = r.plot()
            
#             # Draw center markers
#             for box in boxes:
#                 # Get the coordinates of the bounding box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 # Calculate the center point
#                 center_x = int((x1 + x2) / 2)
#                 center_y = int((y1 + y2) / 2)
#                 # Draw the center marker
#                 draw_center_marker(annotated_frame, (center_x, center_y))

#             # Display the annotated frame
#             cv2.imshow("YOLOv9 Inference", annotated_frame)

#     except torch.cuda.OutOfMemoryError:
#         print("CUDA out of memory. Switching to CPU.")
#         device = 'cpu'
#         model.to(device)
#         continue
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Clear CUDA cache
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

----------------------------------------------------------


yellow, white and black cruise in the center of the object
to do : add FPS and make sure the program is running with cuda
to solve this error "error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'"
do this: pip uninstall opencv-python and then pip install opencv-python


from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov9c-seg.pt')

# Display model information (optional)
model.info()

# Try to use GPU, fall back to CPU if out of memory
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Set a smaller input size
input_size = 320  # You can adjust this value. Smaller size = less memory usage

def draw_center_point(image, center, radius=5):
    # Draw white outline
    cv2.circle(image, center, radius + 2, (255, 255, 255), 2)
    # Draw black inner circle
    cv2.circle(image, center, radius, (0, 0, 0), -1)
    # Draw colored cross
    cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 255), 2)
    cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 255, 255), 2)

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Resize frame to reduce memory usage
    frame = cv2.resize(frame, (input_size, input_size))

    try:
        # Predict with reduced input size
        results = model(frame, 
                        imgsz=input_size,
                        conf=0.25,  # Confidence threshold
                        iou=0.45,  # NMS IoU threshold
                        max_det=100,  # Maximum number of detections per image
                        device=device)

        # Process results
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
            print(f"Detected {len(boxes)} objects")

            # Draw bounding boxes on the frame
            annotated_frame = r.plot()
            
            # Draw center points
            for box in boxes:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                # Calculate the center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # Draw the center point
                draw_center_point(annotated_frame, (center_x, center_y))

            # Display the annotated frame
            cv2.imshow("YOLOv9 Inference", annotated_frame)

    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Switching to CPU.")
        device = 'cpu'
        model.to(device)
        continue
    except Exception as e:
        print(f"An error occurred: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

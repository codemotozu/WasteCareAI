# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# import requests
# from io import BytesIO

## IDENTIFY ANY IMAGE BY URL ADDRESS

# # Check if CUDA is available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load pre-trained ResNet18 model
# weights = ResNet18_Weights.DEFAULT
# model = torchvision.models.resnet18(weights=weights)
# model = model.to(device)
# model.eval()

# # Define image transformations
# transform = weights.transforms()

# # Load class labels
# categories = weights.meta["categories"]


# # Function to predict image class
# def predict_image(image_url):
#     # Download the image
#     response = requests.get(image_url)
#     img = Image.open(BytesIO(response.content))

#     # Transform the image
#     img_t = transform(img)
#     batch_t = torch.unsqueeze(img_t, 0).to(device)

#     # Make prediction
#     with torch.no_grad():
#         out = model(batch_t)

#     # Get top 5 predictions
#     probabilities = torch.nn.functional.softmax(out[0], dim=0)
#     top5_prob, top5_catid = torch.topk(probabilities, 5)

#     # Print results
#     print("\nTop 5 predictions:")
#     for i in range(top5_prob.size(0)):
#         print(f"{categories[top5_catid[i]]:>20}: {top5_prob[i].item()*100:.2f}%")


# # Example usage with a web image
# image_url = "https://plus.unsplash.com/premium_photo-1689266188052-704d33673e69?q=80&w=387&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# predict_image(image_url)

# # Load MNIST dataset
# mnist_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# mnist_dataset = torchvision.datasets.MNIST(
#     root="./data", train=True, download=True, transform=mnist_transform
# )


# # Function to display MNIST image subset as ASCII art
# def display_mnist_subset(image_tensor, start_row, end_row, start_col, end_col):
#     subset = image_tensor[0, start_row:end_row, start_col:end_col].numpy()
#     for i in range(subset.shape[0]):
#         for j in range(subset.shape[1]):
#             pixel_value = subset[i, j]
#             if pixel_value < -0.5:
#                 print("  ", end="")
#             elif pixel_value < 0:
#                 print(" .", end="")
#             elif pixel_value < 0.5:
#                 print(" o", end="")
#             else:
#                 print(" @", end="")
#         print()

# # Display and analyze the first image from the MNIST dataset
# image, label = mnist_dataset[0]
# print(f"\nMNIST Label: {label}")
# print("MNIST Image:")

# print("\nImage tensor shape:", image.shape)
# print("Subset of pixel values (10:15, 10:15):")
# print(image[:, 10:15, 10:15])
# print("Max pixel value:", torch.max(image).item())
# print("Min pixel value:", torch.min(image).item())

# print("\nVisualization of subset (10:15, 10:15):")
# display_mnist_subset(image, 10, 15, 10, 15)


## IDENTIFY WASTE MATERIAL BY URL IMAGE ADDRES

# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.models import resnet18, ResNet18_Weights
# from PIL import Image
# import requests
# from io import BytesIO

# # Check if CUDA is available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Define the class labels
# categories = ["plastic", "carton", "metal", "food"]

# # Load pre-trained ResNet18 model
# weights = ResNet18_Weights.DEFAULT
# model = resnet18(weights=weights)
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, len(categories))
# model = model.to(device)
# model.eval()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Function to predict image class
# def predict_image(image_url):
#     # Download the image
#     response = requests.get(image_url)
#     img = Image.open(BytesIO(response.content)).convert("RGB")

#     # Transform the image
#     img_t = transform(img)
#     batch_t = torch.unsqueeze(img_t, 0).to(device)

#     # Make prediction
#     with torch.no_grad():
#         out = model(batch_t)

#     # Get top prediction
#     probabilities = torch.nn.functional.softmax(out[0], dim=0)
#     top_prob, top_catid = torch.topk(probabilities, 1)

#     # Print result
#     print(f"\nPredicted class: {categories[top_catid[0]]}")
#     print(f"Confidence: {top_prob[0].item()*100:.2f}%")

# # Example usage with a web image
# image_url = "https://images.unsplash.com/photo-1582765114728-428aa4d1ff36?q=80&w=387&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# predict_image(image_url)


## Identifying objetcs within a vide
## topics: image segmentation, object detection and bouncing box

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# Local video file path
video_path = "local_videos/ferrari.mp4"

# Load the Faster R-CNN model for object detection
detection_model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
detection_model.eval()

# Load the DeepLabV3 model for image segmentation
segmentation_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
segmentation_model.eval()

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_model.to(device)
segmentation_model.to(device)

# Open the local video file
video_capture = cv2.VideoCapture("local_videos/ferrari.mp4")

# Set the desired playback speed (e.g., 2x speed)
playback_speed = 2.0
video_capture.set(cv2.CAP_PROP_FPS, video_capture.get(cv2.CAP_PROP_FPS) * playback_speed)

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Preprocess the frame for object detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)

    # Perform object detection with Faster R-CNN
    with torch.no_grad():
        detections = detection_model(img)

    # Preprocess the frame for image segmentation
    seg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_img = Image.fromarray(seg_img)
    seg_img = transforms.ToTensor()(seg_img)
    seg_img = seg_img.unsqueeze(0).to(device)

    # Perform image segmentation with DeepLabV3
    with torch.no_grad():
        segmentation = segmentation_model(seg_img)["out"]

    # Process the segmentation mask
    segmentation = segmentation.argmax(1).squeeze().cpu().numpy()
    segmentation = segmentation.astype(np.uint8)
    segmentation_color = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)
    segmentation_color = cv2.cvtColor(segmentation_color, cv2.COLOR_BGR2RGB)
    segmentation_color = cv2.addWeighted(frame, 0.7, segmentation_color, 0.3, 0)

    # Process the detections
    for detection in detections:
        boxes = detection["boxes"]
        scores = detection["scores"]
        labels = detection["labels"]

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:  # Confidence threshold
                # Draw bounding box
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cv2.rectangle(segmentation_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label and score
                label_name = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"][label]
                label_text = f"{label_name}: {score:.2f}"
                cv2.putText(
                    segmentation_color,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    # Display the segmentation result with bounding boxes
    cv2.imshow("Video Analysis", segmentation_color)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()

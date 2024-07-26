import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import requests
from io import BytesIO

# IDENTIFY ANY IMAGE BY URL ADDRESS

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained ResNet18 model
weights = ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model = model.to(device)
model.eval()

# Define image transformations
transform = weights.transforms()

# Load class labels
categories = weights.meta["categories"]


# Function to predict image class
def predict_image(image_url):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Transform the image
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # Make prediction
    with torch.no_grad():
        out = model(batch_t)

    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Print results
    print("\nTop 5 predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]:>20}: {top5_prob[i].item()*100:.2f}%")


# Example usage with a web image
image_url = "https://plus.unsplash.com/premium_photo-1689266188052-704d33673e69?q=80&w=387&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
predict_image(image_url)

# Load MNIST dataset
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=mnist_transform
)


# Function to display MNIST image subset as ASCII art
def display_mnist_subset(image_tensor, start_row, end_row, start_col, end_col):
    subset = image_tensor[0, start_row:end_row, start_col:end_col].numpy()
    for i in range(subset.shape[0]):
        for j in range(subset.shape[1]):
            pixel_value = subset[i, j]
            if pixel_value < -0.5:
                print("  ", end="")
            elif pixel_value < 0:
                print(" .", end="")
            elif pixel_value < 0.5:
                print(" o", end="")
            else:
                print(" @", end="")
        print()

# Display and analyze the first image from the MNIST dataset
image, label = mnist_dataset[0]
print(f"\nMNIST Label: {label}")
print("MNIST Image:")

print("\nImage tensor shape:", image.shape)
print("Subset of pixel values (10:15, 10:15):")
print(image[:, 10:15, 10:15])
print("Max pixel value:", torch.max(image).item())
print("Min pixel value:", torch.min(image).item())

print("\nVisualization of subset (10:15, 10:15):")
display_mnist_subset(image, 10, 15, 10, 15)


-----------------------------------------------------------

## IDENTIFY WASTE MATERIAL BY URL IMAGE ADDRES

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import requests
from io import BytesIO

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the class labels
categories = ["plastic", "carton", "metal", "food"]

# Load pre-trained ResNet18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(categories))
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict image class
def predict_image(image_url):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # Transform the image
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # Make prediction
    with torch.no_grad():
        out = model(batch_t)

    # Get top prediction
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    # Print result
    print(f"\nPredicted class: {categories[top_catid[0]]}")
    print(f"Confidence: {top_prob[0].item()*100:.2f}%")

# Example usage with a web image
image_url = "https://images.unsplash.com/photo-1582765114728-428aa4d1ff36?q=80&w=387&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
predict_image(image_url)


# Identifying objetcs within a vide
# topics: image segmentation, object detection and bouncing box


-----------------------------------------------------------

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

-----------------------------------------------------------

# INSTANCE SEGMENTATION AND OBJECT DETECTION ON WEBCAM

import pixellib
from pixellib.instance import instance_segmentation
import cv2


segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5',)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Apply instance segmentation
    res = segmentation_model.segmentFrame(frame, show_bboxes=True)
    image = res[1]
    
    cv2.imshow('Instance Segmentation', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

-----------------------------------------------------------


Ich mache dieses Tutorial: https://www.youtube.com/watch?v=6J2PvzhO_Mk&list=PL8b3FgAiLSnsH8BzKnOTCYT8XxQUJ2RVF&index=3&ab_channel=LearnOpenCV
Training YOLOv8 Models for Trash Detection: AI for Ocean Clean-Up

import requests
import zipfile
import os
import glob
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import yaml
from ultralytics import YOLO
import multiprocessing

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")
        
        
        
def create_yolo_command(task, mode, model, imgsz, data, epochs, batch, name, exist_ok, amp):
    command = f"""yolo \\
    task={task} \\
    mode={mode} \\
    model={model} \\
    imgsz={imgsz} \\
    data={data} \\
    epochs={epochs} \\
    batch={batch} \\
    name={name} \\
    exist_ok={exist_ok} \\
    amp={amp}
    """
    return command



def visualize(result_dir):
    """
    Function accepts a list of images and plots
    them in either a 1x1 grid or 2x2 grid.
    """
    image_names = glob.glob(os.path.join(result_dir, '*.jpg'))
    if len(image_names) < 4:
        plt.figure(figsize=(10, 7))
        for i, image_name in enumerate(image_names):
            image = plt.imread(image_name)
            plt.subplot(1, 1, i+1)
            plt.imshow(image)
            plt.axis('off')
            break
    if len(image_names) >= 4:
        plt.figure(figsize=(15, 12))
        for i, image_name in enumerate(image_names):
            image = plt.imread(image_name)
            plt.subplot(2, 2, i+1)
            plt.imshow(image)
            plt.axis('off')
            if i == 3:
                break
    plt.tight_layout()
    plt.show()

def main():
    # Specify a local directory to save the downloaded files
    local_directory = './downloaded_files'
    os.makedirs(local_directory, exist_ok=True)

    # Update the save location
    save_location = os.path.join(local_directory, 'trash_inst_material.zip')

    download_file('https://www.dropbox.com/s/ievh0sesad015z0/trash_inst_material.zip?dl=1', save_location)
    unzip(zip_file=save_location)

   # Download the inference data
    inference_data_location = os.path.join(local_directory, 'trash_segment_inference_data.zip')
    download_file(
        'https://www.dropbox.com/s/smdsotz25al3bi2/trash_segment_inference_data.zip?dl=1',
        inference_data_location
    )
    unzip(zip_file=inference_data_location)

    cwd = os.getcwd()
    print(cwd)

    attr = {
        'path': cwd + '/trash_inst_material',
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'rov', 1: 'plant', 2: 'animal_fish', 3: 'animal_starfish',
            4: 'animal_shells', 5: 'animal_crab', 6: 'animal_eel', 7: 'animal_etc',
            8: 'trash_etc', 9: 'trash_fabric', 10: 'trash_fishing_gear', 11: 'trash_metal',
            12: 'trash_paper', 13: 'trash_plastic', 14: 'trash_rubber', 15: 'trash_wood'
        }
    }

    with open('trashcan_inst_material.yaml', 'w') as f:
        yaml.dump(attr, f)

    EPOCHS = 5
    BATCH = 4  # Reduced batch size
    IM_SIZE = 416  # Reduced image size
    
    
       # Create YOLO command string
    yolo_command = create_yolo_command(
        task='segment',
        mode='train',
        model='yolov8n-seg.pt',
        imgsz=IM_SIZE,
        data='trashcan_inst_material.yaml',
        epochs=EPOCHS,
        batch=BATCH,
        name='yolov8n-seg',
        exist_ok=True,
        amp=False
    )

    print("YOLO Command:")
    print(yolo_command)

    # Initialize the YOLO model
    model = YOLO('yolov8n-seg.pt')  # Using the nano model, which is smaller

    # Start the training
    results = model.train(
        task='segment',
        mode='train',
        data='trashcan_inst_material.yaml',
        imgsz=IM_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        name='yolov8n-seg',
        exist_ok=True,
        amp=False,
        device='0'  # Explicitly specify GPU device
    )

    print(results)


    # Visualize results
    result_dir = os.path.join('runs', 'segment', 'yolov8n-seg')
    visualize(result_dir)

    # Load the trained model for inference
    # model = YOLO('runs/segment/yolov8m_predict/weights/best.pt')
    model = YOLO('/content/drive/MyDrive/v8-seg-c100-models/yolov8m-seg/weights/best.pt')

    # Run inference on videos
    model.predict(
        source='trash_segment_inference_data/manythings.mp4',
        save=True,
        exist_ok=True,
        name='yolov8m_predict_videos1'
    )

    # Visualize the inference results
    visualize('runs/segment/yolov8m_predict_videos1')


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()


-----------------------------------------------------------


Ich mache dieses Tutorial: https://www.youtube.com/watch?v=Dmv4EVBuCTQ&list=PL8b3FgAiLSnsH8BzKnOTCYT8XxQUJ2RVF&index=3&ab_channel=CodeWithAarohi
Train YOLOv10 on Custom Dataset


import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# from models import YOLOv5

from models.models import YOLOv5
from utils.dataset import WasteDataset
from utils.transforms import Resize, ToTensor, Normalize

import argparse
import yaml

def load_data(data_config):
    train_dataset = WasteDataset(data_config['train'], transform=transforms.Compose([
        Resize(640),
        ToTensor(),
        Normalize()
    ]))
    val_dataset = WasteDataset(data_config['val'], transform=transforms.Compose([
        Resize(640),
        ToTensor(),
        Normalize()
    ]))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    return train_loader, val_loader

def train(model, train_loader, val_loader, device, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            
            accuracy = total_correct / total_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")
    
    torch.save(model.state_dict(), 'models/model.pth')

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader = load_data(config['data'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv5(num_classes=len(config['data']['names'])).to(device)
    
    train(model, train_loader, val_loader, device, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data/waste.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    main(args)


-----------------------------------------------------------
Ich mache dieses Tutorial: https://www.youtube.com/watch?v=QtsI0TnwDZs&t=469s&ab_channel=Ultralytics
How to Extract the Outputs from Ultralytics YOLOv8 Model for Custom Projects | Episode 5


import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = "cpu"  # Use CPU instead of CUDA
        print("Using Device: ", self.device)
        self.model = self.load_model()

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8 model
        model.fuse()
        return model

    def predict(self, frame):
        try:
            results = self.model(frame)
        except NotImplementedError:
            # Fallback to CPU if CUDA backend fails
            self.model.to("cpu")
            results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xxyys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            print(boxes)
            
            xyxy = boxes.xyxy
            
            xxyys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        for xxyy in xxyys:
            for box in xxyy:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        return results[0].plot(), xxyys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        
        while True:
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame, xxyys, confidences, class_ids = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1 / (end_time - start_time)
            
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()

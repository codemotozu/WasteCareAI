import torch  # Import PyTorch library.  # Deutsch: Importieren der PyTorch-Bibliothek.
from ultralytics import YOLO  # Import YOLO from the ultralytics library.  # Deutsch: YOLO aus der Ultralyics-Bibliothek importieren.
import cv2  # Import OpenCV for image processing.  # Deutsch: OpenCV für Bildverarbeitung importieren.
import numpy as np  # Import NumPy for numerical operations.  # Deutsch: NumPy für numerische Operationen importieren.

def draw_center_point(image, center, radius=5):  # Define a function to draw a center point on an image.  # Deutsch: Funktion definieren, um einen Mittelpunkt auf ein Bild zu zeichnen.
    cv2.circle(image, center, radius + 2, (255, 255, 255), 2)  # Draw a white outline circle.  # Deutsch: Weißen Umrisskreis zeichnen.
    cv2.circle(image, center, radius, (0, 0, 0), -1)  # Draw a black inner circle.  # Deutsch: Schwarzen Innenkreis zeichnen.
    cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 255), 2)  # Draw a horizontal yellow cross line.  # Deutsch: Horizontale gelbe Kreuzlinie zeichnen.
    cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 255, 255), 2)  # Draw a vertical yellow cross line.  # Deutsch: Vertikale gelbe Kreuzlinie zeichnen.

# Ensure CUDA is available and set to use the NVIDIA GPU
print(f"CUDA available: {torch.cuda.is_available()}")  # Check and print if CUDA is available.  # Deutsch: Überprüfen und anzeigen, ob CUDA verfügbar ist.
if torch.cuda.is_available():  # If CUDA is available.  # Deutsch: Wenn CUDA verfügbar ist.
    torch.cuda.set_device(0)  # Set the first CUDA device (GPU) for use.  # Deutsch: Ersten CUDA-Gerät (GPU) verwenden.
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")  # Print the name of the GPU being used.  # Deutsch: Namen der verwendeten GPU ausgeben.

# Load the YOLO model
model = YOLO("C:/Users/rodri/Downloads/WasteSenseAI-BestModel.pt")  # Load the YOLO model from the specified path.  # Deutsch: YOLO-Modell vom angegebenen Pfad laden.

# Optimize for inference
model.fuse()  # Fuse the model layers for faster inference.  # Deutsch: Modellschichten für schnellere Inferenz verschmelzen.
_ = model.to('cuda')  # Move the model to GPU for computation.  # Deutsch: Modell zur Berechnung auf die GPU verschieben.

# Set a smaller batch size and image size to reduce memory usage
batch_size = 80  # Set the batch size for processing.  # Deutsch: Batch-Größe für Verarbeitung festlegen.
img_size = 864  # Set the image size for inference.  # Deutsch: Bildgröße für Inferenz festlegen.

# Initialize video capture
cap = cv2.VideoCapture(0)  # Start video capture from the webcam.  # Deutsch: Videoaufnahme von der Webcam starten.

while cap.isOpened():  # Loop while the video capture is active.  # Deutsch: Schleife, während die Videoaufnahme aktiv ist.
    ret, frame = cap.read()  # Read a frame from the video.  # Deutsch: Ein Frame aus dem Video lesen.
    if not ret:  # If no frame is read, exit the loop.  # Deutsch: Wenn kein Frame gelesen wird, Schleife beenden.
        break
    
    frame = cv2.resize(frame, (img_size, img_size))  # Resize the frame to match the model's input size.  # Deutsch: Frame auf die Eingabegröße des Modells anpassen.
    
    results = model(frame, conf=0.5, imgsz=img_size)  # Run YOLO inference on the frame with specified confidence.  # Deutsch: YOLO-Inferenz auf dem Frame mit angegebener Konfidenz ausführen.
    
    for r in results:  # Loop through the inference results.  # Deutsch: Durch die Inferenz-Ergebnisse iterieren.
        annotated_frame = r.plot()  # Get the frame with annotations.  # Deutsch: Frame mit Annotationen erhalten.
        
        if r.boxes is not None:  # If there are detected objects.  # Deutsch: Wenn Objekte erkannt wurden.
            for box in r.boxes:  # Loop through the detected bounding boxes.  # Deutsch: Durch die erkannten Begrenzungsrahmen iterieren.
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Extract coordinates of the bounding box.  # Deutsch: Koordinaten des Begrenzungsrahmens extrahieren.
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Calculate the center of the bounding box.  # Deutsch: Mittelpunkt des Begrenzungsrahmens berechnen.
                draw_center_point(annotated_frame, center)  # Draw the center point on the annotated frame.  # Deutsch: Mittelpunkt auf das annotierte Frame zeichnen.
        
        cv2.imshow("YOLOv8 Inference InstanceSegmentation WasteSenseAI", annotated_frame)  # Display the annotated frame.  # Deutsch: Das annotierte Frame anzeigen.
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed.  # Deutsch: Schleife beenden, wenn 'q' gedrückt wird.
        break

cap.release()  # Release the video capture resource.  # Deutsch: Die Videoaufnahme-Ressource freigeben.
cv2.destroyAllWindows()  # Close all OpenCV windows.  # Deutsch: Alle OpenCV-Fenster schließen.

# Clear CUDA cache
if torch.cuda.is_available():  # If CUDA is available.  # Deutsch: Wenn CUDA verfügbar ist.
    torch.cuda.empty_cache()  # Clear the GPU memory cache.  # Deutsch: Den GPU-Speicher-Cache leeren.

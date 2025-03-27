import torch
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model with custom weights
yolo_model = YOLO(r'C:\Users\ranaDherya\Desktop\Bone-Fracture\runs\detect\train3\weights\best.pt')

# Number of classes
num_classes = 7

# Load VGG16 without the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Adjust for your number of fracture types
vgg16_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the VGG16 model
vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# YOLO detection function
def detect_fractures(image):
    # Run the YOLO model on the input image
    results = yolo_model(image)
    
    # Extract bounding boxes from the results
    bboxes = results[0].boxes  # Extract the Boxes object from the results
    
    # Get the (x1, y1, x2, y2) coordinates
    bbox_coords = bboxes.xyxy.cpu().numpy()  # Bounding box coordinates
    
    # Get confidence scores and class IDs
    confidences = bboxes.conf.cpu().numpy()  # Confidence scores
    class_ids = bboxes.cls.cpu().numpy()  # Class IDs for the detected objects

    # Return coordinates, confidences, and class IDs
    return zip(bbox_coords, confidences, class_ids)

# Preprocess ROI for VGG16
def preprocess_for_vgg16(roi):
    roi = cv2.resize(roi, (224, 224))  # Resize to VGG16 input size
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    roi = preprocess_input(roi)
    return roi

# Perform detection and classification
def yolo_vgg16_pipeline(image):
    # Step 1: Detect fractures using YOLO
    detections = detect_fractures(image)

    # Step 2: For each detected region, classify using VGG16
    for bbox_coords, conf, class_id in detections:
        x1, y1, x2, y2 = bbox_coords.astype(int)  # Unpack and cast to int the bounding box coordinates

        # Extract Region of Interest (ROI) from the image
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue  # Skip invalid ROIs

        # Preprocess ROI for VGG16
        roi_preprocessed = preprocess_for_vgg16(roi)

        # Run the VGG16 model on the ROI
        predictions = vgg16_model.predict(roi_preprocessed)
        pred_class = np.argmax(predictions)

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {pred_class}, Conf: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Load an example X-ray image
image = cv2.imread(r'/content/drive/MyDrive/bone-fracture-detection/test/images/image1_26_png.rf.ea3697c11878702d0b7728d240e2eb75.jpg')

# Run the detection + classification pipeline
result_image = yolo_vgg16_pipeline(image)

# Use matplotlib to display the result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Fracture Detection')
plt.axis('off')
plt.show()

# Optionally, save the result image
cv2.imwrite(r'/content/drive/MyDrive/bone-fracture-detection/output/detected_fracture3.jpg', result_image)

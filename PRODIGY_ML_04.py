import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import cv2

# Directory paths
dataset_dir = '/Users/amaymalviya/Downloads/hand_gesture_dataset'

# Image parameters
img_width, img_height = 224, 224

# Function to load and preprocess images
def load_images(dataset_dir, img_width, img_height):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for gesture in os.listdir(dataset_dir):
        gesture_dir = os.path.join(dataset_dir, gesture)
        if os.path.isdir(gesture_dir):
            if gesture not in label_map:
                label_map[gesture] = label_counter
                label_counter += 1
            label = label_map[gesture]

            for img_file in os.listdir(gesture_dir):
                img_path = os.path.join(gesture_dir, img_file)
                img = load_img(img_path, target_size=(img_width, img_height))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)

    return np.array(images), np.array(labels), label_map

# Load and preprocess images
images, labels, label_map = load_images(dataset_dir, img_width, img_height)

# Preprocess images for MobileNetV2
images = preprocess_input(images)

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Extract features
features = base_model.predict(images)
features = features.reshape((features.shape[0], -1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = clf.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Classification report
print(classification_report(y_val, y_pred, target_names=list(label_map.keys())))

# Function to preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_width, img_height))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Real-time hand gesture recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = preprocess_frame(frame_rgb)
    features = base_model.predict(processed_frame)
    features = features.reshape((1, -1))
    prediction = clf.predict(features)
    gesture = list(label_map.keys())[prediction[0]]

    # Display the gesture on the frame
    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

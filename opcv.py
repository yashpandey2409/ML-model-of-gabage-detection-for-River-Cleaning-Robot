import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model(r'C:\Users\yashp\Downloads\New folder\opencv\waste_classification_model.h5')

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define CNN model architecture
model = tf.keras.models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Function to preprocess frame
def preprocess_frame(frame):
    # Resize frame to match input shape of the model
    frame = cv2.resize(frame, (150, 150))
    # Normalize pixel values
    frame = frame / 255.0
    # Expand dimensions to create a batch of size 1
    frame = np.expand_dims(frame, axis=0)
    return frame


# Start video capture from the default camera
cap = cv2.VideoCapture(0)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Predict class
        prediction = model.predict(processed_frame)
        
        # Get the predicted class label
        if prediction > 0.5:  # Assuming threshold of 0.5 for binary classification
            label = 'Garbage'
        else:
            label = 'Non-Garbage'
        
        # Overlay prediction on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Garbage Classification', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
    
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
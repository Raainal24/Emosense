import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the pre-trained model
model = load_model("F:\Study\Mini Project 2\emotion_detection_final_acc75.h5")

# Load the Haar cascade classifier for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # Capture frame and return boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # Convert captured image to grayscale

    # Detect faces in the grayscale image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    # Process each detected face
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)  # Draw rectangle around the face
        roi_gray = gray_img[y:y + h, x:x + w]  # Crop region of interest i.e. face area from the image

        # Convert grayscale face image to RGB
        rgb_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

        # Resize and preprocess the image
        resized_img = cv2.resize(rgb_img, (48, 48))
        resized_img = resized_img.astype('float32') / 255.0
        input_img = np.expand_dims(resized_img, axis=0)

        # Make prediction on the input image
        predictions = model.predict(input_img)

        # Get the predicted emotion label
        max_index = np.argmax(predictions[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        predicted_emotion = emotions[max_index]

        # Display the predicted emotion label on the image
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed image with emotion label
    cv2.imshow('Facial Emotion Analysis', test_img)

    # Wait for 'q' key to be pressed to exit
    if cv2.waitKey(10) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

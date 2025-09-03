import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image, ImageTk
import subprocess
import customtkinter as ct
from tkinter import PhotoImage

# Load the pre-trained model
model = load_model("D:/Studies/Mini Project 2/emotion_detection_final_acc75.h5")

# Load the Haar cascade classifier for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the main application window
root = ct.CTk()
root.geometry("1920x1080")
root.title("EmoSense")

# Load the background image
bg_image = Image.open("D:\Studies\Mini Project 2\python\Final Code\\faces1.png")
bg_image = bg_image.resize((1920, 1080))  # Resize the image to fit the window size
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas to place the background image
canvas = ct.CTkCanvas(root, width=1920, height=1080)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=bg_photo)

# Create the label with the image

# Global variables
cap = None
frame = None
label = None
face_cascade = cv2.CascadeClassifier("D:\Studies\Mini Project 2\python\haarcascade_frontalface_default.xml")

def ensure_camera_opened():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return False
    return True

def update_frame():
    global cap, frame, label

    if not ensure_camera_opened():
        return

    # Read a frame from the camera
    ret, frame = cap.read()
    if ret:
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw boxes around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box for faces

        # Convert frame to RGB format and then to a PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update the label with the new image
        label.configure(image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection

        # Schedule the next frame update
        root.after(10, update_frame)
    else:
        # If no frame is read, try reopening the camera
        cap.release()
        cap = None
        print("Error reading frame from camera. Trying to reopen...")
        root.after(500, update_frame)  # Retry after a delay

def start_camera_button_pressed():
    print("Start Camera button clicked ")
    global cap, frame, label
    subprocess.Popen(["python", "D:\Studies\Mini Project 2\python\Appwithmodeltest.py"])

    # Create label for displaying video feed
    label = ct.CTkLabel(root, text=None)
    label.place(relx=0.5, rely=0.5, anchor="center")
    
    update_frame()
    
def detect_emotions_button_pressed():
    print("Detect Emotions button clicked")
    label.destroy()
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
            if  predicted_emotion == "Neutral":
                predicted_emotion =  "Surprise"
            
                
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

start_camera_button = ct.CTkButton(
    root, text="Start Camera", command=start_camera_button_pressed, 
    width=185,
    height=80,
)
start_camera_button.place(relx=0.5, rely=0.78, anchor="center")
start_camera_button.configure(
    bg_color="white",  # Set background color
    fg_color="blue",      # Set text color
    border_width=10,               # Set border width
)

detect_emotions_button = ct.CTkButton(
    root, text="Detect Emotions ", command=detect_emotions_button_pressed, width=185,height=80
)
detect_emotions_button.place(relx=0.5, rely=0.9, anchor="center")
detect_emotions_button.configure(
    bg_color="white",  # Set background color
    fg_color="indigo",      # Set text color
    border_width=7,               # Set border width
)

# Run the main application loop
root.mainloop()

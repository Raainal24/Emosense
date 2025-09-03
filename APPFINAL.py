import customtkinter as ct
import cv2
from PIL import Image, ImageTk
import subprocess
from tkinter import PhotoImage

# (Optional) Set custom themes
#ct.set_appearance_mode("light")  # Or "Dark" for dark mode

# Create the main application window
root = ct.CTk()
root.geometry("1920x1080")
root.title("EmoSense")

# Load the background image
bg_image = Image.open("F:\\Study\\Mini Project 2\\python\\Final Code\\faces1.png")
bg_image = bg_image.resize((1920, 1080))  # Resize the image to fit the window size
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas to place the background image
canvas = ct.CTkCanvas(root, width=1920, height=1080)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=bg_photo)

# Create a title label with appealing formatting
image_path = "F:\\Study\\Mini Project 2\\python\\Final Code\\LOGO.JPG.png"  # Replace this with the path to your image file
image = PhotoImage(file=image_path)

# Create the label with the image
title_label = ct.CTkLabel(
    root,
    image=image,text=None  # Set the image parameter to the loaded image
)
title_label.place(relx=0.5, rely=0.0, anchor="n") 

# Informational label (placeholder)
info_label = ct.CTkLabel(
    root,
    text="USE OUR APP EMOSENSE AND DETECT YOUR EMOTIONS.\n BECAUSE IT IS IMPORTANT TO EXPRESS EVERY EMOTION THAT YOU FEEL .\n SO COME ON LETS DISCOVER THE LANGUAGE  OF EMOTIONS WITH EMOSENSE - YOUR WINDOW INTO THE \n WORLD OF FEELINGS. \n DECODE EXPRESSIONS, AND CONNECT WITH YOUR EMOTIONS.",
    font=("SHOWCARD GOTHIC", 26),
    text_color="WHITE",
    justify="center" # Center-align informational text as well
)
info_label.place(relx=0.5, rely=0.6, anchor="center")

# Global variables
cap = None
frame = None
label = None
face_cascade = cv2.CascadeClassifier("F:\\Study\\Mini Project 2\\python\\haarcascade_frontalface_default.xml")

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
    subprocess.Popen(["python", "F:\\Study\\Mini Project 2\\python\\Final Code\\Appwithmodeltest.py"])

    # Create label for displaying video feed
    label = ct.CTkLabel(root, text=None)
    label.place(relx=0.5, rely=0.5, anchor="center")

    update_frame()

def detect_emotions_button_pressed():
    print("Detect Emotions button clicked (functionality not yet implemented)")

start_camera_button = ct.CTkButton(
    root, text="Start Camera", command=start_camera_button_pressed, 
    width=185,
    height=80,
)
start_camera_button.place(relx=0.5, rely=0.75, anchor="center")
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

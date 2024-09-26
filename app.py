import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model
import camera
from sklearn.exceptions import NotFittedError

class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title

        self.counters = [1, 1]  # Initialize counters for saving images of two classes

        self.model = model.Model()  # Initialize the model

        self.auto_predict = False  # Flag for auto prediction mode

        self.camera = camera.Camera()  # Initialize the camera

        self.init_gui()  # Setup the GUI

        self.delay = 15  # Delay in milliseconds for updating the frame
        self.update()  # Start updating frames

        self.window.attributes("-topmost", True)  # Keep the window on top
        self.window.mainloop()  # Start the Tkinter main loop

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()  # Create a canvas to display the camera feed

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)  # Button to toggle auto prediction mode

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)  # Ask user for class names

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)  # Button to save images for class one

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)  # Button to save images for class two

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)  # Button to train the model

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)  # Button to make a prediction

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)  # Button to reset the application

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)  # Label to display the predicted class

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict  # Toggle auto prediction mode

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")  # Create directories for classes if they don't exist

        cv.imwrite(f'{class_num}/frame{self.counters[class_num - 1]}.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')
        img.thumbnail((150, 150), PIL.Image.LANCZOS)
        img.save(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')  # Save and resize the frame

        self.counters[class_num - 1] += 1  # Increment the counter for the class

    def reset(self):
        for folder in ['1', '2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete all files in the class directories

        self.counters = [1, 1]  # Reset counters
        self.model = model.Model()  # Reinitialize the model
        self.class_label.config(text="CLASS")  # Reset the class label

    def update(self):
        if self.auto_predict:
            try:
                self.predict()
            except NotFittedError:
                print("Model not trained yet.")  # Predict automatically if auto predict is on

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # Update the canvas with the current frame

        self.window.after(self.delay, self.update)  # Schedule the next update

    def predict(self):
        ret, frame = self.camera.get_frame()
        if ret:
            try:
                prediction = self.model.predict(frame)
                if prediction == 1:
                    self.class_label.config(text=self.classname_one)
                    return self.classname_one
                if prediction == 2:
                    self.class_label.config(text=self.classname_two)
                    return self.classname_two  # Predict the class of the current frame and update the label
            except NotFittedError:
                print("Model not trained yet.")



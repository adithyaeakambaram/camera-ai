from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
from sklearn.utils.validation import check_is_fitted

class Model:
    def __init__(self):
        self.model = LinearSVC(dual=False)  # Initialize a Linear Support Vector Classifier

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img_resized = cv.resize(img, (150, 150))
            img_flattened = img_resized.flatten()
            img_list.append(img_flattened)
            class_list.append(1)

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img_resized = cv.resize(img, (150, 150))
            img_flattened = img_resized.flatten()
            img_list.append(img_flattened)
            class_list.append(2)  # Load and preprocess images for both classes

        img_list = np.array(img_list)
        class_list = np.array(class_list)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")  # Train the model with the images

    def predict(self, frame):
        check_is_fitted(self.model)  # Check if the model is trained
        frame_resized = cv.resize(frame, (150, 150))

        if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
            img_gray = cv.cvtColor(frame_resized, cv.COLOR_RGB2GRAY)
        else:
            img_gray = frame_resized  # Convert the frame to grayscale if needed

        img_flattened = img_gray.flatten()
        prediction = self.model.predict([img_flattened])
        return prediction[0]  # Predict the class of the frame







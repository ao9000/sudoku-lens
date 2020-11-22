import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
from helper_functions import sudoku_cells_reduce_noise
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import cv2


# Load trained model
model = tf.keras.models.load_model('models/model.h5')

# Define testing image filename
test_directory = "test"

# Initialize lists to record score
y_pred, y_true = [], []

for file in os.listdir(test_directory):
    # Loop directories only
    if os.path.isdir(os.path.join(test_directory, file)):
        for image in os.listdir(os.path.join(test_directory, file)):
            # Load testing image
            digit = load_img(os.path.join(test_directory, file, image), color_mode="grayscale", target_size=(28, 28, 1),
                             interpolation="nearest")

            # Preprocess image
            # Convert image into np array
            digit = np.asarray(digit)

            # Image thresholding & invert image
            digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)

            # Remove surrounding noise
            digit = sudoku_cells_reduce_noise(digit)

            if digit is not None:
                # Reshape to fit model input
                digit = digit.reshape((1, 28, 28, 1))

                # Make prediction
                prediction = np.argmax(model.predict(digit), axis=-1)[0]+1

                # Record score
                y_true.append(str(file))
                y_pred.append(str(prediction))

                print(f'Predicted:{prediction}, Actual:{file}')

# Print final scores
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

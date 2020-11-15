import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import PIL
from helper_functions import filter_surrounding_noise
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
            digit = load_img(os.path.join(test_directory, file, image), grayscale=True, target_size=(28, 28))

            # Preprocess image
            # Convert image into np array
            digit = np.array(digit)
            # Reshape to be pixels*width*height & convert images into single channel
            digit = digit.reshape((28, 28, 1))

            # Image thresholding
            digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)

            # Remove surrounding noise
            digit = filter_surrounding_noise(digit)

            if digit is not None:
                # Reshape to be pixels*width*height & convert images into single channel
                digit = digit.reshape((1, 28, 28, 1))

                # Make prediction
                # prediction = np.argmax(model.predict(digit_norm), axis=-1)[0]
                prediction = model.predict_classes(digit)[0]

                # Record score
                y_true.append(str(file))
                y_pred.append(str(prediction))

                print(f'Predicted:{prediction}, Actual:{file}')

                if str(file) != str(prediction):
                    print(image)
                    cv2.imshow("digit", digit.reshape((28, 28, 1)))
                    cv2.waitKey(0)

print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

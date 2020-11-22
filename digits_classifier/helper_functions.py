from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import tensorflow as tf
import cv2


def get_mnist_dataset():
    # Load MNIST dataset dataset
    # 60,000 28x28 grayscale Train images of the 10 digits, 10,000 Test images
    # x_train -> images in numpy array, y_train -> corresponding labels
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Remove 0 from train & test datasets, since we do not need them
    # Get all 0 indexes in train
    train_remove_indexes = [index for index, label in enumerate(y_train) if label == 0]

    # Remove items in train where match index
    x_train = np.asarray([image for index, image in enumerate(x_train) if index not in train_remove_indexes])
    y_train = np.asarray([label for index, label in enumerate(y_train) if index not in train_remove_indexes])

    # Remove 0 from train & test datasets, since we do not need them
    # Get all 0 indexes in test
    test_remove_indexes = [index for index, label in enumerate(y_test) if label == 0]

    # Remove items in test where match index
    x_test = np.asarray([image for index, image in enumerate(x_test) if index not in test_remove_indexes])
    y_test = np.asarray([label for index, label in enumerate(y_test) if index not in test_remove_indexes])

    return (x_train, y_train), (x_test, y_test)


def preprocess_train_dataset(data):
    # Reshape to be samples*pixels*width*height & convert images into single channel
    data = data.reshape((data.shape[0], 28, 28, 1))

    # Prepare image for modelling
    # Normalize pixel values 0-1
    data_norm = data.astype('float32')
    data_norm = data_norm / 255.0

    return data_norm


def preprocess_train_label(label):
    # One hot Code
    # Converts a class vector (integers) to binary class matrix
    label = to_categorical(label - 1, 9)

    return label


def sudoku_cells_reduce_noise(digit_inv):
    # Eliminate surrounding noise
    # Detect contours
    cnts, hierarchy = cv2.findContours(digit_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours over 5 pixel square area
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 5]

    # Check if any contour is detected
    if cnts:
        # Sort to largest contour (Digit)
        cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        # Get coordinates, width, height of contour
        x, y, width, height = cv2.boundingRect(cnt)

        # Crop area
        digit_inv = digit_inv[y:y + height, x:x + width]

        # Create a black mat
        new_digit_inv = np.zeros((28, 28), np.uint8)

        # if Digit is too small, enlarge it
        if cv2.contourArea(cnt) < 25:
            # Maintain aspect ratio
            aspect_ratio = 15 / float(height)
            new_dimensions = (int(width * aspect_ratio), 15)
            digit_inv = cv2.resize(digit_inv, new_dimensions, interpolation=cv2.INTER_NEAREST)

            # Update width & height
            height, width = digit_inv.shape

        # Paste detected contour in the middle to center image
        new_digit_inv[14-height//2:14-height//2+height, 14-width//2:14-width//2+width] = digit_inv

        return new_digit_inv
    else:
        # No contour detected
        return None


def build_model():
    # Build the CNN Model for digit classification purposes
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_accuracy_graph(history):
    # Set x axis to 1 step
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot training graphs
    # Accuracy graph
    # Plot
    plt.plot(history.history['accuracy'], color='blue', label="Train accuracy")
    plt.plot(history.history['val_accuracy'], color='orange', label="Val accuracy")
    # Set labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    # Save plot
    plt.savefig("models/accuracy.png")


def plot_loss_graph(history):
    # Set x axis to 1 step
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Loss graph
    # Plot
    plt.plot(history.history['loss'], color='blue', label="Train loss")
    plt.plot(history.history['val_loss'], color='orange', label="Val loss")
    # Set labels
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    # Save plot
    plt.savefig("models/loss.png")

import tensorflow as tf
from helper_functions import build_model, plot_training_graphs, preprocess_dataset, preprocess_label, k_fold_cross_validation


# Load MNIST dataset dataset
# 60,000 28x28 grayscale Train images of the 10 digits, 10,000 Test images
# x_train -> images in numpy array, y_train -> corresponding labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Preprocess dataset
x_train = preprocess_dataset(x_train)
x_test = preprocess_dataset(x_test)

# Preprocess labels
y_train = preprocess_label(y_train)
y_test = preprocess_label(y_test)

# Build model
model = build_model()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Save model to disk
model.save("models/model.h5")

# Generate training graphs
plot_training_graphs(history)

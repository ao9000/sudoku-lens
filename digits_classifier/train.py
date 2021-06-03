"""
    Training script to train the model used for the digit classifier
"""

from helper_functions import build_model, plot_accuracy_graph, plot_loss_graph, preprocess_train_dataset, preprocess_train_label, get_mnist_dataset

# Load MNIST dataset dataset without 0 data
(x_train, y_train), (x_test, y_test) = get_mnist_dataset()


# Preprocess dataset
x_train = preprocess_train_dataset(x_train)
x_test = preprocess_train_dataset(x_test)

# Preprocess labels
y_train = preprocess_train_label(y_train)
y_test = preprocess_train_label(y_test)

# Build model
model = build_model()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Save model to disk
model.save("models/model.h5")

# Generate training graphs
plot_accuracy_graph(history)
plot_loss_graph(history)

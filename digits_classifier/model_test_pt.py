from digits_classifier import sudoku_cells_reduce_noise
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import cv2
from helper_functions_pt import MNISTClassifier, get_mnist_transform
import torch
from PIL import Image


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
state_dict = torch.load("models/pt_cnn/ft_model_epoch15.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Print model summary
print("Model summary")
print(model)

# Define testing image filename
test_directory = "test"

# Initialize lists to record score
y_pred, y_true = [], []

for file in os.listdir(test_directory):
    # Loop directories only
    if os.path.isdir(os.path.join(test_directory, file)):
        for image in os.listdir(os.path.join(test_directory, file)):
            # Load testing image
            digit = cv2.imread(os.path.join(test_directory, file, image), cv2.IMREAD_GRAYSCALE)

            # Preprocess image
            # Image thresholding & invert image
            digit_inv = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)

            # Remove surrounding noise
            denoised_digit = sudoku_cells_reduce_noise(digit_inv)

            if denoised_digit is not None:
                digit = Image.fromarray(denoised_digit)
                # Reshape to fit model input, [1,28,28]
                digit_tensor = get_mnist_transform()(digit)
                # Add batch dim, send to device
                digit_tensor = digit_tensor.unsqueeze(0).to(device)

                # Make prediction
                with torch.no_grad():
                    logits = model(digit_tensor)
                    prediction = torch.argmax(logits, dim=1).item()+1

                # Save fail detections
                if str(file) != str(prediction):
                    cv2.imwrite(f"fails/{image} Predicted:{prediction}.png", denoised_digit.reshape((28,28,1)))

                # Record score
                y_true.append(str(file))
                y_pred.append(str(prediction))

                print(f'Predicted:{prediction}, Actual:{file}')

# Print final scores
print(f"Total images: {len(y_pred)}")
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

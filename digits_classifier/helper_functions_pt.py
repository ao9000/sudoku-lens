import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from helper_functions_tf import sudoku_cells_reduce_noise
from PIL import Image
import numpy as np


def get_mnist_transform():
    transform = torchvision.transforms.Compose([
        # Convert to pytorch image tensor
        torchvision.transforms.ToTensor(),
        # Mean and std of mnist digit dataset
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform


def get_mnist_dataset_loader(save_path, train, transform, batch_size):
    # Target transform = remap the remaining labels from 1-9 to 0-8
    dataset = torchvision.datasets.MNIST(save_path,
                                         train=train,
                                         download=True,
                                         transform=transform,
                                         target_transform=lambda y: y - 1)

    # Remove class 0, since we do not need them
    nonzero_mask = (dataset.targets != 0)
    nonzero_indices = nonzero_mask.nonzero(as_tuple=False).squeeze().tolist()
    filtered = torch.utils.data.Subset(dataset, nonzero_indices)

    return torch.utils.data.DataLoader(
        filtered,
        batch_size=batch_size,
        shuffle=True if train else False,
        pin_memory=True
    )

# Model definition
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 9) # Change from 10 classes (0-9) to 9 classes (1-9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def build_model(optimizer_name, **kwargs):
    # Init model network
    model = MNISTClassifier()
    name = optimizer_name.lower()
    if name == "adam":
        # All keyword args (lr, weight_decay, betas, etc.) go into Adam(...)
        optimizer = optim.Adam(model.parameters(), **kwargs)

    elif name == "sgd":
        # For SGD you might want at least lr and optionally momentum, etc.
        optimizer = optim.SGD(model.parameters(), **kwargs)

    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **kwargs)

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}. "f"Choose from 'adam', 'sgd', 'rmsprop'")
    return model, optimizer

def plot_training_graph(history):
    fig = plt.figure()
    plt.plot(history['train_samples_seen'], history['train_loss'], color='blue')
    plt.scatter(history['test_samples_seen'], history['test_loss'], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    # Save plot (and close)
    plt.savefig("models/train_loss.png")
    plt.close(fig)

def plot_test_graph(history):
    x = history['test_samples_seen']
    loss_y = history['test_loss']
    acc_y = history['test_acc']

    fig, ax1 = plt.subplots()

    # Plot test loss on the left y-axis
    color_loss = 'tab:blue'
    ax1.set_xlabel('Number of Samples Seen')
    ax1.set_ylabel('Test Loss', color=color_loss)
    ax1.plot(x, loss_y, color=color_loss, marker='o', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    color_acc = 'tab:orange'
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.plot(x, acc_y, color=color_acc, marker='s', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 1)  # Ensure the accuracy axis is between 0 and 1

    # Title and legends
    plt.title('Test Loss and Accuracy vs. Samples Seen')
    fig.tight_layout()  # Prevent label overlap

    # Show legend manually combining both axes
    lines_loss, labels_loss = ax1.get_legend_handles_labels()
    lines_acc, labels_acc = ax2.get_legend_handles_labels()
    ax1.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc='upper right')

    plt.savefig("models/test_loss_acc.png")
    plt.close(fig)


def get_custom_test_dataset_loader(dataset_path, batch_size):
    def loader(img_path):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        digit_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)
        denoised_digit = sudoku_cells_reduce_noise(digit_inv)
        if denoised_digit is not None:
            return Image.fromarray(denoised_digit)
        return digit_inv

    test_dataset = ImageFolder(
        root=dataset_path,
        loader=loader,
        transform=get_mnist_transform(),
    )

    return DataLoader(test_dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True,
                      )


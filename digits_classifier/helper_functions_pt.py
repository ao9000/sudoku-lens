import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from digits_classifier import sudoku_cells_reduce_noise
from PIL import Image
from matplotlib.ticker import MultipleLocator


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
    """
    Source: https://nextjournal.com/gkoehler/pytorch-mnist
    """
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


def plot_accuracy_graph(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_acc'], label="Train Accuracy", color='blue')
    plt.plot(epochs, history['mnist_test_acc'], label="Mnist Test Accuracy", color='orange')
    plt.plot(epochs, history['sudoku_test_acc'], label="Sudoku Digits Test Accuracy", color='green')

    # Labels, title, legend
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pt_cnn/accuracy.png")
    plt.close(fig)


def plot_loss_graph(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_loss'], label="Train Loss", color='blue')
    plt.plot(epochs, history['mnist_test_loss'], label="Mnist Test Loss", color='orange')
    plt.plot(epochs, history['sudoku_test_loss'], label="Custom Test Loss", color='green')

    # Labels, title, legend
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pt_cnn/loss.png")
    plt.close(fig)


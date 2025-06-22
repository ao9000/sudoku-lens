import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from digits_classifier import sudoku_cells_reduce_noise
from PIL import Image
from matplotlib.ticker import MultipleLocator
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np


def get_mnist_emnist_mean_std():
    # Mean and std of combined MNIST and EMNIST datasets
    # For digits 1-9
    return (0.159903,), (0.323860,)

def get_mnist_transform():
    transform = torchvision.transforms.Compose([
        # Convert to pytorch image tensor
        torchvision.transforms.ToTensor(),
        # Mean and std of mnist digit dataset
        torchvision.transforms.Normalize(get_mnist_emnist_mean_std()[0], get_mnist_emnist_mean_std()[1]),
    ])
    return transform


def get_mnist_emnist_dataset_loader(save_path, train, batch_size):
    def deskew_pil(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        m = cv2.moments(arr)
        if abs(m['mu02']) < 1e-2:
            return img
        skew = m['mu11'] / m['mu02']
        M = np.array([[1, skew, -0.5 * 28 * skew],
                      [0, 1, 0]], dtype=np.float32)
        deskewed = cv2.warpAffine(arr, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return Image.fromarray(deskewed)

    # Get mean and std for normalization
    mean, std = get_mnist_emnist_mean_std()

    # Build two pipelines
    #    - train_pipeline: random aug → deskew → toTensor → normalize
    #    - eval_pipeline: deskew → toTensor → normalize
    train_pipeline = T.Compose([
        # 50% chance to randomly rotate ±10°
        T.RandomApply([T.RandomRotation(10)], p=0.5),

        # 50% chance to randomly translate up to 10% (x/y)
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),

        # 50% chance to randomly scale between 0.9×–1.1×
        T.RandomApply([T.RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5),

        # 50% chance to randomly shear ±5°
        T.RandomApply([T.RandomAffine(degrees=0, shear=5)], p=0.5),

        # then always deskew → toTensor → normalize
        T.Lambda(deskew_pil),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # No augmentations
    eval_pipeline = T.Compose([
        T.Lambda(deskew_pil),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    mnist_transform  = train_pipeline if train else eval_pipeline

    emnist_base = [T.Lambda(lambda img: TF.hflip(TF.rotate(img, -90)))]
    emnist_pipeline = (emnist_base + (train_pipeline.transforms if train else eval_pipeline.transforms))
    emnist_transform = T.Compose(emnist_pipeline)

    # MNIST dataset 1-9
    # Target transform = remap the remaining labels from 1-9 to 0-8
    mnist_ds = torchvision.datasets.MNIST(save_path,
                                         train=train,
                                         download=True,
                                         transform=mnist_transform,
                                         target_transform=lambda y: y - 1)

    # Remove class 0, since we do not need them
    nonzero_mask = (mnist_ds.targets != 0)
    nonzero_indices = nonzero_mask.nonzero(as_tuple=False).squeeze().tolist()
    mnist_filtered = torch.utils.data.Subset(mnist_ds, nonzero_indices)


    # EMNIST dataset 1-9
    emnist_ds = torchvision.datasets.EMNIST(
        root=save_path,
        split='digits',
        train=train,
        download=True,
        transform=emnist_transform,
        target_transform=lambda y: y - 1  # same remapping
    )

    # Build a mask selecting only original labels 1–9
    mask = (emnist_ds.targets >= 1) & (emnist_ds.targets <= 9)
    indices = mask.nonzero(as_tuple=False).squeeze().tolist()
    emnist_filtered = torch.utils.data.Subset(emnist_ds, indices)

    # Combine into one dataset
    combined = ConcatDataset([mnist_filtered, emnist_filtered])
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=train,
        pin_memory=True
    )

# Version 1
# # Model definition
# class MNISTClassifier(nn.Module):
#     """
#     Source: https://nextjournal.com/gkoehler/pytorch-mnist
#     """
#     def __init__(self):
#         super(MNISTClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 9) # Change from 10 classes (0-9) to 9 classes (1-9)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

# Model definition
class MNISTClassifier(nn.Module):
    """
    v1 Source: https://nextjournal.com/gkoehler/pytorch-mnist
    v2 Source: https://github.com/PyTorch/examples/blob/main/mnist/main.py
    Had to find better model due to not able to make mistake on android app
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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

    elif name == "adadelta":
        # Adadelta typically only takes lr and rho; defaults mirror PyTorch example
        optimizer = optim.Adadelta(model.parameters(), **kwargs)

    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name!r}. "
            f"Choose from 'adam', 'sgd', 'rmsprop', 'adadelta'"
        )

    return model, optimizer


def get_custom_test_dataset_loader(dataset_path, train, batch_size):
    # train = true = 80% split
    # train = false = 20% split
    # train = None = 100% of the data
    def loader(img_path):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        digit_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)
        denoised_digit = sudoku_cells_reduce_noise(digit_inv)
        if denoised_digit is not None:
            return Image.fromarray(denoised_digit)
        raise RuntimeError("Bad data")

    test_dataset = ImageFolder(
        root=dataset_path,
        loader=loader,
        transform=get_mnist_transform(),
    )

    # Calculate train test split if needed
    if train is None:
        # Return all data
        return DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True if train else False,
                          pin_memory=True,
                          )
    # Train is either true or false
    # Calculate train test split
    split_ratio = 0.8
    total_len = len(test_dataset)
    train_len = int(total_len * split_ratio)
    test_len  = total_len - train_len
    seed=42

    train_subset, test_subset = random_split(test_dataset,
                                             [train_len, test_len],
                                             generator=torch.Generator().manual_seed(seed))

    print(f"Train len: {len(train_subset)}")
    print(f"Test len: {len(test_subset)}")

    chosen_subset = train_subset if train else test_subset

    return DataLoader(
        chosen_subset,
        batch_size=batch_size,
        shuffle=(True if train else False),
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


def plot_accuracy_graph_ft(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_acc'], label="Sudoku Train Accuracy", color='blue')
    plt.plot(epochs, history['test_acc'], label="Sudoku Test Accuracy", color='orange')

    # Labels, title, legend
    plt.title('Fine-tuning Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pt_cnn/ft_accuracy.png")
    plt.close(fig)


def plot_loss_graph_ft(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_loss'], label="Sudoku Train Loss", color='blue')
    plt.plot(epochs, history['test_loss'], label="Sudoku Test Loss", color='orange')

    # Labels, title, legend
    plt.title('Fine-tuning Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pt_cnn/ft_loss.png")
    plt.close(fig)

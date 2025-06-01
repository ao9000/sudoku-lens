import torch
import torch.nn.functional as F
from helper_functions_pt import get_mnist_dataset_loader, build_model, plot_training_graph, plot_test_graph, get_mnist_transform, get_custom_test_dataset_loader
from tqdm import tqdm


# Constants
train_epochs = 10
batch_size=32
log_batch_interval = 150

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load dataset
transform = get_mnist_transform()
train_loader = get_mnist_dataset_loader('train', True, transform, batch_size)
val_loader = get_mnist_dataset_loader('val', False, transform, batch_size)


# Building model
model, optimizer = build_model(
    optimizer_name="sgd",
    lr=0.01,
    momentum=0.9,
)

# Print model summary
print("Model summary")
print(model)

# Keep track of training progress
history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
}

def train_one_epoch(model, optimizer, train_loader, epoch):
    # Switch model to train mode
    model.train()

    # Logging stats
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # Loop all minibatch training data
    prog_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for data, target in prog_bar:
    # for batch_idx, (data, target) in enumerate(train_loader):
        # Data = [batch_size, channel, width, height
        # batch_idx = index of batch of batch_sized tensors
        # target = ground truth label

        # Accumulates gradients
        optimizer.zero_grad()

        # Forward pass, compute model predictions
        output = model(data)

        # Compute the loss, negative log‐likelihood loss
        loss = F.nll_loss(output, target)

        # Backward Pass: Compute Gradients
        loss.backward()

        # Update Weights using Optimizer
        optimizer.step()

        # Accumulate epoch stats
        batch_size = data.size(0)
        running_loss += loss.item() * batch_size
        pred = output.data.max(1)[1]
        running_correct += pred.eq(target).sum().item()
        total_samples += batch_size

    # Compute epoch stats
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    # Logging
    # Print
    print(f'Epoch: {epoch}, Train Acc: {epoch_acc:.4f}, Train Loss: {epoch_loss:.4f}')
    # Record stats
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
            
    # Checkpoint model weights & optimizer
    torch.save(model.state_dict(), f'models/pt_cnn_model_epoch{epoch}.pth')
    torch.save(optimizer.state_dict(), f'models/pt_cnn_optimizer_epoch{epoch}.pth')

def test(model, test_loader):
    # Run one pass of samples in val_dataloader
    # Switch model to eval mode
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # Disable Gradient Computation
    with torch.no_grad():
        # Loop Over the Test DataLoader samples
        for data, target in test_loader:
            # Forward Pass: Compute Network’s Output
            output = model(data)

            batch_size = data.size(0)
            # Compute sum of Loss for This Batch, negative log‐likelihood loss
            loss  = F.nll_loss(output, target, reduction='mean')
            running_loss += loss.item() * batch_size

            # Get Predicted Class Labels
            pred = output.data.max(1)[1]
            # Sum Correct in pred results tensor
            running_correct += pred.eq(target).sum().item()
            total_samples += batch_size

    # Divide sum loss to get mean loss
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    # Print
    print(f"Epoch {epoch:>2d}, Test Acc: {epoch_acc:.4f}, Test Loss: {epoch_loss:.4f}")

    # Record
    history['test_loss'].append(epoch_loss)
    history['test_acc'].append(epoch_acc)


# Run training
epoch=0
test(model, val_loader)
for epoch in range(1, train_epochs + 1):
    train_one_epoch(model, optimizer, train_loader, epoch)
    print("Mnist test set")
    test(model, val_loader)
    print("Custom test set")
    test(model, get_custom_test_dataset_loader('test', 32))

# Generate training graphs
# plot_training_graph(history)
# plot_test_graph(history)


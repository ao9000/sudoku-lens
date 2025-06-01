import torch
import torchvision
import torch.nn.functional as F
from helper_functions_pt import get_mnist_dataset_loader, build_model, plot_training_graph, plot_test_graph, get_mnist_transform, get_custom_test_dataset_loader


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
    "train_samples_seen": [],
    "test_loss": [],
    "test_samples_seen": [i*len(train_loader.dataset) for i in range(train_epochs+1)],
    "test_acc": [],
}

def train_one_epoch(model, optimizer, train_loader, epoch):
    # Switch model to train mode
    model.train()

    # Loop all minibatch training data
    for batch_idx, (data, target) in enumerate(train_loader):
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

        # Logging
        if batch_idx % log_batch_interval == 0:
            # Print
            print(f'Epoch: {epoch}, Samples: {batch_idx * len(data)}/{len(train_loader.dataset)}, Loss: {loss.item():.3f}')
            # Record training loss & samples seen
            history['train_loss'].append(loss.item())
            history['train_samples_seen'].append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            
    # Checkpoint model weights & optimizer
    torch.save(model.state_dict(), f'models/model_epoch{epoch}.pth')
    torch.save(optimizer.state_dict(), f'models/optimizer_epoch{epoch}.pth')

def test(model, test_loader):
    # Run one pass of samples in val_dataloader
    # Switch model to eval mode
    model.eval()
    epoch_test_loss = 0
    correct = 0

    # Disable Gradient Computation
    with torch.no_grad():
        # Loop Over the Test DataLoader samples
        for data, target in test_loader:
            # Forward Pass: Compute Network’s Output
            output = model(data)

            # Compute sum of Loss for This Batch, negative log‐likelihood loss
            epoch_test_loss += F.nll_loss(output, target, size_average=False).item()

            # Get Predicted Class Labels
            pred = output.data.max(1, keepdim=True)[1]

            # Sum Correct in pred results tensor
            correct += pred.eq(target.data.view_as(pred)).sum()

    # Divide sum loss to get mean loss
    epoch_test_loss /= len(test_loader.dataset)
    # Record
    history['test_loss'].append(epoch_test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    # Print
    print(f'Test, Average loss: {epoch_test_loss:.3f}, Score: {correct}/{len(test_loader.dataset)}, Accuracy: {correct / len(test_loader.dataset):.3f}')


# Run training
test(model, val_loader)
for epoch in range(1, train_epochs + 1):
    train_one_epoch(model, optimizer, train_loader, epoch)
    test(model, val_loader)
    print("Custom test set")
    test(model, get_custom_test_dataset_loader('test', 32))


plot_training_graph(history)
plot_test_graph(history)


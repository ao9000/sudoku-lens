from helper_functions_pt import get_custom_test_dataset_loader, plot_accuracy_graph_ft, plot_loss_graph_ft, build_model
import torch
from tqdm import tqdm
import torch.nn.functional as F

# Constants
epoch=14
batch_size = 64

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# Load checkpoint
ckpt_model_path = "models/pt_cnn/model_epoch14.pth"
# ckpt_optimizer_path = "models/pt_cnn/optimizer_epoch10.pth"
# Load ckpt state
# model, optimizer = build_model(
#     optimizer_name="sgd",
#     lr=lr,
#     momentum=0.9,
# ) # Pre-training was lr=0.01

model, optimizer = build_model(
    optimizer_name="adadelta",
    lr=0.5
)


# Dont load the prev optimizer, since new dataset
state_dict = torch.load(ckpt_model_path)
model.load_state_dict(state_dict)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Load dataset
sudoku_digits_dataset = "test"
train_loader_sudoku = get_custom_test_dataset_loader(
    dataset_path=sudoku_digits_dataset,
    train=True,
    batch_size=batch_size
)
test_loader_sudoku = get_custom_test_dataset_loader(
    dataset_path=sudoku_digits_dataset,
    train=False,
    batch_size=batch_size
)

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
    print(f'Epoch: {epoch:>2d}, Train Acc: {epoch_acc:.4f}, Train Loss: {epoch_loss:.4f}')
    # Record stats
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)

    # Checkpoint model weights & optimizer
    torch.save(model.state_dict(), f'models/pt_cnn/ft_model_epoch{epoch}.pth')
    torch.save(optimizer.state_dict(), f'models/pt_cnn/ft_optimizer_epoch{epoch}.pth')


def test(model, test_loader):
    # Run one pass of samples in val_dataloader
    # Switch model to eval mode
    model.eval()
    # Record stats
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
            loss = F.nll_loss(output, target, reduction='mean')
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

    # # Record
    # history['test_loss'].append(epoch_loss)
    # history['test_acc'].append(epoch_acc)
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # Finetune Training loop
    for epoch in range(1, epoch+1):
        train_one_epoch(model, optimizer, train_loader_sudoku, epoch)
        loss, acc = test(model, test_loader_sudoku)
        history['test_loss'].append(loss)
        history['test_acc'].append(acc)

    # Generate training graphs
    print("Saving graphs")
    plot_accuracy_graph_ft(history)
    plot_loss_graph_ft(history)
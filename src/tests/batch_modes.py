import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *


max_epochs = 30
patience = 5

# !!!
configs = [[1, 0.005], [32, 0.05], [1024, 0.1]]     # [bs, lr]

losses_epochs = np.full((len(configs), max_epochs), np.nan)
losses_iter = np.full((len(configs), max_epochs*50000), np.nan)
accs_epochs = np.full((len(configs), max_epochs), np.nan)
accs_iter = np.full((len(configs), max_epochs*50000), np.nan)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) 
full_train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_size = int(5/6 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)

for i, (batch_size, learning_rate) in enumerate(configs):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cpu")

    torch.manual_seed(0)
    model = nn.Sequential (
        nn.Flatten(),
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    global_iter = 0

    def train(model, device, train_loader, criterion, optimizer, epoch):
        
        global global_iter
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0

        for j, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            epoch_loss += loss.item()
            epoch_acc += accuracy(outputs, labels)

            losses_iter[i, global_iter] = loss.item()
            accs_iter[i, global_iter] = accuracy(outputs, labels)
            global_iter += 1

            if (j+1) % (10000//batch_size) == 0:
                print(f"Epoch: {epoch}, batch: {j+1}, loss: {running_loss/(10000//batch_size):.4f}, accuracy: {running_acc/(10000//batch_size):.4f}")
                running_loss = 0.0
                running_acc = 0.0
        
        print(f'Trained epoch: {epoch}')


        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        losses_epochs[i, epoch-1] = epoch_loss
        accs_epochs[i, epoch-1] = epoch_acc
        return epoch_loss, epoch_acc
        
    def validate(model, device, val_loader, criterion, epoch):

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) 

                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        return val_loss, val_acc

    def test(model, device, test_loader, criterion):

        model.load_state_dict(torch.load(DATA_DIR / "best_model.pth", weights_only=True)) 

        model.eval()
        test_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += accuracy(outputs, labels)

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        print(f'\nLoss: {test_loss:.4f}')
        print(f'Accuracy: {test_acc:.4f}')

        print("\n\nTEST OVER")
        return test_loss, test_acc

    best_val_acc = 0
    counter = 0
    for epoch in range(1, max_epochs + 1):
        
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)
        print(f"Validation: loss = {val_loss}, accuracy = {val_acc}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_model.pth")
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        if epoch == max_epochs:
            print("Stopped at max_epochs")
    print("\n\nTRAINING OVER")

    test_loss, test_acc = test(model, device, test_loader, criterion)


np.savez_compressed(DATA_DIR / "batch_modes.npz", LE=losses_epochs, LI=losses_iter, AE=accs_epochs, AI=accs_iter)


# plotting everything
data = np.load(DATA_DIR / "batch_modes.npz")
losses_epochs = data["LE"]
losses_iter = data["LI"]
accs_epochs = data["AE"]
accs_iter = data["AI"]

batch_sizes = [1, 32, 1024]
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Batch modes")

ax_li = fig.add_subplot(2, 1, 1)
ax_li.grid()
ax_li.set_title("Iteration losses")
ax_li.set_xlabel("Iteration")
ax_li.set_ylabel("Loss")
for i, bs in enumerate(batch_sizes):
    y = losses_iter[i]
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    if bs==1:
        y_plot = y[::50]
        x_plot = x[::50]
    else:
        y_plot = y
        x_plot = x
    ax_li.plot(x_plot, y_plot, '-', label=f'{bs}')
ax_li.legend(title="batch size", loc="upper right")

ax_ai = fig.add_subplot(2, 1, 2)
ax_ai.grid()
ax_ai.set_title("Iteration accuracies")
ax_ai.set_xlabel("Iteration")
ax_ai.set_ylabel("Accuracy")
for i, bs in enumerate(batch_sizes):
    y = accs_iter[i]
    valid = ~np.isnan(y)
    y=y[valid]
    if bs==1:
        y = y[::50]
    x = np.arange(1, len(y)+1)
    ax_ai.plot(x, y, '-', label=f'{bs}')
ax_ai.legend(title="batch size", loc="upper right")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "batch_modes.png")
plt.show()
    

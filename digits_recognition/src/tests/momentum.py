import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dataloader import train_loader, val_loader, test_loader


momentum = 0.9
learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.08]
learning_rate = 0.08

max_epochs = 30
patience = 5

batch_size = 32


losses_epochs = np.full((max_epochs), np.nan)
accs_epochs = np.full((max_epochs), np.nan)

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

# accuracies = np.empty(len(learning_rates))

# for k, learning_rate in enumerate(learning_rates):

criterion = nn.CrossEntropyLoss()
# !!! set momentum
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train(model, device, train_loader, criterion, optimizer, epoch):
    
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    train_loss = 0.0
    train_acc = 0.0

    for j, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

        if (j+1) % 400 == 0:
            print(f"Epoch: {epoch}, batch: {j+1}, loss: {running_loss/400:.4f}, accuracy: {running_acc/400:.4f}")
            running_loss = 0.0
            running_acc = 0.0
    
    print(f'Trained epoch: {epoch}')

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc
    
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
    losses_epochs[epoch-1] = val_loss
    accs_epochs[epoch-1] = val_acc

    print(f"Validation: loss = {val_loss}, accuracy = {val_acc}")
    
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

    # accuracies[k] = test_acc

    lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {lr}, momentum: {momentum}")
    print("\n\nTEST OVER")

    visualize_sample(device, model, test_loader)
    return test_loss, test_acc

best_val_acc = 0
counter = 0
for epoch in range(1, max_epochs + 1):
    
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)

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


acc = np.load(DATA_DIR / "accuracies_momentum.npy")
plt.figure()
plt.scatter(learning_rates, acc)
plt.title("Accuracies through LR")
plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Test accuracy")
plt.savefig(PLOTS_DIR / "accuracies_momentum.png", dpi=300)
plt.show()


np.save(DATA_DIR / "losses_momentum.npy", losses_epochs)
np.save(DATA_DIR / "accs_momentum.npy", accs_epochs)


losses_momentum = np.load(DATA_DIR / "losses_momentum.npy")
accs_momentum = np.load(DATA_DIR / "accs_momentum.npy")
losses = np.load(DATA_DIR / "losses.npy")
accs = np.load(DATA_DIR / "accs.npy")


fig = plt.figure(figsize=(10, 6))
fig.suptitle("Momentum impact")

ax_l = fig.add_subplot(2, 1, 1)
ax_l.grid()
ax_l.set_title("Loss comparison")
ax_l.set_xlabel("Epoch")
ax_l.set_ylabel("Loss")
for i, l in enumerate([losses_momentum, losses]):
    y = l
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    if i==0:
        ax_l.plot(x, y, 'r-')
    else:
        ax_l.plot(x, y, 'y-')

ax_a = fig.add_subplot(2, 1, 2)
ax_a.grid()
ax_a.set_title("Accuracy comparison")
ax_a.set_xlabel("Epoch")
ax_a.set_ylabel("Accuracy")
for i, a in enumerate([accs_momentum, accs]):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    if i==0:
        ax_a.plot(x, y, 'r-')
    else:
        ax_a.plot(x, y, 'y-')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "momentum_comp.png")
plt.show()

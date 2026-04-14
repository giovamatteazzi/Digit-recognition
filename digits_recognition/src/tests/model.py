import torch.nn as nn
import torch.optim as optim

import numpy as np

from utils import *
from dataloader import train_loader, val_loader, test_loader

def zero(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
def normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)
def kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


dropout = False
batch_norm = False
weigh_init = kaiming    # zero, normal, kaiming, xavier
act_func = lambda: nn.Tanh()    # ReLU(), LeakyReLU(negative_slope=0.01), Sigmoid(), Tanh()


torch.manual_seed(0)

# !!! defining the model: Linear, BatchNorm, Act_func, Dropout
layers = [
    nn.Flatten(),
    nn.Linear(28*28, 128),
]
if batch_norm:
    layers.append(nn.BatchNorm1d(128))
layers.append(act_func())
if dropout:
    layers.append(nn.Dropout(0.5))

layers.append(nn.Linear(128, 64))
if batch_norm:
    layers.append(nn.BatchNorm1d(64))
layers.append(act_func())
if dropout:
    layers.append(nn.Dropout(0.5))

layers.append(nn.Linear(64,10))

model = nn.Sequential (*layers)

if weigh_init != kaiming:
    model.apply(weigh_init)


max_epochs = 30
patience = 5

batch_size = 32
learning_rate = 0.05

losses_epochs = np.full((max_epochs), np.nan)
accs_epochs = np.full((max_epochs), np.nan)

device = torch.device("cpu")


model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


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
    print(f"Dropout = {dropout}, Batch Normalization = {batch_norm}")
    print(f"Weight initialization = {weigh_init.__name__}, Activation function = {act_func().__class__.__name__}")
    
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

    print(f"PARAMETERS: Dropout = {dropout}, Batch Normalization = {batch_norm}")
    print(f"Weight initialization = {weigh_init.__name__}, Activation function = {act_func().__class__.__name__}")
    print("\n\nTEST OVER")
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


np.save(DATA_DIR / "losses_config.npy", losses_epochs)
np.save(DATA_DIR / "accs_config.npy", accs_epochs)


# plotting

losses = np.load(DATA_DIR / "losses.npy")
accs = np.load(DATA_DIR / "accs.npy")

losses_config = np.load(DATA_DIR / "losses_config.npy")
accs_config = np.load(DATA_DIR / "accs_config.npy")


fig = plt.figure(figsize=(10, 6))
fig.suptitle("Tanh")

ax_l = fig.add_subplot(2, 1, 1)
ax_l.grid()
ax_l.set_title("Loss comparison")
ax_l.set_xlabel("Epoch")
ax_l.set_ylabel("Loss")
for i, l in enumerate([losses_config, losses]):
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
for i, a in enumerate([accs_config, accs]):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    if i==0:
        ax_a.plot(x, y, 'r-')
    else:
        ax_a.plot(x, y, 'y-')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "tanh.png", dpi=300)
plt.show()
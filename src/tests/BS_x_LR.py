import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *


max_epochs = 20
patience = 2

# !!!
batch_sizes = [1, 32, 64, 256, 1024]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]


heatmap = np.empty((len(batch_sizes), len(learning_rates)))


for i, batch_size in enumerate(batch_sizes):
    for j, learning_rate in enumerate(learning_rates):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) 

        full_train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        train_size = int(5/6 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cpu")

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

        def train(model, device, train_loader, criterion, optimizer, epoch):
            
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            train_loss = 0.0
            train_acc = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
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
        print(f"IT: {len(learning_rates)*i+j+1}, BS: {batch_sizes[i]}, LR: {learning_rates[j]}, ACC: {test_acc}")
        # creating heatmap
        heatmap[i, j] = test_acc*100

print(heatmap)
np.save(DATA_DIR / "heatmap.npy", heatmap)


# plotting
heatmap = np.load(DATA_DIR / "heatmap.npy")
plt.figure()
plt.imshow(heatmap, cmap="Blues", vmin=85, vmax=100)
plt.xticks(range(len(learning_rates)), learning_rates)
plt.yticks(range(len(batch_sizes)), batch_sizes)
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.title("Test Accuracy")
plt.colorbar()
for i in range(len(batch_sizes)):
    for j in range(len(learning_rates)):
        plt.text(j, i, f"{heatmap[i, j]:.4f}", ha='center', va='center')
plt.savefig(PLOTS_DIR / "BS_LR_heatmap.png", dpi = 300)
plt.show()


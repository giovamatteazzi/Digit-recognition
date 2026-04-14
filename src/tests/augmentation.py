import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils import *


learning_rate = 0.05
batch_size = 32
max_epochs = 30
patience = 5


# applicating trasformations to training_set
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,                 # rotation
        translate=(0.05, 0.05),     # translation
        scale=(0.9, 1.1),           # scaling
        shear=5                     # skewing
    ),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x + 0.15 * torch.randn_like(x), 0., 1.)),    # introduces also noise
    transforms.Normalize((0.1307,), (0.3081,))    # proper mean and std for MNIST
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


full_dataset = datasets.MNIST(root='.', train=True, download=True)
num_samples = len(full_dataset)
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_size = int(5/6 * num_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(datasets.MNIST(root='.', train=True, download=False, transform=train_transform), train_indices)
val_dataset = Subset(datasets.MNIST(root='.', train=True, download=False, transform=test_transform), val_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

accs_epochs = np.full((max_epochs), np.nan)
train_accs_epoch = np.full((max_epochs), np.nan)

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
    
        if (i + 1) % 200 == 0:
            print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0 
    print(f'Trained epoch: {epoch}')

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    train_accs_epoch[epoch-1] = train_acc

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

    accs_epochs[epoch-1] = val_acc
    
    return val_loss, val_acc

def test(model, device, test_loader, criterion):

    model.load_state_dict(torch.load(DATA_DIR / "best_model.pth", weights_only=True))

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += accuracy(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f'\nLoss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.4f}')

    print("\n\nTEST OVER")

    visualize_sample(device, model, test_loader)

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

test(model, device, test_loader, criterion)


np.save(DATA_DIR / "accs_aug2.npy", accs_epochs)
np.save(DATA_DIR / "train_accuracy_aug2.npy", train_accs_epoch)


accs_aug1 = np.load(DATA_DIR / "accs_aug1.npy")
accs_aug_train1 = np.load(DATA_DIR / "train_accuracy_aug1.npy")

plt.figure()
plt.title("Augmentation")
plt.xlabel("epoch")
plt.ylabel("accuracy")


y = accs_aug1
valid = ~np.isnan(y)
y=y[valid]
x = np.arange(1, len(y)+1)
plt.plot(x, y, 'b-')

y = accs_aug_train1
valid = ~np.isnan(y)
y=y[valid]
x = np.arange(1, len(y)+1)
plt.plot(x, y, 'g-')

plt.grid()
plt.savefig(PLOTS_DIR / "augmentation.png", dpi=300)
plt.show()

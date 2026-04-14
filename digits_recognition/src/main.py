import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from tests.utils import *



learning_rate = 0.05
batch_size = 32
max_epochs = 30
patience = 5



# reshaping and normalizing dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # proper mean and std for MNIST
])   ## data augmentation


# loading training, validation and test set
full_train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)   # downloads MNIST
train_size = int(5/6 * len(full_train_dataset))     # 50k train - 10k validation
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])     # train and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # test


# defining the device
device = torch.device("cpu")


# defining the model
torch.manual_seed(0)
model = nn.Sequential (
    nn.Flatten(),
    nn.Linear(28*28,128),   ## number of neurons
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10)
) ## dropout, layer-normalization, weight_init, activation_fx
model.to(device)


# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate) ## momentum, weight_decay
## lr scheduler


def train(model, device, train_loader, criterion, optimizer, epoch):
    
    model.train()    # activates dropout and batch normalization
    running_loss = 0.0     # for checking running loss and accuracy during training epochs (every 200 iterations)
    running_acc = 0.0
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()   # reset gradient
        outputs = model(inputs)     # pass forward
        loss = criterion(outputs, labels)
        loss.backward()    # calculates loss and gradient
        optimizer.step()    # effectively updates weights

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
    return train_loss, train_acc
    

def validate(model, device, val_loader, criterion, epoch):

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():    # disactivate gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) 

            val_loss += loss.item()      # doesn't calculates gradient nor update weights
            val_acc += accuracy(outputs, labels)
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    return val_loss, val_acc

def test(model, device, test_loader, criterion):    # analogous to validate, different meaning

    model.load_state_dict(torch.load("data/best_model.pth", weights_only=True))     # loads best model (from early stopping)

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
    visualize_all_stats(all_labels, all_preds, test_loss, test_acc, test_loader)
    save_confmat(all_labels, all_preds)
    visualize_sample(device, model, test_loader)
    draw_interface(model)
    return test_loss, test_acc

# implementing early stopping
best_val_acc = 0
counter = 0
for epoch in range(1, max_epochs + 1):
    
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)
    print(f"Validation: loss = {val_loss}, accuracy = {val_acc}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "data/best_model.pth")    # saves best model seen so far
    else:
        counter += 1
    
    if counter >= patience:     # not improving for {patience} epochs
        print(f"Early stopping triggered at epoch {epoch}")
        break
    if epoch == max_epochs:
        print("Stopped at max_epochs")
print("\n\nTRAINING OVER")

test(model, device, test_loader, criterion)

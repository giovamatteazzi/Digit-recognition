import torch.nn as nn
import torch.optim as optim

from utils import *
from dataloader import train_loader, val_loader, test_loader


max_epochs = 30
patience = 5

batch_size = 32

learning_rate = 0.1


train_accs_epochs = np.full((max_epochs), np.nan)
val_accs_epochs = np.full((max_epochs), np.nan)

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

    train_accs_epochs[epoch-1] = train_acc

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
    val_accs_epochs[epoch-1] = val_acc

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


np.save(DATA_DIR / "train_accs.npy", train_accs_epochs)
np.save(DATA_DIR / "val_accs.npy", val_accs_epochs)

# plotting

accs_train = np.load(DATA_DIR / "train_accs.npy")
accs_val = np.load(DATA_DIR / "val_accs.npy")

plt.figure()
plt.title("Early stopping")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

y = accs_train
valid = ~np.isnan(y)
y=y[valid]
x = np.arange(1, len(y)+1)
plt.plot(x, y, 'g-')

y = accs_val
valid = ~np.isnan(y)
y=y[valid]
x = np.arange(1, len(y)+1)
plt.plot(x, y, 'b-')

plt.grid()
plt.savefig(PLOTS_DIR / "earlystoppping.png", dpi=300)
plt.show()
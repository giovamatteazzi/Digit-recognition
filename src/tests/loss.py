import torch.nn as nn
import torch.optim as optim

from utils import *
from dataloader import train_loader, val_loader, test_loader


weight_decay = 0.0001


max_epochs = 30
patience = 5

batch_size = 32

learning_rate = 0.1


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

# define chosen loss function
# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.L1Loss()
# criterion = torch.nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


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
        # softmax for higher stability
        outputs = torch.softmax(outputs, dim=1)
        # L1 and L2 require labels shaped [batch_size x 10] -> onehot
        labels_onehot = nn.functional.one_hot(labels, num_classes=10).float()
        loss = criterion(outputs, labels_onehot)
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
    
def validate(model, device, val_loader, epoch):

    model.eval()
    val_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            val_acc += accuracy(outputs, labels)
  
    val_acc /= len(val_loader)
    accs_epochs[epoch-1] = val_acc

    print(f"Validation: accuracy = {val_acc}")
    
    return val_acc

def test(model, device, test_loader):

    model.load_state_dict(torch.load(DATA_DIR / "best_model.pth", weights_only=True)) 

    model.eval()
    test_acc = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_acc += accuracy(outputs, labels)

    test_acc /= len(test_loader)

    print(f'Accuracy: {test_acc:.4f}')

    print("\n\nTEST OVER")
    return test_acc

best_val_acc = 0
counter = 0
for epoch in range(1, max_epochs + 1):
    
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
    val_acc = validate(model, device, val_loader, epoch)

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

test_acc = test(model, device, test_loader)


# np.save(DATA_DIR / "accs_L1.npy", accs_epochs)
# np.save(DATA_DIR / "accs_L2.npy", accs_epochs)
np.save(DATA_DIR / "accs_L1reg.npy", accs_epochs)
# np.save(DATA_DIR / "accs_L2reg.npy", accs_epochs)
# np.save(DATA_DIR / "accs_CEreg.npy", accs_epochs)


accs_L1 = np.load(DATA_DIR / "accs_L1.npy")
accs_L2 = np.load(DATA_DIR / "accs_L2.npy")
accs_L1reg = np.load(DATA_DIR / "accs_L1reg.npy")
accs_L2reg = np.load(DATA_DIR / "accs_L2reg.npy")
accs_CEreg = np.load(DATA_DIR / "accs_CEreg.npy")
accs = np.load(DATA_DIR / "accs.npy")


plt.figure()
plt.title("Accuracy comparison")
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")

colors = ['r', 'c', 'y']
labels = ["L1", "L2", "CE"]

for a, c, l in zip([accs_L1, accs_L2, accs], colors, labels):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    plt.plot(x, y,  c + '-', label=l)
plt.legend(loc="upper right")

plt.savefig(PLOTS_DIR / "losses_comp1.png", dpi=300)
plt.show()


fig = plt.figure()
fig.suptitle("Accuracy comparison")
colors = ['y', 'r']

ax1 = fig.add_subplot(3,1,1)
ax1.grid()
ax1.set_title("L1")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
for a, c in zip([accs_L1, accs_L1reg], colors):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    plt.plot(x, y,  c + '-')

ax2 = fig.add_subplot(3,1,2)
ax2.grid()
ax2.set_title("L2")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
for a, c in zip([accs_L2, accs_L2reg], colors):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    plt.plot(x, y,  c + '-')

ax3 = fig.add_subplot(3,1,3)
ax3.grid()
ax3.set_title("CE")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy")
for a, c in zip([accs, accs_CEreg], colors):
    y = a
    valid = ~np.isnan(y)
    y=y[valid]
    x = np.arange(1, len(y)+1)
    plt.plot(x, y,  c + '-')  

plt.tight_layout()
plt.savefig(PLOTS_DIR / "losses_comp2.png", dpi=300)
plt.show()

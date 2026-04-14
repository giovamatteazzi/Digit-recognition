import torch.nn as nn
import torch.optim as optim

from utils import *
from dataloader import train_loader, val_loader, test_loader    # more deterministic now (batch_size is fixed)


max_epochs = 30
patience = 5

batch_size = 32

initial_learning_rate = 0.1
gamma = 0.5
step_size = 5


device = torch.device("cpu")

# LeakyReLU as activation
torch.manual_seed(0)
model = nn.Sequential (
    nn.Flatten(),
    nn.Linear(28*28,128),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Linear(128, 64),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Linear(64,10)
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


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

    print(f"Validation: loss = {val_loss}, accuracy = {val_acc}")
    
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

            _ , preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f'\nLoss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.4f}')

    visualize_all_stats(all_labels, all_preds, test_loss, test_acc, test_loader)
    visualize_sample(device, model, test_loader)
    visualize_mistakes(device, model, test_loader)
    save_confmat(all_labels, all_preds)
    draw_interface(model)
    return test_loss, test_acc


best_val_acc = 0
counter = 0
for epoch in range(1, max_epochs + 1):
    
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)
    lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {lr}")
    scheduler.step()
    
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

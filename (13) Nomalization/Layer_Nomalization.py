import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(777)

train_dataset = datasets.MNIST(
    root="./datas/MNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root="./datas/MNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
loader_test = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28*1, 392)
        self.ln1 = nn.LayerNorm(392)
        self.fc2 = nn.Linear(392, 196)
        self.ln2 = nn.LayerNorm(196)
        self.fc3 = nn.Linear(196, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# Hyperparameters
train_epochs = 10
learning_rate = 0.01

model = Model()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(epoch):
    model.train()

    for data, targets in loader_train:

        y_pred = model(data)
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch {:4d}/{} Loss: {:.6f} Accuracy {:.6f}"
            .format(epoch, train_epochs, loss.item(), test()))
    
def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    return 100. * correct / data_num

for epoch in range(1, train_epochs + 1):
    train(epoch)
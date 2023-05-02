import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7, random_state=0)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train.astype(np.int8), dtype=torch.long)
y_test = torch.tensor(y_test.astype(np.int8), dtype=torch.long)

ds_train = TensorDataset(x_train, y_train)
ds_test = TensorDataset(x_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

# model = nn.Sequential(
#     nn.Linear(28*28*1, 392),
#     nn.ReLU(),
#     nn.Linear(392, 196),
#     nn.ReLU(),
#     nn.Linear(196, 10)
# )

model = nn.Sequential()
model.add_module("fc1", nn.Linear(28*28*1, 392))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2", nn.Linear(392, 196))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(196, 10))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()

    for data, targets in loader_train:

        y_pred = model(data)
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print("Epoch {:4d}/{} Loss: {:.6f}"
    #         .format(epoch, train_epochs, loss.item()))
    
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
    # print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
    #                                                data_num, 100. * correct / data_num))
    return 100. * correct / data_num

train_epochs = 100
for epoch in range(train_epochs):
    train(epoch)

# test()
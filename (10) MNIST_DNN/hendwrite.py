import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digits = load_digits()

x_train = torch.tensor(digits.data, dtype=torch.float32)
y_train = torch.tensor(digits.target, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []

train_epochs = 1000
for epoch in range(train_epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Loss: {:.6f}"
              .format(epoch, train_epochs, loss.item()))
    losses.append(loss.item())

plt.plot(losses)
plt.show()

# rand = random.randint(0, len(digits.images) - 1)

# prediction = model(torch.tensor(digits.data[rand], dtype=torch.float32)).argmax()

# plt.imshow(digits.images[rand], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.title('label: {}, prediction: {}'.format(digits.target[rand], prediction))
# plt.show()
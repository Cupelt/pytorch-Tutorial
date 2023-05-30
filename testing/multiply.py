import torch
import torch.nn as nn
import torch.optim as optim

x1 = torch.randint(1, 10, (10, 1)).float()
x2 = torch.randint(1, 10, (10, 1)).float()

x_train = torch.cat((x1, x2), dim=1)
y_train = x1 * x2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 36)
        self.linear2 = nn.Linear(36, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = x ** 2
        x = self.linear2(x)
        return x

model = Model()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_epochs = 20000
for epoch in range(train_epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Loss: {:.6f}"
              .format(epoch, train_epochs, loss.item()))


print(model(torch.FloatTensor([[64, 64]])))
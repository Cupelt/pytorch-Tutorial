import torch
import torch.nn as nn

x1 = torch.randint(1, 100, (100, 1)).float()
x2 = torch.randint(1, 100, (100, 1)).float()

x_train = torch.cat((x1, x2), dim=1)
y_train = x1 * x2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = Model()
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

train_epochs = 100000
for epoch in range(train_epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Loss: {:.6f}"
              .format(epoch, train_epochs, loss.item()))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]
y_data = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

num_epochs = 20000
for epoch in range(num_epochs + 1):

    eps = 1e-7

    y_pred = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    loss = -(y_train * torch.log(torch.clamp(y_pred, eps, 1-eps)) + (1 - y_train) * torch.log(torch.clamp(1 - y_pred, eps, 1-eps))).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, num_epochs, loss.item()
        ))

prediction = y_pred >= torch.FloatTensor([0.5])
print(prediction)
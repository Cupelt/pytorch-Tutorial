import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999
for epoch in range(nb_epochs + 1):

    hypothesis = x_train * W + b

    loss = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} loss: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), loss.item()
        ))

x_result = torch.FloatTensor([[10]])
print(x_result * W + b)
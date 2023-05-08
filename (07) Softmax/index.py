import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

y_one_hot = torch.zeros(8, 3) # 데이터 갯수, 출력 크기
y_one_hot.scatter_(1, y_train.unsqueeze(dim = 1), 1)

W = torch.zeros((4, 3), requires_grad=True) # 데이터 크기, 출력 크기
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)

num_epochs = 50000
for epoch in range(num_epochs + 1):

    y_pred = F.softmax(x_train.matmul(W) + b, dim=1)
    loss = (y_one_hot * -torch.log(y_pred)).sum(dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} loss: {:.6f}'.format(
            epoch, num_epochs, loss.item()
        ))

print(torch.round(y_pred, decimals=4))
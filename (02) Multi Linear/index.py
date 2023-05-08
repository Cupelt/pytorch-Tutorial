import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1,  0], 
                             [1,  1], 
                             [2,  1], 
                             [1,  2],   
                             [3,  3]])

y_train = torch.FloatTensor([[1], [2], [3], [3], [6]])

#가중치와 편향 선언
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#경사 하강법
optimizer = optim.SGD([W, b], lr=1e-3)

num_epochs = 100000
for epoch in range(num_epochs + 1):

    # H(x) 계산
    y_pred = x_train.matmul(W) + b

    # loss 계산
    loss = torch.mean((y_pred - y_train) ** 2)

    # 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} prediction: {} Loss: {:.6f}'.format(
            epoch, num_epochs, y_pred.squeeze().detach(), loss.item()
        ))

x_result = torch.FloatTensor([[73,  80]])
print(x_result.matmul(W) + b)
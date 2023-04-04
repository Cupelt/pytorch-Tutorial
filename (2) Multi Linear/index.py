import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[73,  80,  75], 
                             [93,  88,  93], 
                             [89,  91,  80], 
                             [96,  98,  100],   
                             [73,  66,  70]])

y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

#가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#경사 하강법
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # loss 계산
    loss = torch.mean((hypothesis - y_train) ** 2)

    # loss로 H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), loss.item()
        ))

x_result = torch.FloatTensor([[73,  80,  75]])
print(x_result.matmul(W) + b)

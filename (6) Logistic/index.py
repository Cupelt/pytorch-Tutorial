import torch
import torch.optim as optim

from data import CustomData

torch.manual_seed(1)

dataset = CustomData()

# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

x_train = torch.FloatTensor(dataset.x_data)
y_train = torch.FloatTensor(dataset.y_data)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    loss = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} loss: {:.6f}'.format(
            epoch, nb_epochs, loss.item()
        ))

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
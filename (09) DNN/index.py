import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]).to(device)
y_train = torch.FloatTensor([[1], [0], [0], [1]]).to(device)

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
).to(device)

optimizer = optim.SGD(model.parameters(), lr=5) # lr=5 기울기 소실 현상이 일어남
criterion = nn.BCELoss().to(device)

num_epochs = 10000
for epoch in range(num_epochs + 1):

    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} loss: {:.6f}'.format(
            epoch, num_epochs, loss.item()
        ))

print(torch.round(model(x_train)))